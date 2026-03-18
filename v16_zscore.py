"""
v16-Z: Z-score нормализация velocity фичей по месяцу.

Идея: velocity фичи (cnt_1h/6h/24h/7d/30d, amt_sum_*) имеют distribution shift
между train (Oct'24-May'25) и test (Jun-Aug'25). Z-score по месяцу убирает
абсолютные значения, оставляя только "насколько аномально для этого месяца".

Подход: НЕ добавляем фичи, а ЗАМЕНЯЕМ velocity фичи на z-score версии.
121 фич остаётся, но velocity → z_velocity.
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from pathlib import Path
from datetime import datetime
import gc, json, time, importlib.util

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_OUT  = ROOT / 'models_v16'
SUBMIT_OUT  = ROOT / 'submissions'

for d in [MODELS_OUT, SUBMIT_OUT]:
    d.mkdir(exist_ok=True)

def _ram_gb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1048576
    except: pass
    return 0.0

def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")} RAM:{_ram_gb():.1f}GB] {msg}', flush=True)

spec9 = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
v9 = importlib.util.module_from_spec(spec9)
spec9.loader.exec_module(v9)

spec14 = importlib.util.spec_from_file_location("v14", ROOT / "pipeline_v14.py")
v14 = importlib.util.module_from_spec(spec14)
spec14.loader.exec_module(v14)

seeds = [42, 123, 777, 2024, 31337]

LGBM_PARAMS = dict(
    objective='binary', metric='average_precision',
    device='gpu', gpu_platform_id=0, gpu_device_id=0,
    learning_rate=0.02,
    num_leaves=127, min_child_samples=200,
    subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
    n_jobs=4, verbose=-1,
)

# Velocity features to z-score normalize per month
VELOCITY_COLS = [
    'cnt_1h', 'cnt_6h', 'cnt_24h', 'cnt_7d', 'cnt_30d',
    'amt_sum_1h', 'amt_sum_6h', 'amt_sum_24h', 'amt_sum_7d', 'amt_sum_30d',
    'secs_since_last',
]


def add_month_column(df):
    """Extract month from event_dttm."""
    if 'month' not in df.columns and 'event_dttm' in df.columns:
        df = df.with_columns(
            pl.col('event_dttm').str.slice(5, 2).cast(pl.Int32).alias('month')
        )
    return df


def compute_monthly_stats(train_df):
    """Compute per-month mean/std for velocity features from training data."""
    train_df = add_month_column(train_df)
    stats = {}
    for col in VELOCITY_COLS:
        if col not in train_df.columns:
            continue
        monthly = (
            train_df
            .group_by('month')
            .agg([
                pl.col(col).mean().alias('mean'),
                pl.col(col).std().alias('std'),
            ])
        )
        stats[col] = monthly
    return stats


def zscore_velocity(df, monthly_stats):
    """Replace velocity features with z-score normalized versions."""
    df = add_month_column(df)
    for col, stats_df in monthly_stats.items():
        if col not in df.columns:
            continue
        # Join month stats
        stats_renamed = stats_df.rename({'mean': f'_m_{col}', 'std': f'_s_{col}'})
        df = df.join(stats_renamed, on='month', how='left')

        # Global fallback for unseen months
        global_mean = df[col].mean()
        global_std = df[col].std()
        if global_std is None or global_std == 0:
            global_std = 1.0

        df = df.with_columns(
            ((pl.col(col) - pl.col(f'_m_{col}').fill_null(global_mean)) /
             pl.col(f'_s_{col}').fill_null(global_std).clip(lower_bound=0.01)
            ).alias(col)
        ).drop([f'_m_{col}', f'_s_{col}'])

    return df


def train_model(name, X_train, y_train, X_val, y_val, n_trees=10000, patience=300):
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    spw = min(spw, 50.0)
    log(f'\n{"="*60}')
    log(f'Training: {name} ({X_train.shape[1]} feats, {n_trees} trees)')
    log(f'{"="*60}')

    preds = np.zeros(len(X_val), dtype=np.float64)
    for seed in seeds:
        m = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed,
            scale_pos_weight=spw, n_estimators=n_trees)
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)])
        p = m.predict_proba(X_val)[:, 1]
        preds += p
        prauc = average_precision_score(y_val, p)
        log(f'  Seed {seed}: iter={m.best_iteration_}, val={prauc:.6f}')
        m.booster_.save_model(str(MODELS_OUT / f'{name}_s{seed}.txt'))
        del m; gc.collect()

    preds /= len(seeds)
    ensemble_prauc = average_precision_score(y_val, preds)
    log(f'  Ensemble ({name}): val={ensemble_prauc:.6f}')
    return ensemble_prauc


if __name__ == '__main__':
    t_start = time.time()
    log('=== V16-Z: Z-SCORE VELOCITY ===')

    # Load profiles & data
    old_deep_profs = pl.read_parquet(FEATURES_IN / 'deep_customer_profiles.parquet')
    old_mcc_profs = pl.read_parquet(FEATURES_IN / 'customer_mcc_profiles.parquet')
    if old_mcc_profs['mcc_code'].dtype != pl.Int32:
        old_mcc_profs = old_mcc_profs.with_columns(pl.col('mcc_code').cast(pl.Int32))
    old_profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())
    val_ids = set(pl.read_parquet(FEATURES_IN / 'val_event_ids.parquet')['event_id'].to_list())

    # Load train
    log('Loading data...')
    train_df = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    train_df = train_df.filter(~pl.col('event_id').is_in(val_ids))

    # Compute monthly stats BEFORE adding features (from raw velocity)
    log('Computing monthly velocity stats...')
    monthly_stats = compute_monthly_stats(train_df)
    for col, st in monthly_stats.items():
        log(f'  {col}: {st.to_pandas().to_dict("records")}')

    # Z-score normalize velocity features
    log('Z-scoring velocity features...')
    train_df = zscore_velocity(train_df, monthly_stats)

    # Now add features (ratios etc will be computed from z-scored velocities)
    train_df = v9.add_features(train_df)
    train_df = v9.add_customer_profiles(train_df, old_profiles)
    train_df = v14.add_anomaly_features(train_df, old_deep_profs, old_mcc_profs)

    # Val
    val_df = pl.read_parquet(FEATURES_IN / 'val_proper.parquet')
    val_df = val_df.with_columns(pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'))
    val_df = zscore_velocity(val_df, monthly_stats)
    val_df = v9.add_features(val_df)
    val_df = v9.add_customer_profiles(val_df, old_profiles)
    val_df = v14.add_anomaly_features(val_df, old_deep_profs, old_mcc_profs)

    base_feats = [c for c in v9.FEATURE_COLS if c in train_df.columns and c in val_df.columns]
    anom_feats = [c for c in v14.ANOMALY_FEATURES if c in train_df.columns and c in val_df.columns]
    all_feats = base_feats + anom_feats

    y_train = train_df['target'].to_numpy().astype(int)
    y_val = val_df['is_fraud'].to_numpy().astype(int)
    log(f'Train: {len(train_df):,}, Val: {len(val_df):,}, Fraud: {y_val.sum()}')
    log(f'Features: {len(all_feats)}')

    X_tr = train_df.select(all_feats).to_pandas().astype(np.float32)
    X_va = val_df.select(all_feats).to_pandas().astype(np.float32)

    prauc = train_model('Z_zscore_velocity', X_tr, y_train, X_va, y_val)

    # Generate submission
    log('\n=== SUBMIT ===')
    test_df = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    test_df = test_df.unique(subset=['event_id'], keep='first')
    test_df = zscore_velocity(test_df, monthly_stats)
    test_df = v9.add_features(test_df)
    test_df = v9.add_customer_profiles(test_df, old_profiles)
    test_df = v14.add_anomaly_features(test_df, old_deep_profs, old_mcc_profs)

    available = [f for f in all_feats if f in test_df.columns]
    X_test = test_df.select(available).to_pandas().astype(np.float32)
    event_ids = test_df['event_id'].to_numpy()

    preds = np.mean([
        lgb.Booster(model_file=str(MODELS_OUT / f'Z_zscore_velocity_s{s}.txt')).predict(X_test)
        for s in seeds], axis=0)

    sample = pl.read_csv(DATA / 'sample_submit.csv')
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    sub = pl.DataFrame({'event_id': event_ids, 'predict': preds.astype(np.float64)})
    sub = sample.select('event_id').join(sub, on='event_id', how='left')
    n_null = sub['predict'].is_null().sum()
    if n_null > 0:
        sub = sub.with_columns(pl.col('predict').fill_null(sub['predict'].drop_nulls().median()))

    path = SUBMIT_OUT / f'submit_v16_Z_zscore_{ts}.csv'
    sub.write_csv(path)
    log(f'Saved: {path.name}')
    log(f'Val: {prauc:.6f}, v14-C ref: 0.044')
    log(f'Time: {(time.time()-t_start)/60:.0f} min')
    log('DONE')
