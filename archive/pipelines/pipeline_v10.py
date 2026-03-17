"""
Pipeline v10: Proper Validation + Iterative Improvement.

KEY INSIGHT: val didn't correlate with LB because our val split (random rows)
doesn't match test (one random day per customer). Fix this first.

Val strategy:
  - Pick one random day per customer from full 85M rows
  - Extract ALL their ops on that day → realistic test-like val set
  - Train on remaining data
  - PR-AUC on this val should correlate with LB

Then iterate model improvements with trusted val signal.
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from pathlib import Path
from datetime import datetime
import gc, json, time, os

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
CHUNKS_DIR  = FEATURES_IN / '_tmp_train'
MODELS_OUT  = ROOT / 'models_v10'
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

# Import feature engineering from v9
import importlib.util
spec = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
v9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9)

LGBM_PARAMS = dict(
    objective='binary', metric='average_precision',
    device='gpu', gpu_platform_id=0, gpu_device_id=0,
    learning_rate=0.02,
    num_leaves=127, min_child_samples=200,
    subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
    n_jobs=4, verbose=-1,
)

seeds = [42, 123, 777, 2024, 31337]


# ═══════════════════════════════════════════════════════
# Step 1: Build proper val split from chunks
# ═══════════════════════════════════════════════════════

def build_val_split(val_seed=42):
    """
    Mimic test: pick one random day per customer from full 85M rows.
    Returns val DataFrame with all features.
    """
    val_path = FEATURES_IN / 'val_proper.parquet'
    val_ids_path = FEATURES_IN / 'val_event_ids.parquet'

    if val_path.exists() and val_ids_path.exists():
        log(f'Val split already exists: {val_path}')
        return pl.read_parquet(val_path), pl.read_parquet(val_ids_path)

    chunk_files = sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith('.parquet')])
    log(f'Building proper val from {len(chunk_files)} chunks...')

    # Pass 1: Collect all (customer_id, date) pairs
    log('Pass 1: collecting customer dates...')
    all_cust_dates = []
    for i, cf in enumerate(chunk_files):
        path = CHUNKS_DIR / cf
        df = pl.read_parquet(path, columns=['customer_id', 'event_dttm'])
        df = df.with_columns(pl.col('event_dttm').cast(pl.Date).alias('date'))
        cust_dates = df.select(['customer_id', 'date']).unique()
        all_cust_dates.append(cust_dates)
        if (i + 1) % 5 == 0:
            log(f'  Chunk {i+1}/{len(chunk_files)}, RAM: {_ram_gb():.1f}GB')
        del df; gc.collect()

    cust_dates = pl.concat(all_cust_dates).unique()
    del all_cust_dates; gc.collect()
    log(f'Unique (customer, date) pairs: {len(cust_dates):,}')

    # Pick one random date per customer
    np.random.seed(val_seed)
    cust_groups = cust_dates.group_by('customer_id').agg(pl.col('date').alias('dates'))
    val_picks = []
    for row in cust_groups.iter_rows():
        cid, dates = row
        pick = dates[np.random.randint(len(dates))]
        val_picks.append((cid, pick))

    val_keys = pl.DataFrame(val_picks, schema={'customer_id': pl.Int64, 'date': pl.Date})
    n_customers = len(val_keys)
    log(f'Val: {n_customers:,} customers, one day each')
    del cust_dates, cust_groups; gc.collect()

    # Pass 2: Extract val rows from chunks
    log('Pass 2: extracting val rows...')
    val_parts = []
    for i, cf in enumerate(chunk_files):
        path = CHUNKS_DIR / cf
        df = pl.read_parquet(path)
        df = df.with_columns(pl.col('event_dttm').cast(pl.Date).alias('_date'))
        # Join with val_keys to filter
        val_rows = df.join(val_keys, left_on=['customer_id', '_date'], right_on=['customer_id', 'date'], how='inner')
        val_rows = val_rows.drop('_date')
        if len(val_rows) > 0:
            val_parts.append(val_rows)
        if (i + 1) % 5 == 0:
            log(f'  Chunk {i+1}/{len(chunk_files)}: {len(val_rows):,} val rows, RAM: {_ram_gb():.1f}GB')
        del df, val_rows; gc.collect()

    val_df = pl.concat(val_parts)
    del val_parts; gc.collect()
    log(f'Val set: {len(val_df):,} rows, {val_df["customer_id"].n_unique():,} customers')

    # Save val event_ids for filtering train
    val_ids = val_df.select(['event_id'])
    val_ids.write_parquet(val_ids_path)
    val_df.write_parquet(val_path)
    log(f'Saved: {val_path}')

    return val_df, val_ids


# ═══════════════════════════════════════════════════════
# Step 2: Build train set (assembled data minus val)
# ═══════════════════════════════════════════════════════

def build_train_val_sets():
    """Load train (from assembled) minus val rows, and val set."""
    val_df, val_ids = build_val_split()

    # Join labels to val
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    label_ids = set(labels['event_id'].to_list())
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())

    # Val labels: fraud=1, everything else=0
    val_df = val_df.with_columns(
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud')
    )
    n_val_fraud = (val_df['is_fraud'] == 1).sum()
    n_val_total = len(val_df)
    log(f'Val: {n_val_total:,} rows, {n_val_fraud:,} fraud ({100*n_val_fraud/n_val_total:.3f}%)')

    # Train: assembled data minus val event_ids
    val_id_set = set(val_ids['event_id'].to_list())
    train_df = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    n_before = len(train_df)
    train_df = train_df.filter(~pl.col('event_id').is_in(val_id_set))
    n_removed = n_before - len(train_df)
    log(f'Train: {len(train_df):,} rows (removed {n_removed:,} val-overlap rows)')

    return train_df, val_df, labels


# ═══════════════════════════════════════════════════════
# Step 3: Train and evaluate
# ═══════════════════════════════════════════════════════

def train_and_evaluate():
    t_start = time.time()
    log('=== Pipeline v10: Proper Validation ===')

    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    train_df, val_df, labels = build_train_val_sets()

    # Feature engineering
    train_df = v9.add_features(train_df)
    train_df = v9.add_customer_profiles(train_df, profiles)
    val_df = v9.add_features(val_df)
    val_df = v9.add_customer_profiles(val_df, profiles)

    available_feats = [c for c in v9.FEATURE_COLS if c in train_df.columns and c in val_df.columns]
    log(f'Features: {len(available_feats)}')

    X_train = train_df.select(available_feats).to_pandas().astype(np.float32)
    y_train = train_df['target'].to_numpy().astype(int)

    X_val = val_df.select(available_feats).to_pandas().astype(np.float32)
    y_val = val_df['is_fraud'].to_numpy().astype(int)

    log(f'Train: {X_train.shape}, fraud rate: {y_train.mean():.4f}')
    log(f'Val: {X_val.shape}, fraud rate: {y_val.mean():.6f}')

    del train_df; gc.collect()

    # ── Method 1: Standard binary (same as v8) ──
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    log(f'\n--- Method 1: Standard binary, spw={spw:.1f} ---')

    preds_m1 = np.zeros(len(X_val), dtype=np.float64)
    for i, seed in enumerate(seeds):
        m = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed,
            scale_pos_weight=spw, n_estimators=10000)
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)])
        best_iter = m.best_iteration_
        preds_m1 += m.predict_proba(X_val)[:, 1]
        prauc = average_precision_score(y_val, m.predict_proba(X_val)[:, 1])
        log(f'  Seed {seed}: iter={best_iter}, val PR-AUC={prauc:.6f}')
        m.booster_.save_model(str(MODELS_OUT / f'm1_lgbm_s{seed}.txt'))
        del m; gc.collect()
    preds_m1 /= len(seeds)
    prauc_m1 = average_precision_score(y_val, preds_m1)
    log(f'Method 1 ensemble: val PR-AUC = {prauc_m1:.6f}')

    # ── Method 2: Higher learning rate, more regularization ──
    log(f'\n--- Method 2: lr=0.05, more regularization ---')
    params_m2 = {**LGBM_PARAMS, 'learning_rate': 0.05,
                 'reg_alpha': 2.0, 'reg_lambda': 10.0,
                 'num_leaves': 63, 'min_child_samples': 500,
                 'colsample_bytree': 0.4}

    preds_m2 = np.zeros(len(X_val), dtype=np.float64)
    for i, seed in enumerate(seeds):
        m = lgb.LGBMClassifier(**params_m2, random_state=seed,
            scale_pos_weight=spw, n_estimators=10000)
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        best_iter = m.best_iteration_
        preds_m2 += m.predict_proba(X_val)[:, 1]
        prauc = average_precision_score(y_val, m.predict_proba(X_val)[:, 1])
        log(f'  Seed {seed}: iter={best_iter}, val PR-AUC={prauc:.6f}')
        m.booster_.save_model(str(MODELS_OUT / f'm2_lgbm_s{seed}.txt'))
        del m; gc.collect()
    preds_m2 /= len(seeds)
    prauc_m2 = average_precision_score(y_val, preds_m2)
    log(f'Method 2 ensemble: val PR-AUC = {prauc_m2:.6f}')

    # ── Method 3: Rank blend of M1 + M2 ──
    preds_blend = 0.5 * rankdata(preds_m1) + 0.5 * rankdata(preds_m2)
    prauc_blend = average_precision_score(y_val, preds_blend)
    log(f'\nMethod 3 (M1+M2 blend): val PR-AUC = {prauc_blend:.6f}')

    # Save val results
    results = {
        'method_1_standard': prauc_m1,
        'method_2_regularized': prauc_m2,
        'method_3_blend': prauc_blend,
        'val_rows': len(X_val),
        'val_fraud': int(y_val.sum()),
        'val_fraud_rate': float(y_val.mean()),
    }
    with open(MODELS_OUT / 'val_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    log(f'\nResults saved to {MODELS_OUT}/val_results.json')

    # Print summary
    log('\n=== SUMMARY ===')
    log(f'Method 1 (standard):     val PR-AUC = {prauc_m1:.6f}')
    log(f'Method 2 (regularized):  val PR-AUC = {prauc_m2:.6f}')
    log(f'Method 3 (M1+M2 blend):  val PR-AUC = {prauc_blend:.6f}')
    log(f'Val set: {len(X_val):,} rows, {int(y_val.sum()):,} fraud')
    log(f'Total time: {(time.time()-t_start)/60:.1f} min')

    return results, X_val, y_val, available_feats


def generate_submissions(available_feats):
    """Generate test submissions using best models."""
    log('\n=== Generating test submissions ===')
    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    df_test = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    df_test = df_test.unique(subset=['event_id'], keep='first')
    df_test = v9.add_features(df_test)
    df_test = v9.add_customer_profiles(df_test, profiles)

    X_test = df_test.select(available_feats).to_pandas().astype(np.float32)
    event_ids = df_test['event_id'].to_numpy()
    log(f'Test: {X_test.shape}')

    # Method 1 predictions
    preds_m1 = np.mean([
        lgb.Booster(model_file=str(MODELS_OUT / f'm1_lgbm_s{s}.txt')).predict(X_test)
        for s in seeds], axis=0)

    # Method 2 predictions
    preds_m2 = np.mean([
        lgb.Booster(model_file=str(MODELS_OUT / f'm2_lgbm_s{s}.txt')).predict(X_test)
        for s in seeds], axis=0)

    # Blend
    preds_blend = 0.5 * rankdata(preds_m1) + 0.5 * rankdata(preds_m2)

    sample = pl.read_csv(DATA / 'sample_submit.csv')
    ts = datetime.now().strftime('%Y%m%d_%H%M')

    methods = {
        'M1_standard': preds_m1,
        'M2_regularized': preds_m2,
        'M3_blend': preds_blend,
    }

    for name, score in methods.items():
        sub = pl.DataFrame({'event_id': event_ids, 'predict': score})
        sub = sample.select('event_id').join(sub, on='event_id', how='left')
        n_null = sub['predict'].is_null().sum()
        if n_null > 0:
            sub = sub.with_columns(pl.col('predict').fill_null(sub['predict'].drop_nulls().median()))
        path = SUBMIT_OUT / f'submit_v10_{name}_{ts}.csv'
        sub.write_csv(path)
        log(f'Saved: {path.name}')


if __name__ == '__main__':
    results, X_val, y_val, available_feats = train_and_evaluate()
    generate_submissions(available_feats)
    log('DONE')
