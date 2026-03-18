"""Continue v16 ablation: run C, BC, D experiments (B already done)."""
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

# Import pipelines
spec9 = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
v9 = importlib.util.module_from_spec(spec9)
spec9.loader.exec_module(v9)

spec14 = importlib.util.spec_from_file_location("v14", ROOT / "pipeline_v14.py")
v14 = importlib.util.module_from_spec(spec14)
spec14.loader.exec_module(v14)

spec16 = importlib.util.spec_from_file_location("v16", ROOT / "pipeline_v16.py")
v16 = importlib.util.module_from_spec(spec16)
spec16.loader.exec_module(v16)

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

def train_model(name, X_train, y_train, X_val, y_val, n_trees=10000, patience=300):
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    spw = min(spw, 50.0)
    log(f'\n{"="*60}')
    log(f'Training: {name}')
    log(f'Features: {X_train.shape[1]}, Trees: {n_trees}, spw: {spw:.1f}')
    log(f'{"="*60}')

    preds = np.zeros(len(X_val), dtype=np.float64)
    for i, seed in enumerate(seeds):
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
    log('=== V16 CONTINUE: C, BC, D ===')

    # Load profiles
    old_deep_profs = pl.read_parquet(FEATURES_IN / 'deep_customer_profiles.parquet')
    old_mcc_profs = pl.read_parquet(FEATURES_IN / 'customer_mcc_profiles.parquet')
    if old_mcc_profs['mcc_code'].dtype != pl.Int32:
        old_mcc_profs = old_mcc_profs.with_columns(pl.col('mcc_code').cast(pl.Int32))
    old_profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')

    pretest_profs = pl.read_parquet(FEATURES_IN / 'pretest_deep_profiles.parquet')
    pretest_mcc = pl.read_parquet(FEATURES_IN / 'pretest_mcc_profiles.parquet')
    if pretest_mcc['mcc_code'].dtype != pl.Int32:
        pretest_mcc = pretest_mcc.with_columns(pl.col('mcc_code').cast(pl.Int32))

    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())
    val_ids = set(pl.read_parquet(FEATURES_IN / 'val_event_ids.parquet')['event_id'].to_list())

    blended_profs = v16.build_blended_profiles(pretest_profs, old_deep_profs)
    gc.collect()

    # Load data
    log('Loading data...')
    val_df = pl.read_parquet(FEATURES_IN / 'val_proper.parquet')
    val_df = val_df.with_columns(pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'))
    val_df = v9.add_features(val_df)
    val_df = v9.add_customer_profiles(val_df, old_profiles)

    train_df = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    train_df = train_df.filter(~pl.col('event_id').is_in(val_ids))
    train_df = v9.add_features(train_df)
    train_df = v9.add_customer_profiles(train_df, old_profiles)

    y_train = train_df['target'].to_numpy().astype(int)
    y_val = val_df['is_fraud'].to_numpy().astype(int)
    log(f'Train: {len(train_df):,}, Val: {len(val_df):,}, Val fraud: {y_val.sum()}')

    # Add v14 anomaly features
    train_df = v14.add_anomaly_features(train_df, old_deep_profs, old_mcc_profs)
    val_df = v14.add_anomaly_features(val_df, old_deep_profs, old_mcc_profs)

    base_feats = [c for c in v9.FEATURE_COLS if c in train_df.columns and c in val_df.columns]
    anom_feats = [c for c in v14.ANOMALY_FEATURES if c in train_df.columns and c in val_df.columns]
    feats_a = base_feats + anom_feats

    # Add pretest + drift features (both needed for BC)
    train_df = v16.add_pretest_anomaly_features(train_df, pretest_profs, pretest_mcc)
    val_df = v16.add_pretest_anomaly_features(val_df, pretest_profs, pretest_mcc)
    train_df = v16.add_drift_features(train_df, pretest_profs, old_deep_profs)
    val_df = v16.add_drift_features(val_df, pretest_profs, old_deep_profs)

    pt_feats = [c for c in v16.PRETEST_ANOMALY_FEATURES if c in train_df.columns]
    dr_feats = [c for c in v16.DRIFT_FEATURES if c in train_df.columns]
    gc.collect()

    results = {'B_plus_pretest': 0.053538}  # Already done

    # ── EXP C: v14-C + drift ──
    log('\n>>> EXP C: + DRIFT FEATURES <<<')
    feats_c = feats_a + dr_feats
    log(f'Features: {len(feats_c)}')
    X_tr = train_df.select(feats_c).to_pandas().astype(np.float32)
    X_va = val_df.select(feats_c).to_pandas().astype(np.float32)
    results['C_plus_drift'] = train_model('C_plus_drift', X_tr, y_train, X_va, y_val)
    del X_tr, X_va; gc.collect()

    # ── EXP BC: v14-C + pretest + drift ──
    log('\n>>> EXP BC: + PRETEST + DRIFT <<<')
    feats_bc = feats_a + pt_feats + dr_feats
    log(f'Features: {len(feats_bc)}')
    X_tr = train_df.select(feats_bc).to_pandas().astype(np.float32)
    X_va = val_df.select(feats_bc).to_pandas().astype(np.float32)
    results['BC_pretest_drift'] = train_model('BC_pretest_drift', X_tr, y_train, X_va, y_val)
    del X_tr, X_va, train_df, val_df; gc.collect()

    # ── EXP D: Blended profiles ──
    log('\n>>> EXP D: BLENDED PROFILES <<<')
    val_df_d = pl.read_parquet(FEATURES_IN / 'val_proper.parquet')
    val_df_d = val_df_d.with_columns(pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'))
    val_df_d = v9.add_features(val_df_d)
    val_df_d = v9.add_customer_profiles(val_df_d, old_profiles)
    val_df_d = v14.add_anomaly_features(val_df_d, blended_profs, old_mcc_profs)

    train_df_d = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    train_df_d = train_df_d.filter(~pl.col('event_id').is_in(val_ids))
    train_df_d = v9.add_features(train_df_d)
    train_df_d = v9.add_customer_profiles(train_df_d, old_profiles)
    train_df_d = v14.add_anomaly_features(train_df_d, blended_profs, old_mcc_profs)

    feats_d = [c for c in feats_a if c in train_df_d.columns and c in val_df_d.columns]
    log(f'Features: {len(feats_d)} (blended profiles)')

    y_train_d = train_df_d['target'].to_numpy().astype(int)
    y_val_d = val_df_d['is_fraud'].to_numpy().astype(int)
    X_tr = train_df_d.select(feats_d).to_pandas().astype(np.float32)
    X_va = val_df_d.select(feats_d).to_pandas().astype(np.float32)
    results['D_blended'] = train_model('D_blended', X_tr, y_train_d, X_va, y_val_d)
    del X_tr, X_va, train_df_d, val_df_d; gc.collect()

    # Results
    results['A_baseline_ref'] = 0.044
    log(f'\n{"="*60}')
    log('ИТОГИ V16 ABLATION')
    log(f'{"="*60}')
    sorted_res = sorted(results.items(), key=lambda x: -x[1])
    for name, prauc in sorted_res:
        delta = 100 * (prauc / 0.044 - 1)
        log(f'{name:25s}: val={prauc:.6f} ({delta:+.1f}% vs v14-C)')

    with open(MODELS_OUT / 'results_v16.json', 'w') as f:
        json.dump(results, f, indent=2)

    total_min = (time.time() - t_start) / 60
    log(f'\nВсего: {total_min:.0f} мин')
    log('DONE')
