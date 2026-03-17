"""
Generate v16 submissions using EXISTING v14-C models but with PRETEST profiles for test data.

Key insight: v14-C models were trained with old profiles (pretrain+train, 177M rows).
But test data is from the same epoch as pretest (Jun-Aug'25).
Using pretest profiles for anomaly features at test time should reduce distribution shift.

Experiments:
A: v14-C models + OLD profiles (baseline, reproduce v14-C)
B: v14-C models + PRETEST deep profiles for anomaly features
C: v14-C models + HYBRID profiles (pretest if >=10 tx, else old)
D: v16-A models (trained with hybrid) + HYBRID profiles (if models exist)
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from pathlib import Path
from datetime import datetime
import gc, time

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_V14  = ROOT / 'models_v14'
MODELS_V16  = ROOT / 'models_v16'
SUBMIT_OUT  = ROOT / 'submissions'
RAW_TEST    = ROOT / 'Pre-test_Test'

SUBMIT_OUT.mkdir(exist_ok=True)

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


# Import v9 for existing features
import importlib.util
spec = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
v9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9)

# Import v16 for anomaly features
spec16 = importlib.util.spec_from_file_location("v16", ROOT / "pipeline_v16.py")
v16 = importlib.util.module_from_spec(spec16)
spec16.loader.exec_module(v16)

seeds = [42, 123, 777, 2024, 31337]

ANOMALY_FEATURES = v16.ANOMALY_FEATURES


def prepare_test_features(profiles, mcc_profiles, old_profiles):
    """Build test features with given profiles for anomaly computation."""
    test_df = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    test_df = test_df.unique(subset=['event_id'], keep='first')
    test_df = v9.add_features(test_df)
    test_df = v9.add_customer_profiles(test_df, old_profiles)
    test_df = v16.add_anomaly_features(test_df, profiles, mcc_profiles)
    return test_df


def generate_submission(name, test_df, feats, model_dir, model_prefix):
    """Generate submission CSV."""
    available_feats = [f for f in feats if f in test_df.columns]
    missing = set(feats) - set(available_feats)
    if missing:
        log(f'  WARNING: missing features: {missing}')

    X_test = test_df.select(available_feats).to_pandas().astype(np.float32)
    event_ids = test_df['event_id'].to_numpy()

    model_files = [model_dir / f'{model_prefix}_s{s}.txt' for s in seeds]
    existing = [f for f in model_files if f.exists()]
    log(f'  Using {len(existing)} models from {model_dir.name}/{model_prefix}')

    preds = np.mean([
        lgb.Booster(model_file=str(f)).predict(X_test)
        for f in existing], axis=0)

    sample = pl.read_csv(DATA / 'sample_submit.csv')
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    sub = pl.DataFrame({'event_id': event_ids, 'predict': preds.astype(np.float64)})
    sub = sample.select('event_id').join(sub, on='event_id', how='left')
    n_null = sub['predict'].is_null().sum()
    if n_null > 0:
        sub = sub.with_columns(pl.col('predict').fill_null(sub['predict'].drop_nulls().median()))
        log(f'  Filled {n_null} nulls')

    path = SUBMIT_OUT / f'submit_v16_{name}_{ts}.csv'
    sub.write_csv(path)
    log(f'  Saved: {path.name}')

    # Stats
    log(f'  predict range: [{preds.min():.6f}, {preds.max():.6f}], mean={preds.mean():.6f}')
    log(f'  top-10 predictions: {sorted(preds)[-10:][::-1]}')
    return path


if __name__ == '__main__':
    t_start = time.time()
    log('=== GENERATE V16 SUBMISSIONS ===')

    # Load profiles
    old_deep_profs = pl.read_parquet(FEATURES_IN / 'deep_customer_profiles.parquet')
    old_mcc_profs = pl.read_parquet(FEATURES_IN / 'customer_mcc_profiles.parquet')
    if old_mcc_profs['mcc_code'].dtype != pl.Int32:
        old_mcc_profs = old_mcc_profs.with_columns(pl.col('mcc_code').cast(pl.Int32))
    old_profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')

    pretest_deep = pl.read_parquet(FEATURES_IN / 'pretest_deep_profiles.parquet')
    pretest_mcc = pl.read_parquet(FEATURES_IN / 'pretest_mcc_profiles.parquet')
    if pretest_mcc['mcc_code'].dtype != pl.Int32:
        pretest_mcc = pretest_mcc.with_columns(pl.col('mcc_code').cast(pl.Int32))

    # Build hybrid profiles
    hybrid_profs = v16.build_hybrid_profiles(pretest_deep, old_deep_profs)
    mcc_cols = ['customer_id', 'mcc_code', 'mcc_amt_mean', 'mcc_n_tx_total']
    pretest_cust_ids = set(
        pretest_deep.filter(pl.col('prof_n_tx') >= v16.MIN_PRETEST_TX)['customer_id'].to_list()
    )
    hybrid_mcc = pl.concat([
        pretest_mcc.filter(pl.col('customer_id').is_in(pretest_cust_ids)).select(mcc_cols),
        old_mcc_profs.filter(~pl.col('customer_id').is_in(pretest_cust_ids)).select(mcc_cols),
    ])

    # Feature list (same as v14-C)
    base_feats = list(v9.FEATURE_COLS)
    anom_feats = list(ANOMALY_FEATURES)
    all_feats = base_feats + anom_feats  # 121 features

    log(f'Features: {len(all_feats)} (base={len(base_feats)}, anomaly={len(anom_feats)})')

    # ── A: v14-C baseline (old profiles for test) ──
    log('\n=== A: v14-C models + OLD profiles (baseline) ===')
    test_a = prepare_test_features(old_deep_profs, old_mcc_profs, old_profiles)
    path_a = generate_submission('A_old_profiles', test_a, all_feats, MODELS_V14, 'C_base_plus_anomaly')
    del test_a; gc.collect()

    # ── B: v14-C models + PRETEST profiles (only pretest, no fallback) ──
    log('\n=== B: v14-C models + PRETEST profiles ===')
    test_b = prepare_test_features(pretest_deep, pretest_mcc, old_profiles)
    path_b = generate_submission('B_pretest_profiles', test_b, all_feats, MODELS_V14, 'C_base_plus_anomaly')
    del test_b; gc.collect()

    # ── C: v14-C models + HYBRID profiles (pretest if >=10 tx, else old) ──
    log('\n=== C: v14-C models + HYBRID profiles ===')
    test_c = prepare_test_features(hybrid_profs, hybrid_mcc, old_profiles)
    path_c = generate_submission('C_hybrid_profiles', test_c, all_feats, MODELS_V14, 'C_base_plus_anomaly')
    del test_c; gc.collect()

    # ── D: v16-A models + HYBRID profiles (if v16 models exist) ──
    v16_model_check = MODELS_V16 / 'A_hybrid_profiles_s42.txt'
    if v16_model_check.exists():
        log('\n=== D: v16-A models + HYBRID profiles ===')
        test_d = prepare_test_features(hybrid_profs, hybrid_mcc, old_profiles)
        path_d = generate_submission('D_v16_hybrid', test_d, all_feats, MODELS_V16, 'A_hybrid_profiles')
        del test_d; gc.collect()
    else:
        log('\n=== D: SKIPPED (v16 models not found) ===')

    total_min = (time.time() - t_start) / 60
    log(f'\nВсего: {total_min:.1f} мин')
    log('DONE')
