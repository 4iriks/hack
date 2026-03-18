"""Generate submission from already-trained B_plus_pretest models (v16)."""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
import gc, importlib.util

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_OUT  = ROOT / 'models_v16'
SUBMIT_OUT  = ROOT / 'submissions'

def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

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

if __name__ == '__main__':
    log('=== GENERATE SUBMIT v16-B ===')

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

    # Build test features
    log('Building test features...')
    test_df = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    test_df = test_df.unique(subset=['event_id'], keep='first')
    test_df = v9.add_features(test_df)
    test_df = v9.add_customer_profiles(test_df, old_profiles)

    # v14 anomaly features (old profiles)
    test_df = v14.add_anomaly_features(test_df, old_deep_profs, old_mcc_profs)

    # Pretest anomaly features (additive)
    test_df = v16.add_pretest_anomaly_features(test_df, pretest_profs, pretest_mcc)

    # Feature list: v14-C + pretest
    base_feats = [c for c in v9.FEATURE_COLS if c in test_df.columns]
    anom_feats = [c for c in v14.ANOMALY_FEATURES if c in test_df.columns]
    pt_feats = [c for c in v16.PRETEST_ANOMALY_FEATURES if c in test_df.columns]
    feats = base_feats + anom_feats + pt_feats
    log(f'Features: {len(base_feats)} base + {len(anom_feats)} anom + {len(pt_feats)} pretest = {len(feats)}')

    available_feats = [f for f in feats if f in test_df.columns]
    X_test = test_df.select(available_feats).to_pandas().astype(np.float32)
    event_ids = test_df['event_id'].to_numpy()

    # Predict
    log('Predicting...')
    preds = np.mean([
        lgb.Booster(model_file=str(MODELS_OUT / f'B_plus_pretest_s{s}.txt')).predict(X_test)
        for s in seeds], axis=0)

    # Format submission
    sample = pl.read_csv(DATA / 'sample_submit.csv')
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    sub = pl.DataFrame({'event_id': event_ids, 'predict': preds.astype(np.float64)})
    sub = sample.select('event_id').join(sub, on='event_id', how='left')
    n_null = sub['predict'].is_null().sum()
    if n_null > 0:
        sub = sub.with_columns(pl.col('predict').fill_null(sub['predict'].drop_nulls().median()))
        log(f'Filled {n_null} nulls')

    path = SUBMIT_OUT / f'submit_v16_B_plus_pretest_{ts}.csv'
    sub.write_csv(path)
    log(f'Saved: {path.name}')
    log(f'Predict stats: min={preds.min():.6f}, max={preds.max():.6f}, mean={preds.mean():.6f}')
    log('DONE')
