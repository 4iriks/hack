"""Quick: сгенерировать сабмит v14-C (base+anomaly, 121 features)."""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
import gc

ROOT = Path('/home/vadim/PyPr/hak')
DATA = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_OUT = ROOT / 'models_v14'
SUBMIT_OUT = ROOT / 'submissions'

import importlib.util
spec = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
v9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9)

# Import anomaly feature builder from v14
spec14 = importlib.util.spec_from_file_location("v14", ROOT / "pipeline_v14.py")
v14 = importlib.util.module_from_spec(spec14)
spec14.loader.exec_module(v14)

seeds = [42, 123, 777, 2024, 31337]

def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

if __name__ == '__main__':
    log('=== GENERATE V14-C SUBMISSION ===')

    # Load profiles
    deep_profiles = pl.read_parquet(FEATURES_IN / 'deep_customer_profiles.parquet')
    mcc_profiles = pl.read_parquet(FEATURES_IN / 'customer_mcc_profiles.parquet')
    if mcc_profiles['mcc_code'].dtype != pl.Int32:
        mcc_profiles = mcc_profiles.with_columns(pl.col('mcc_code').cast(pl.Int32))
    old_profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')

    # Load test features
    log('Loading test features...')
    test_df = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    test_df = test_df.unique(subset=['event_id'], keep='first')
    test_df = v9.add_features(test_df)
    test_df = v9.add_customer_profiles(test_df, old_profiles)
    test_df = v14.add_anomaly_features(test_df, deep_profiles, mcc_profiles)

    # Feature lists
    base_feats = [c for c in v9.FEATURE_COLS if c in test_df.columns]
    anom_feats = [c for c in v14.ANOMALY_FEATURES if c in test_df.columns]
    all_feats = base_feats + anom_feats
    log(f'Features: {len(all_feats)} ({len(base_feats)} base + {len(anom_feats)} anomaly)')

    X_test = test_df.select(all_feats).to_pandas().astype(np.float32)
    event_ids = test_df['event_id'].to_numpy()

    # Generate for C (10K trees) and D (20K trees) if models exist
    for name in ['C_base_plus_anomaly', 'D_all_20k']:
        model_files = [MODELS_OUT / f'{name}_s{s}.txt' for s in seeds]
        existing = [f for f in model_files if f.exists()]
        if len(existing) < 5:
            log(f'{name}: only {len(existing)}/5 models, skipping')
            continue

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

        path = SUBMIT_OUT / f'submit_v14_{name}_{ts}.csv'
        sub.write_csv(path)
        log(f'Saved: {path.name}')

    # Also blend v10 + v14-C (rank-based)
    log('\nGenerating v10+v14C blend...')
    from scipy.stats import rankdata

    # v10 predictions
    v10_preds = np.mean([
        lgb.Booster(model_file=str(ROOT / 'models_v10' / f'lgbm_s{s}.txt')).predict(
            test_df.select(base_feats).to_pandas().astype(np.float32))
        for s in seeds], axis=0)

    # v14-C predictions
    v14c_preds = np.mean([
        lgb.Booster(model_file=str(MODELS_OUT / f'C_base_plus_anomaly_s{s}.txt')).predict(X_test)
        for s in seeds], axis=0)

    # Rank blend
    r_v10 = rankdata(v10_preds) / len(v10_preds)
    r_v14 = rankdata(v14c_preds) / len(v14c_preds)

    for w10 in [0.3, 0.5, 0.7]:
        blend = w10 * r_v10 + (1 - w10) * r_v14
        sub = pl.DataFrame({'event_id': event_ids, 'predict': blend.astype(np.float64)})
        sub = sample.select('event_id').join(sub, on='event_id', how='left')
        n_null = sub['predict'].is_null().sum()
        if n_null > 0:
            sub = sub.with_columns(pl.col('predict').fill_null(sub['predict'].drop_nulls().median()))

        ts = datetime.now().strftime('%Y%m%d_%H%M')
        w14 = int((1-w10)*100)
        path = SUBMIT_OUT / f'submit_blend_v10_{int(w10*100)}_v14c_{w14}_{ts}.csv'
        sub.write_csv(path)
        log(f'Saved: {path.name}')

    log('DONE')
