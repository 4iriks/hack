"""
Pipeline v9 FINAL: Quick script to do final training + test prediction.
Uses iteration counts from val stage (which completed successfully).
Fixes LGBM GPU crash by using min_child_samples=200, num_leaves=127.
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from pathlib import Path
from datetime import datetime
import gc, json, time

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_OUT  = ROOT / 'models_v9'
SUBMIT_OUT  = ROOT / 'submissions'
MODELS_OUT.mkdir(exist_ok=True)

# Iteration counts from val stage
S1_ITERS = [572, 554, 485, 553, 594]  # avg ~552
S2_ITERS = [531, 330, 406, 404, 370]  # avg ~408
S3_ITERS = [312, 341, 236, 326, 307]  # avg ~304

# Optimized weights from val
WEIGHTS = {'suspicious': 0.66, 'fraud_clf': 0.0749, 'main+susp': 0.2651}

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

def main():
    t_start = time.time()
    log('Loading data...')

    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    label_ids = set(labels['event_id'].to_list())
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())

    df = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    df = v9.add_features(df)
    df = v9.add_customer_profiles(df, profiles)

    available_feats = [c for c in v9.FEATURE_COLS if c in df.columns]
    log(f'Features: {len(available_feats)}, Rows: {len(df):,}')

    df = df.with_columns(
        pl.col('event_id').is_in(label_ids).cast(pl.Int8).alias('is_suspicious'),
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'),
    )

    X_full = df.select(available_feats).to_pandas().astype(np.float32)
    y_susp = df['is_suspicious'].to_numpy()
    y_fraud = df['is_fraud'].to_numpy()
    y_target = df['target'].to_numpy().astype(int)
    labeled_mask = df['event_id'].is_in(label_ids).to_numpy()
    log(f'RAM: {_ram_gb():.1f}GB')

    # ── Stage 1: Suspicious Detector ──
    avg_s1 = int(np.mean(S1_ITERS) * 1.1)
    spw1 = (y_susp == 0).sum() / max((y_susp == 1).sum(), 1)
    log(f'Stage 1: {avg_s1} iters, spw={spw1:.1f}')

    for seed in seeds:
        m = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed,
            scale_pos_weight=spw1, n_estimators=avg_s1)
        m.fit(X_full, y_susp)
        m.booster_.save_model(str(MODELS_OUT / f's1_lgbm_s{seed}.txt'))
        log(f'  S1[{seed}] done')
        del m; gc.collect()

    # Get S1 predictions on full data for Stage 3
    s1_preds = np.mean([
        lgb.Booster(model_file=str(MODELS_OUT / f's1_lgbm_s{s}.txt')).predict(X_full)
        for s in seeds], axis=0)

    # ── Stage 2: Fraud Classifier (labeled only) ──
    X_labeled = X_full[labeled_mask]
    y_labeled_fraud = y_fraud[labeled_mask]
    avg_s2 = int(np.mean(S2_ITERS) * 1.1)
    spw2 = (y_labeled_fraud == 0).sum() / max((y_labeled_fraud == 1).sum(), 1)
    log(f'Stage 2: {avg_s2} iters, {len(X_labeled):,} rows, spw={spw2:.2f}')

    for seed in seeds:
        m = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed,
            scale_pos_weight=spw2, n_estimators=avg_s2)
        m.fit(X_labeled, y_labeled_fraud)
        m.booster_.save_model(str(MODELS_OUT / f's2_lgbm_s{seed}.txt'))
        log(f'  S2[{seed}] done')
        del m; gc.collect()
    del X_labeled, y_labeled_fraud; gc.collect()

    # ── Stage 3: Main + susp score ──
    X_plus = X_full.copy()
    X_plus['susp_score'] = s1_preds.astype(np.float32)
    avg_s3 = int(np.mean(S3_ITERS) * 1.1)
    spw3 = (y_target == 0).sum() / max((y_target == 1).sum(), 1)
    log(f'Stage 3: {avg_s3} iters, spw={spw3:.1f}')

    for seed in seeds:
        m = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed,
            scale_pos_weight=spw3, n_estimators=avg_s3)
        m.fit(X_plus, y_target)
        m.booster_.save_model(str(MODELS_OUT / f's3_lgbm_s{seed}.txt'))
        log(f'  S3[{seed}] done')
        del m; gc.collect()

    del X_full, X_plus, y_susp, y_fraud, y_target, s1_preds, df; gc.collect()

    # ── Test predictions ──
    log('Predicting test...')
    df_test = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    df_test = df_test.unique(subset=['event_id'], keep='first')
    df_test = v9.add_features(df_test)
    df_test = v9.add_customer_profiles(df_test, profiles)
    X_test = df_test.select(available_feats).to_pandas().astype(np.float32)
    event_ids = df_test['event_id'].to_numpy()
    log(f'Test: {X_test.shape}')

    # Predictions from each stage
    test_s1 = np.mean([lgb.Booster(model_file=str(MODELS_OUT / f's1_lgbm_s{s}.txt')).predict(X_test)
                        for s in seeds], axis=0)
    test_s2 = np.mean([lgb.Booster(model_file=str(MODELS_OUT / f's2_lgbm_s{s}.txt')).predict(X_test)
                        for s in seeds], axis=0)
    X_test_plus = X_test.copy()
    X_test_plus['susp_score'] = test_s1.astype(np.float32)
    test_s3 = np.mean([lgb.Booster(model_file=str(MODELS_OUT / f's3_lgbm_s{s}.txt')).predict(X_test_plus)
                        for s in seeds], axis=0)

    # Generate submissions for multiple methods
    sample = pl.read_csv(DATA / 'sample_submit.csv')
    ts = datetime.now().strftime('%Y%m%d_%H%M')

    methods = {
        'A_susp_x_fraud': rankdata(test_s1) * rankdata(test_s2),
        'C_main_susp': test_s3,
        'E_optimized': (WEIGHTS['suspicious'] * rankdata(test_s1) +
                       WEIGHTS['fraud_clf'] * rankdata(test_s2) +
                       WEIGHTS['main+susp'] * rankdata(test_s3)),
    }

    for name, score in methods.items():
        sub = pl.DataFrame({'event_id': event_ids, 'predict': score})
        sub = sample.select('event_id').join(sub, on='event_id', how='left')
        n_null = sub['predict'].is_null().sum()
        if n_null > 0:
            sub = sub.with_columns(pl.col('predict').fill_null(sub['predict'].drop_nulls().median()))
        path = SUBMIT_OUT / f'submit_v9_{name}_{ts}.csv'
        sub.write_csv(path)
        log(f'Saved: {path.name}')

    log(f'\nTotal: {(time.time()-t_start)/60:.1f} min')
    log('DONE')

if __name__ == '__main__':
    main()
