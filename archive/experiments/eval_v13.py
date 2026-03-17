"""Quick eval: загрузить модели v13, посчитать val PR-AUC для каждого эксперимента."""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from pathlib import Path
from datetime import datetime
import json

ROOT = Path('/home/vadim/PyPr/hak')
DATA = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_OUT = ROOT / 'models_v13'

import importlib.util
spec = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
v9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9)

seeds = [42, 123, 777, 2024, 31337]

LEAK_TOP2 = {'dormancy_days', 'month'}
LEAK_TOP5 = {'dormancy_days', 'month', 'weekday', 'session_ops_before', 'cust_tenure_days'}

def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

if __name__ == '__main__':
    log('Loading val data...')
    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())

    val_df = pl.read_parquet(FEATURES_IN / 'val_proper.parquet')
    val_df = val_df.with_columns(
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'))
    val_df = v9.add_features(val_df)
    val_df = v9.add_customer_profiles(val_df, profiles)

    all_feats = [c for c in v9.FEATURE_COLS if c in val_df.columns]
    y_val = val_df['is_fraud'].to_numpy().astype(int)

    log(f'Val: {len(val_df):,}, fraud: {y_val.sum()}')

    experiments = {
        'A_no_top2': {'feats': [f for f in all_feats if f not in LEAK_TOP2], 'clip': None},
        'B_no_top5': {'feats': [f for f in all_feats if f not in LEAK_TOP5], 'clip': None},
        'C_30k_trees': {'feats': all_feats, 'clip': None},
        'D_clip_dormancy': {'feats': all_feats, 'clip': 200},
        'E_no_top2_reg': {'feats': [f for f in all_feats if f not in LEAK_TOP2], 'clip': None},
        'F_no_top5_30k': {'feats': [f for f in all_feats if f not in LEAK_TOP5], 'clip': None},
    }

    results = {}

    for name, cfg in experiments.items():
        feats = cfg['feats']
        X_val = val_df.select(feats).to_pandas().astype(np.float32)
        if cfg['clip'] and 'dormancy_days' in feats:
            X_val['dormancy_days'] = X_val['dormancy_days'].clip(upper=cfg['clip'])

        preds = np.zeros(len(X_val), dtype=np.float64)
        seed_results = []
        for s in seeds:
            model_path = MODELS_OUT / f'{name}_s{s}.txt'
            if not model_path.exists():
                log(f'  MISSING: {model_path.name}')
                continue
            m = lgb.Booster(model_file=str(model_path))
            p = m.predict(X_val)
            preds += p
            prauc = average_precision_score(y_val, p)
            seed_results.append(prauc)

        preds /= len(seeds)
        ensemble_prauc = average_precision_score(y_val, preds)
        results[name] = ensemble_prauc

        log(f'{name:25s}: ensemble={ensemble_prauc:.6f} | seeds: {[f"{x:.4f}" for x in seed_results]}')

    # Sort
    log(f'\n{"="*60}')
    log('ИТОГИ V13 (sorted)')
    log(f'{"="*60}')
    log(f'v10 baseline (reference):      val=0.039')

    sorted_res = sorted(results.items(), key=lambda x: -x[1])
    for name, prauc in sorted_res:
        delta = 100 * (prauc / 0.039 - 1)
        log(f'{name:30s}: val={prauc:.6f} ({delta:+.1f}% vs v10)')

    best_name = sorted_res[0][0]
    log(f'\nЛучший: {best_name} (val={sorted_res[0][1]:.6f})')

    # Save
    with open(MODELS_OUT / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    log('Saved results.json')
