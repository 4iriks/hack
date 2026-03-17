"""
Pipeline v11 finish: Train CatBoost (LGBM + XGB already done) + blend + submit.
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from scipy.optimize import minimize
from pathlib import Path
from datetime import datetime
import gc, json, time, os

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_OUT  = ROOT / 'models_v11'
SUBMIT_OUT  = ROOT / 'submissions'

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

# Import from v11
import importlib.util
spec = importlib.util.spec_from_file_location("v11", ROOT / "pipeline_v11.py")
v11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v11)

seeds = [42, 123, 777, 2024, 31337]

if __name__ == '__main__':
    t0 = time.time()

    X_train, y_train, X_val, y_val, available_feats = v11.load_data()

    # Train CatBoost only
    log('\n=== CatBoost training ===')
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    cb_preds = np.zeros(len(X_val), dtype=np.float64)
    cb_iters = []
    for seed in seeds:
        m = cb.CatBoostClassifier(
            iterations=50000, learning_rate=0.01,
            depth=7, l2_leaf_reg=5.0,
            bootstrap_type='Bernoulli', subsample=0.7,
            scale_pos_weight=spw,
            task_type='GPU', devices='0',
            eval_metric='PRAUC',
            random_seed=seed, verbose=0,
            early_stopping_rounds=500,
            use_best_model=True,
        )
        m.fit(X_train, y_train,
              eval_set=(X_val, y_val),
              verbose=0)
        best = m.get_best_iteration()
        cb_iters.append(best)
        p = m.predict_proba(X_val)[:, 1]
        cb_preds += p
        prauc = average_precision_score(y_val, p)
        log(f'  Seed {seed}: iter={best}, val={prauc:.6f}')
        m.save_model(str(MODELS_OUT / f'catboost_s{seed}.cbm'))
        del m; gc.collect()
    cb_preds /= len(seeds)
    cb_prauc = average_precision_score(y_val, cb_preds)
    log(f'CatBoost ensemble: val={cb_prauc:.6f}, iters={cb_iters}')

    # Load LGBM and XGB predictions on val
    log('\n=== Loading LGBM + XGB predictions ===')
    lgbm_preds = np.mean([
        lgb.Booster(model_file=str(MODELS_OUT / f'lgbm_s{s}.txt')).predict(
            X_val.values) for s in seeds], axis=0)
    lgbm_prauc = average_precision_score(y_val, lgbm_preds)

    xgb_preds = np.zeros(len(X_val), dtype=np.float64)
    for s in seeds:
        m = xgb.XGBClassifier()
        m.load_model(str(MODELS_OUT / f'xgb_s{s}.json'))
        xgb_preds += m.predict_proba(X_val)[:, 1]
        del m
    xgb_preds /= len(seeds)
    xgb_prauc = average_precision_score(y_val, xgb_preds)

    # Results
    log('\n=== INDIVIDUAL RESULTS ===')
    log(f'LGBM:     val={lgbm_prauc:.6f}')
    log(f'XGBoost:  val={xgb_prauc:.6f}')
    log(f'CatBoost: val={cb_prauc:.6f}')

    # Equal blend
    equal = (rankdata(lgbm_preds) + rankdata(xgb_preds) + rankdata(cb_preds)) / 3
    eq_prauc = average_precision_score(y_val, equal)
    log(f'Equal blend: val={eq_prauc:.6f}')

    # Optimized blend
    names = ['LGBM', 'XGB', 'CatBoost']
    weights, opt_score = v11.optimize_blend(
        [lgbm_preds, xgb_preds, cb_preds], y_val, names)

    # Save results
    results = {
        'lgbm': lgbm_prauc, 'xgb': xgb_prauc, 'catboost': cb_prauc,
        'equal_blend': eq_prauc, 'optimized_blend': opt_score,
        'weights': dict(zip(names, weights.tolist())),
        'features': len(available_feats),
    }
    with open(MODELS_OUT / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate submissions
    v11.generate_submissions(available_feats, weights)

    log(f'\nTotal time: {(time.time()-t0)/60:.1f} min')
    log('DONE')
