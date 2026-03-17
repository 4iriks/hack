"""
Post-processing experiments on proper val.
Try fundamentally different approaches to scoring, no retraining needed.
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.ensemble import IsolationForest
from scipy.stats import rankdata
from scipy.optimize import minimize_scalar
from pathlib import Path
from datetime import datetime
import gc, time, os

ROOT = Path('/home/vadim/PyPr/hak')
FEATURES_IN = ROOT / 'features'
MODELS_V10 = ROOT / 'models_v10'
MODELS_V11 = ROOT / 'models_v11'
SUBMIT_OUT = ROOT / 'submissions'
DATA = ROOT / 'main_data'

seeds = [42, 123, 777, 2024, 31337]

def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

# Import feature engineering
import importlib.util
spec = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
v9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9)

spec11 = importlib.util.spec_from_file_location("v11", ROOT / "pipeline_v11.py")
v11 = importlib.util.module_from_spec(spec11)
spec11.loader.exec_module(v11)


def load_val_data():
    """Load val data with features and labels."""
    val_df = pl.read_parquet(FEATURES_IN / 'val_proper.parquet')
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())
    val_df = val_df.with_columns(
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'))
    return val_df


def get_v10_preds(val_df, feats):
    """Get v10 model predictions on val."""
    X = val_df.select(feats).to_pandas().astype(np.float32)
    preds = np.mean([
        lgb.Booster(model_file=str(MODELS_V10 / f'm1_lgbm_s{s}.txt')).predict(X)
        for s in seeds], axis=0)
    return preds


def get_v11_preds(val_df, feats):
    """Get v11 model predictions on val."""
    X = val_df.select(feats).to_pandas().astype(np.float32)
    preds = np.mean([
        lgb.Booster(model_file=str(MODELS_V11 / f'lgbm_s{s}.txt')).predict(X)
        for s in seeds], axis=0)
    return preds


if __name__ == '__main__':
    log('=== Post-processing experiments ===')

    # Load val data
    val_df = load_val_data()
    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    val_feat = v9.add_features(val_df.clone())
    val_feat = v9.add_customer_profiles(val_feat, profiles)

    # V10 features
    v10_feats = [c for c in v9.FEATURE_COLS if c in val_feat.columns]
    log(f'V10 features: {len(v10_feats)}')

    y_val = val_df['is_fraud'].to_numpy().astype(int)
    customer_ids = val_df['customer_id'].to_numpy()
    event_dttm = val_df['event_dttm'].to_numpy()
    n_fraud = y_val.sum()
    log(f'Val: {len(y_val):,} rows, {n_fraud} fraud')

    # Get base predictions
    v10_preds = get_v10_preds(val_feat, v10_feats)
    v10_prauc = average_precision_score(y_val, v10_preds)
    log(f'\nBaseline v10: val={v10_prauc:.6f}')

    # V11 features
    pretest_prof = pl.read_parquet(FEATURES_IN / 'pretest_profiles.parquet')
    mcc_rep = pl.read_parquet(FEATURES_IN / 'mcc_reputation.parquet')
    day_stats = pl.read_parquet(FEATURES_IN / 'day_stats.parquet')
    val_v11 = v11.add_new_features(val_feat.clone(), pretest_prof, mcc_rep, day_stats)
    v11_feats = [c for c in (v9.FEATURE_COLS + v11.NEW_FEATURE_COLS) if c in val_v11.columns]
    v11_preds = get_v11_preds(val_v11, v11_feats)
    v11_prauc = average_precision_score(y_val, v11_preds)
    log(f'Baseline v11: val={v11_prauc:.6f}')
    del val_v11; gc.collect()

    results = {}
    results['v10_baseline'] = v10_prauc
    results['v11_baseline'] = v11_prauc

    # ═══════════════════════════════════════
    # Experiment 1: Power transforms of v10
    # ═══════════════════════════════════════
    log('\n--- Exp 1: Power transforms ---')
    for power in [0.3, 0.5, 0.7, 1.5, 2.0, 3.0]:
        transformed = v10_preds ** power
        prauc = average_precision_score(y_val, transformed)
        log(f'  power={power}: val={prauc:.6f}')
        results[f'power_{power}'] = prauc

    # Optimal power via search
    def neg_prauc_power(p):
        return -average_precision_score(y_val, v10_preds ** p)
    opt = minimize_scalar(neg_prauc_power, bounds=(0.1, 5.0), method='bounded')
    best_power = opt.x
    best_prauc = -opt.fun
    log(f'  optimal power={best_power:.3f}: val={best_prauc:.6f}')
    results['power_optimal'] = best_prauc
    results['power_optimal_value'] = best_power

    # ═══════════════════════════════════════
    # Experiment 2: Rank-based transforms
    # ═══════════════════════════════════════
    log('\n--- Exp 2: Rank transforms ---')
    rank_preds = rankdata(v10_preds) / len(v10_preds)
    prauc = average_precision_score(y_val, rank_preds)
    log(f'  Rank: val={prauc:.6f}')
    results['rank'] = prauc

    # Log rank
    log_rank = np.log1p(rankdata(v10_preds))
    prauc = average_precision_score(y_val, log_rank)
    log(f'  Log-rank: val={prauc:.6f}')
    results['log_rank'] = prauc

    # ═══════════════════════════════════════
    # Experiment 3: Ensemble v10 + v11
    # ═══════════════════════════════════════
    log('\n--- Exp 3: v10+v11 ensemble ---')
    for w in [0.3, 0.5, 0.7, 0.8, 0.9]:
        blend = w * rankdata(v10_preds) + (1-w) * rankdata(v11_preds)
        prauc = average_precision_score(y_val, blend)
        log(f'  v10={w:.1f}, v11={1-w:.1f}: val={prauc:.6f}')
        results[f'v10v11_w{w}'] = prauc

    # ═══════════════════════════════════════
    # Experiment 4: Customer-day boosting
    # ═══════════════════════════════════════
    log('\n--- Exp 4: Customer-day boosting ---')

    # For each customer, boost all scores if ANY score is high
    unique_custs = np.unique(customer_ids)
    for threshold_pct in [90, 95, 99]:
        threshold = np.percentile(v10_preds, threshold_pct)
        boosted = v10_preds.copy()
        for cid in unique_custs:
            mask = customer_ids == cid
            cust_scores = v10_preds[mask]
            max_score = cust_scores.max()
            if max_score > threshold:
                # Boost all transactions for this customer
                boosted[mask] = cust_scores * (1 + max_score)
        prauc = average_precision_score(y_val, boosted)
        log(f'  Boost (top {100-threshold_pct}%): val={prauc:.6f}')
        results[f'boost_top{100-threshold_pct}pct'] = prauc

    # Customer-day: multiply each score by customer's max score
    log('\n--- Exp 4b: Score × customer_max ---')
    boosted2 = v10_preds.copy()
    for cid in unique_custs:
        mask = customer_ids == cid
        cust_max = v10_preds[mask].max()
        boosted2[mask] = v10_preds[mask] * cust_max
    prauc = average_precision_score(y_val, boosted2)
    log(f'  score × cust_max: val={prauc:.6f}')
    results['score_x_custmax'] = prauc

    # Score × customer mean
    boosted3 = v10_preds.copy()
    for cid in unique_custs:
        mask = customer_ids == cid
        cust_mean = v10_preds[mask].mean()
        boosted3[mask] = v10_preds[mask] * cust_mean
    prauc = average_precision_score(y_val, boosted3)
    log(f'  score × cust_mean: val={prauc:.6f}')
    results['score_x_custmean'] = prauc

    # Rank within customer
    log('\n--- Exp 4c: Customer-relative ranking ---')
    cust_rank = np.zeros_like(v10_preds)
    for cid in unique_custs:
        mask = customer_ids == cid
        cust_scores = v10_preds[mask]
        cust_rank[mask] = rankdata(cust_scores) / len(cust_scores)
    prauc = average_precision_score(y_val, cust_rank)
    log(f'  Within-customer rank: val={prauc:.6f}')
    results['cust_rank'] = prauc

    # Combined: global rank × customer rank
    combined = rankdata(v10_preds) * cust_rank
    prauc = average_precision_score(y_val, combined)
    log(f'  global_rank × cust_rank: val={prauc:.6f}')
    results['global_x_cust_rank'] = prauc

    # ═══════════════════════════════════════
    # Experiment 5: Isolation Forest anomaly
    # ═══════════════════════════════════════
    log('\n--- Exp 5: Isolation Forest ---')
    X_val = val_feat.select(v10_feats).to_pandas().astype(np.float32)

    # Train IF on val itself (unsupervised)
    for contamination in [0.001, 0.005, 0.01]:
        iso = IsolationForest(n_estimators=500, contamination=contamination,
                              random_state=42, n_jobs=4)
        iso.fit(X_val)
        anomaly_scores = -iso.score_samples(X_val)  # higher = more anomalous
        prauc = average_precision_score(y_val, anomaly_scores)
        log(f'  IF(contam={contamination}): val={prauc:.6f}')
        results[f'IF_c{contamination}'] = prauc

        # Combine with v10
        combo = rankdata(v10_preds) + rankdata(anomaly_scores)
        prauc = average_precision_score(y_val, combo)
        log(f'  IF + v10: val={prauc:.6f}')
        results[f'IF_plus_v10_c{contamination}'] = prauc

        # v10 × IF
        combo2 = rankdata(v10_preds) * rankdata(anomaly_scores)
        prauc = average_precision_score(y_val, combo2)
        log(f'  IF × v10: val={prauc:.6f}')
        results[f'IF_times_v10_c{contamination}'] = prauc

    # ═══════════════════════════════════════
    # Experiment 6: Stacking
    # ═══════════════════════════════════════
    log('\n--- Exp 6: v10 × v11 product ---')
    product = v10_preds * v11_preds
    prauc = average_precision_score(y_val, product)
    log(f'  v10 × v11: val={prauc:.6f}')
    results['v10_times_v11'] = prauc

    # ═══════════════════════════════════════
    # Summary: find best
    # ═══════════════════════════════════════
    log('\n=== TOP 10 METHODS ===')
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (name, prauc) in enumerate(sorted_results[:10]):
        marker = ' ← BEST' if i == 0 else ''
        log(f'  {i+1}. {name}: val={prauc:.6f}{marker}')

    best_name, best_prauc = sorted_results[0]
    log(f'\nBest method: {best_name} (val={best_prauc:.6f} vs baseline {v10_prauc:.6f})')
    log(f'Improvement: {best_prauc - v10_prauc:+.6f} ({100*(best_prauc/v10_prauc - 1):+.1f}%)')

    # Save results
    import json
    with open(ROOT / 'models_v11' / 'postproc_results.json', 'w') as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)

    log('\nDONE')
