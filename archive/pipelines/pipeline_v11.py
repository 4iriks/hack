"""
Pipeline v11: New features + full ensemble.

NEW FEATURES:
  1. Fresh customer profiles from pretest (14M rows, Jun-Aug 2025)
  2. MCC reputation from train labels (fraud rate per MCC)
  3. Day-level aggregations from full 85M train rows
  4. Sequence/position features within day

MODELS: LGBM + XGBoost + CatBoost × 50K trees × 5 seeds + optimized blend
VALIDATION: proper val from v10 (one random day per customer)

Calibration: v10 val=0.039 → LB=0.097
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
CHUNKS_DIR  = FEATURES_IN / '_tmp_train'
PRETEST_DIR = ROOT / 'Pre-test_Test'
PRETRAIN_DIR= ROOT / 'Pre-train_Train'
MODELS_OUT  = ROOT / 'models_v11'
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

seeds = [42, 123, 777, 2024, 31337]


# ═══════════════════════════════════════════════════════════════════
# PHASE 1: Build new feature tables (cached)
# ═══════════════════════════════════════════════════════════════════

def build_pretest_profiles():
    """Build fresh customer profiles from pretest data (Jun-Aug 2025)."""
    cache = FEATURES_IN / 'pretest_profiles.parquet'
    if cache.exists():
        log(f'Pretest profiles cached: {cache}')
        return pl.read_parquet(cache)

    log('Building pretest profiles from 14M rows...')
    pt = pl.read_parquet(PRETEST_DIR / 'pretest.parquet')
    pt = pt.with_columns(pl.col('event_dttm').str.to_datetime().alias('event_dttm'))
    log(f'Pretest: {len(pt):,} rows, {pt["customer_id"].n_unique():,} customers')

    # Parse amount
    amt = pt['operaton_amt'].cast(pl.Float64)

    profiles = pt.group_by('customer_id').agg([
        # Amount stats
        pl.col('operaton_amt').cast(pl.Float64).mean().alias('pt_avg_amt'),
        pl.col('operaton_amt').cast(pl.Float64).std().alias('pt_std_amt'),
        pl.col('operaton_amt').cast(pl.Float64).median().alias('pt_med_amt'),
        pl.col('operaton_amt').cast(pl.Float64).max().alias('pt_max_amt'),
        pl.col('operaton_amt').cast(pl.Float64).quantile(0.95).alias('pt_p95_amt'),

        # Activity
        pl.len().alias('pt_n_tx'),
        pl.col('mcc_code').n_unique().alias('pt_n_unique_mcc'),
        pl.col('channel_indicator_type').n_unique().alias('pt_n_unique_channel'),

        # Time patterns
        pl.col('event_dttm').dt.hour().cast(pl.Float64).mean().alias('pt_avg_hour'),
        pl.col('event_dttm').dt.hour().cast(pl.Float64).std().alias('pt_hour_std'),
        (pl.col('event_dttm').dt.hour().is_between(0, 6)).mean().cast(pl.Float64).alias('pt_pct_night'),

        # Tenure in pretest
        ((pl.col('event_dttm').max() - pl.col('event_dttm').min()).dt.total_seconds() / 86400).alias('pt_active_days'),

        # Last activity timestamp (for dormancy vs test)
        pl.col('event_dttm').max().dt.epoch('s').alias('pt_last_epoch'),

        # Security signals
        pl.col('phone_voip_call_state').sum().alias('pt_voip_sum'),
        pl.col('web_rdp_connection').sum().alias('pt_rdp_sum'),
    ])

    # Average gap between transactions
    gaps = (pt.sort('customer_id', 'event_dttm')
            .with_columns(
                pl.col('event_dttm').diff().over('customer_id').dt.total_seconds().alias('gap_sec'))
            .filter(pl.col('gap_sec').is_not_null())
            .group_by('customer_id').agg(
                pl.col('gap_sec').mean().alias('pt_avg_gap_sec'),
                pl.col('gap_sec').median().alias('pt_med_gap_sec'),
            ))

    profiles = profiles.join(gaps, on='customer_id', how='left')

    # High risk event type percentage
    high_risk = pt.with_columns(
        pl.col('event_type_nm').is_in([3, 7, 8]).alias('is_hr')
    ).group_by('customer_id').agg(
        pl.col('is_hr').mean().cast(pl.Float64).alias('pt_pct_high_risk')
    )
    profiles = profiles.join(high_risk, on='customer_id', how='left')

    # Fill nulls
    for c in profiles.columns:
        if c != 'customer_id':
            profiles = profiles.with_columns(pl.col(c).fill_null(0.0))

    profiles.write_parquet(cache)
    log(f'Pretest profiles: {len(profiles):,} customers → {cache}')
    del pt; gc.collect()
    return profiles


def build_mcc_reputation():
    """Build MCC fraud reputation from train labels."""
    cache = FEATURES_IN / 'mcc_reputation.parquet'
    if cache.exists():
        log(f'MCC reputation cached: {cache}')
        return pl.read_parquet(cache)

    log('Building MCC reputation from train labels...')
    labels = pl.read_parquet(DATA / 'train_labels.parquet')

    # Need mcc_code for labeled events — read from train chunks
    chunk_files = sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith('.parquet')])
    parts = []
    for cf in chunk_files:
        df = pl.read_parquet(CHUNKS_DIR / cf, columns=['event_id', 'mcc_code'])
        # Only keep labeled events
        df = df.filter(pl.col('event_id').is_in(labels['event_id']))
        parts.append(df)
    mcc_events = pl.concat(parts).unique(subset=['event_id'])
    del parts; gc.collect()

    # Join labels
    mcc_events = mcc_events.join(labels, on='event_id', how='inner')
    log(f'Labeled events with MCC: {len(mcc_events):,}')

    # Compute per-MCC stats
    reputation = mcc_events.group_by('mcc_code').agg([
        pl.len().alias('mcc_n_labeled'),
        (pl.col('target') == 1).sum().alias('mcc_n_fraud'),
        (pl.col('target') == 0).sum().alias('mcc_n_confirmed'),
        pl.col('target').mean().alias('mcc_fraud_rate'),
    ])

    # Fraud lift vs average
    avg_fraud_rate = mcc_events['target'].mean()
    reputation = reputation.with_columns(
        (pl.col('mcc_fraud_rate') / (avg_fraud_rate + 1e-6)).alias('mcc_fraud_lift'),
        # Smoothed fraud rate (Bayesian: add 10 pseudo-events at average rate)
        ((pl.col('mcc_n_fraud') + 10 * avg_fraud_rate) /
         (pl.col('mcc_n_labeled') + 10)).alias('mcc_fraud_rate_smooth'),
    )

    reputation.write_parquet(cache)
    log(f'MCC reputation: {len(reputation):,} MCCs → {cache}')
    return reputation


def build_day_stats():
    """Build day-level stats from full 85M train rows."""
    cache = FEATURES_IN / 'day_stats.parquet'
    if cache.exists():
        log(f'Day stats cached: {cache}')
        return pl.read_parquet(cache)

    log('Building day-level stats from 85M rows...')
    chunk_files = sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith('.parquet')])

    # Aggregate per chunk, then combine
    all_parts = []
    for i, cf in enumerate(chunk_files):
        df = pl.read_parquet(CHUNKS_DIR / cf,
                             columns=['customer_id', 'event_dttm', 'operaton_amt', 'mcc_code', 'session_id'])
        df = df.with_columns([
            pl.col('event_dttm').cast(pl.Date).alias('date'),
            pl.col('event_dttm').dt.epoch('s').alias('epoch_s'),
        ])

        day = df.group_by(['customer_id', 'date']).agg([
            pl.len().alias('_n'),
            pl.col('operaton_amt').sum().alias('_amt_sum'),
            pl.col('operaton_amt').max().alias('_amt_max'),
            pl.col('operaton_amt').mean().alias('_amt_mean'),
            pl.col('operaton_amt').std().alias('_amt_std'),
            pl.col('mcc_code').n_unique().alias('_mcc_nunique'),
            pl.col('epoch_s').min().alias('_t_min'),
            pl.col('epoch_s').max().alias('_t_max'),
            pl.col('session_id').n_unique().alias('_sess_nunique'),
        ])
        all_parts.append(day)
        if (i + 1) % 8 == 0:
            log(f'  Chunk {i+1}/{len(chunk_files)}, RAM: {_ram_gb():.1f}GB')
        del df; gc.collect()

    # Combine and re-aggregate (chunks may split a customer's day)
    combined = pl.concat(all_parts)
    del all_parts; gc.collect()
    log(f'Pre-aggregation: {len(combined):,} rows')

    day_stats = combined.group_by(['customer_id', 'date']).agg([
        pl.col('_n').sum().alias('day_n_ops'),
        pl.col('_amt_sum').sum().alias('day_total_amt'),
        pl.col('_amt_max').max().alias('day_max_amt'),
        # Weighted mean
        (pl.col('_amt_mean') * pl.col('_n')).sum() / pl.col('_n').sum(),
        pl.col('_mcc_nunique').max().alias('day_n_unique_mcc'),  # approximate
        pl.col('_t_min').min().alias('day_first_epoch'),
        pl.col('_t_max').max().alias('day_last_epoch'),
        pl.col('_sess_nunique').max().alias('day_n_sessions'),   # approximate
    ])

    day_stats = day_stats.with_columns([
        (pl.col('day_last_epoch') - pl.col('day_first_epoch')).alias('day_time_span_sec'),
    ])

    # Drop intermediate columns
    day_stats = day_stats.drop(['_amt_mean'])

    day_stats.write_parquet(cache)
    log(f'Day stats: {len(day_stats):,} (customer, date) pairs → {cache}')
    return day_stats


def add_new_features(df, pretest_prof, mcc_rep, day_stats, is_test=False):
    """Augment dataframe with all new features."""
    n_before = len(df.columns)

    # 1. Pretest profiles
    df = df.join(pretest_prof, on='customer_id', how='left')

    # Deviation features: current transaction vs pretest behavior
    if 'pt_avg_amt' in df.columns and 'operaton_amt' in df.columns:
        df = df.with_columns([
            ((pl.col('operaton_amt') - pl.col('pt_avg_amt')) / (pl.col('pt_std_amt') + 1)).alias('pt_amt_zscore'),
            (pl.col('operaton_amt') / (pl.col('pt_med_amt') + 1)).alias('pt_amt_vs_median'),
        ])
    if 'pt_last_epoch' in df.columns and 'event_dttm' in df.columns:
        df = df.with_columns(
            ((pl.col('event_dttm').dt.epoch('s') - pl.col('pt_last_epoch')) / 86400).alias('pt_dormancy_days'))
    if 'pt_avg_hour' in df.columns and 'hour' in df.columns:
        df = df.with_columns(
            (pl.col('hour').cast(pl.Float64) - pl.col('pt_avg_hour')).abs().alias('pt_hour_deviation'))

    # 2. MCC reputation
    df = df.join(mcc_rep, on='mcc_code', how='left')

    # 3. Day-level stats
    df = df.with_columns(pl.col('event_dttm').cast(pl.Date).alias('_date'))

    if is_test:
        # For test, compute day stats from the data itself (one day per customer)
        test_day = df.group_by(['customer_id', '_date']).agg([
            pl.len().alias('day_n_ops'),
            pl.col('operaton_amt').sum().alias('day_total_amt'),
            pl.col('operaton_amt').max().alias('day_max_amt'),
            pl.col('operaton_amt').mean().alias('day_mean_amt'),
            pl.col('mcc_code').n_unique().alias('day_n_unique_mcc'),
            pl.col('event_dttm').dt.epoch('s').min().alias('day_first_epoch'),
            pl.col('event_dttm').dt.epoch('s').max().alias('day_last_epoch'),
        ])
        test_day = test_day.with_columns(
            (pl.col('day_last_epoch') - pl.col('day_first_epoch')).alias('day_time_span_sec'))
        df = df.join(test_day, on=['customer_id', '_date'], how='left')
    else:
        # For train/val, use precomputed day stats from full data
        df = df.join(day_stats, left_on=['customer_id', '_date'],
                     right_on=['customer_id', 'date'], how='left')

    # Per-transaction features from day context
    if 'day_total_amt' in df.columns:
        df = df.with_columns([
            (pl.col('operaton_amt') / (pl.col('day_total_amt') + 1)).alias('amt_pct_of_day'),
            (pl.col('operaton_amt') >= pl.col('day_max_amt')).cast(pl.Int8).alias('is_max_amt_day'),
        ])

    # 4. Sequence features within day (position, cumulative)
    df = df.sort(['customer_id', 'event_dttm'])
    df = df.with_columns([
        pl.col('event_id').cum_count().over(['customer_id', '_date']).alias('pos_in_day'),
        pl.col('operaton_amt').cum_sum().over(['customer_id', '_date']).alias('cum_amt_in_day'),
    ])
    if 'day_n_ops' in df.columns:
        df = df.with_columns(
            (pl.col('pos_in_day').cast(pl.Float64) / (pl.col('day_n_ops') + 1)).alias('pct_pos_in_day'))
    if 'day_first_epoch' in df.columns and 'event_dttm' in df.columns:
        df = df.with_columns(
            (pl.col('event_dttm').dt.epoch('s') - pl.col('day_first_epoch')).alias('secs_since_day_start'))

    # Time between consecutive transactions within day
    df = df.with_columns(
        pl.col('event_dttm').diff().over(['customer_id', '_date']).dt.total_seconds().alias('gap_in_day_sec'))
    df = df.with_columns(pl.col('gap_in_day_sec').fill_null(0))

    df = df.drop(['_date'])

    # Fill nulls for all new columns
    for c in df.columns:
        if c.startswith('pt_') or c.startswith('mcc_') or c.startswith('day_') or \
           c in ('amt_pct_of_day', 'is_max_amt_day', 'pos_in_day', 'cum_amt_in_day',
                  'pct_pos_in_day', 'secs_since_day_start', 'gap_in_day_sec'):
            df = df.with_columns(pl.col(c).fill_null(0.0))

    log(f'  Features: {n_before} → {len(df.columns)}')
    return df


# New feature columns to add to model
NEW_FEATURE_COLS = [
    # Pretest profiles
    'pt_avg_amt', 'pt_std_amt', 'pt_med_amt', 'pt_max_amt', 'pt_p95_amt',
    'pt_n_tx', 'pt_n_unique_mcc', 'pt_n_unique_channel',
    'pt_avg_hour', 'pt_hour_std', 'pt_pct_night', 'pt_active_days',
    'pt_avg_gap_sec', 'pt_med_gap_sec', 'pt_pct_high_risk',
    'pt_voip_sum', 'pt_rdp_sum',
    # Pretest deviations
    'pt_amt_zscore', 'pt_amt_vs_median', 'pt_dormancy_days', 'pt_hour_deviation',
    # MCC reputation
    'mcc_n_labeled', 'mcc_n_fraud', 'mcc_fraud_rate', 'mcc_fraud_lift',
    'mcc_fraud_rate_smooth',
    # Day-level
    'day_n_ops', 'day_total_amt', 'day_max_amt', 'day_n_unique_mcc',
    'day_time_span_sec',
    # Day context per transaction
    'amt_pct_of_day', 'is_max_amt_day',
    # Sequence/position
    'pos_in_day', 'cum_amt_in_day', 'pct_pos_in_day',
    'secs_since_day_start', 'gap_in_day_sec',
]


# ═══════════════════════════════════════════════════════════════════
# PHASE 2: Load data + augment features
# ═══════════════════════════════════════════════════════════════════

def load_data():
    """Load train/val with all features."""
    log('=== Loading data ===')

    # Build new feature tables
    pretest_prof = build_pretest_profiles()
    mcc_rep = build_mcc_reputation()
    day_stats = build_day_stats()

    # Load cached val
    val_df = pl.read_parquet(FEATURES_IN / 'val_proper.parquet')
    val_ids = pl.read_parquet(FEATURES_IN / 'val_event_ids.parquet')
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())

    val_df = val_df.with_columns(
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'))

    # Load train
    val_id_set = set(val_ids['event_id'].to_list())
    train_df = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    train_df = train_df.filter(~pl.col('event_id').is_in(val_id_set))

    # Apply v9 feature engineering
    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    train_df = v9.add_features(train_df)
    train_df = v9.add_customer_profiles(train_df, profiles)
    val_df = v9.add_features(val_df)
    val_df = v9.add_customer_profiles(val_df, profiles)
    del profiles; gc.collect()

    # Add new features
    log('Adding new features to train...')
    train_df = add_new_features(train_df, pretest_prof, mcc_rep, day_stats)
    log('Adding new features to val...')
    val_df = add_new_features(val_df, pretest_prof, mcc_rep, day_stats)

    # All feature columns
    all_feats = v9.FEATURE_COLS + NEW_FEATURE_COLS
    available_feats = [c for c in all_feats if c in train_df.columns and c in val_df.columns]
    log(f'Total features: {len(available_feats)}')

    X_train = train_df.select(available_feats).to_pandas().astype(np.float32)
    y_train = train_df['target'].to_numpy().astype(int)
    X_val = val_df.select(available_feats).to_pandas().astype(np.float32)
    y_val = val_df['is_fraud'].to_numpy().astype(int)

    log(f'Train: {X_train.shape}, fraud rate: {y_train.mean():.4f}')
    log(f'Val: {X_val.shape}, fraud: {(y_val==1).sum()} ({100*y_val.mean():.3f}%)')

    del train_df, val_df, pretest_prof, mcc_rep, day_stats; gc.collect()
    return X_train, y_train, X_val, y_val, available_feats


# ═══════════════════════════════════════════════════════════════════
# PHASE 3: Train models
# ═══════════════════════════════════════════════════════════════════

def train_lgbm(X_train, y_train, X_val, y_val):
    """LGBM with 50K trees, lr=0.01."""
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    log(f'\n=== LGBM: 50K trees, lr=0.01, spw={spw:.1f} ===')

    params = dict(
        objective='binary', metric='average_precision',
        device='gpu', gpu_platform_id=0, gpu_device_id=0,
        learning_rate=0.01,
        num_leaves=127, min_child_samples=200,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
        n_jobs=4, verbose=-1,
    )

    preds = np.zeros(len(X_val), dtype=np.float64)
    iters = []
    for seed in seeds:
        m = lgb.LGBMClassifier(**params, random_state=seed,
            scale_pos_weight=spw, n_estimators=50000)
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(0)])
        best = m.best_iteration_
        iters.append(best)
        p = m.predict_proba(X_val)[:, 1]
        preds += p
        prauc = average_precision_score(y_val, p)
        log(f'  Seed {seed}: iter={best}, val={prauc:.6f}')
        m.booster_.save_model(str(MODELS_OUT / f'lgbm_s{seed}.txt'))
        del m; gc.collect()
    preds /= len(seeds)
    prauc = average_precision_score(y_val, preds)
    log(f'LGBM ensemble: val={prauc:.6f}, iters={iters}')
    return preds


def train_xgb(X_train, y_train, X_val, y_val):
    """XGBoost with GPU, 50K trees."""
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    log(f'\n=== XGBoost: 50K trees, lr=0.01 ===')

    preds = np.zeros(len(X_val), dtype=np.float64)
    iters = []
    for seed in seeds:
        m = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='aucpr',
            device='cuda', tree_method='hist',
            n_estimators=50000, learning_rate=0.01,
            max_depth=7, min_child_weight=200,
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=0.5, reg_lambda=3.0,
            scale_pos_weight=spw,
            random_state=seed, n_jobs=4, verbosity=0,
            early_stopping_rounds=500,
        )
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
        best = m.best_iteration
        iters.append(best)
        p = m.predict_proba(X_val)[:, 1]
        preds += p
        prauc = average_precision_score(y_val, p)
        log(f'  Seed {seed}: iter={best}, val={prauc:.6f}')
        m.save_model(str(MODELS_OUT / f'xgb_s{seed}.json'))
        del m; gc.collect()
    preds /= len(seeds)
    prauc = average_precision_score(y_val, preds)
    log(f'XGBoost ensemble: val={prauc:.6f}, iters={iters}')
    return preds


def train_catboost(X_train, y_train, X_val, y_val):
    """CatBoost with GPU, 50K trees."""
    log(f'\n=== CatBoost: 50K trees, lr=0.01 ===')

    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    preds = np.zeros(len(X_val), dtype=np.float64)
    iters = []
    for seed in seeds:
        m = cb.CatBoostClassifier(
            iterations=50000, learning_rate=0.01,
            depth=7, l2_leaf_reg=5.0,
            bootstrap_type='Bernoulli', subsample=0.7,
            colsample_bylevel=0.6,
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
        iters.append(best)
        p = m.predict_proba(X_val)[:, 1]
        preds += p
        prauc = average_precision_score(y_val, p)
        log(f'  Seed {seed}: iter={best}, val={prauc:.6f}')
        m.save_model(str(MODELS_OUT / f'catboost_s{seed}.cbm'))
        del m; gc.collect()
    preds /= len(seeds)
    prauc = average_precision_score(y_val, preds)
    log(f'CatBoost ensemble: val={prauc:.6f}, iters={iters}')
    return preds


# ═══════════════════════════════════════════════════════════════════
# PHASE 4: Blend + Submit
# ═══════════════════════════════════════════════════════════════════

def optimize_blend(preds_list, y_val, names):
    """Find optimal rank-blend weights."""
    ranks = [rankdata(p) for p in preds_list]

    def neg_prauc(w):
        w = np.abs(w)
        w = w / w.sum()
        blend = sum(wi * ri for wi, ri in zip(w, ranks))
        return -average_precision_score(y_val, blend)

    n = len(preds_list)
    best_score = -1
    best_w = np.ones(n) / n

    for _ in range(100):
        w0 = np.random.dirichlet(np.ones(n))
        res = minimize(neg_prauc, w0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-6})
        if -res.fun > best_score:
            best_score = -res.fun
            best_w = np.abs(res.x)
            best_w = best_w / best_w.sum()

    log(f'Optimal weights: {dict(zip(names, best_w.round(4)))}')
    log(f'Optimized blend: val={best_score:.6f}')
    return best_w, best_score


def generate_submissions(available_feats, weights):
    """Generate test submissions."""
    log('\n=== Generating test submissions ===')

    # Load and prepare test data
    pretest_prof = pl.read_parquet(FEATURES_IN / 'pretest_profiles.parquet')
    mcc_rep = pl.read_parquet(FEATURES_IN / 'mcc_reputation.parquet')
    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')

    df_test = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    df_test = df_test.unique(subset=['event_id'], keep='first')
    df_test = v9.add_features(df_test)
    df_test = v9.add_customer_profiles(df_test, profiles)
    df_test = add_new_features(df_test, pretest_prof, mcc_rep, None, is_test=True)

    X_test = df_test.select(available_feats).to_pandas().astype(np.float32)
    event_ids = df_test['event_id'].to_numpy()
    log(f'Test: {X_test.shape}')

    # LGBM
    lgbm_preds = np.mean([
        lgb.Booster(model_file=str(MODELS_OUT / f'lgbm_s{s}.txt')).predict(X_test)
        for s in seeds], axis=0)

    # XGBoost
    xgb_preds = np.zeros(len(X_test), dtype=np.float64)
    for s in seeds:
        m = xgb.XGBClassifier()
        m.load_model(str(MODELS_OUT / f'xgb_s{s}.json'))
        xgb_preds += m.predict_proba(X_test)[:, 1]
        del m
    xgb_preds /= len(seeds)

    # CatBoost
    cb_preds = np.zeros(len(X_test), dtype=np.float64)
    for s in seeds:
        m = cb.CatBoostClassifier()
        m.load_model(str(MODELS_OUT / f'catboost_s{s}.cbm'))
        cb_preds += m.predict_proba(X_test)[:, 1]
        del m
    cb_preds /= len(seeds)

    # Blends
    ranks = [rankdata(lgbm_preds), rankdata(xgb_preds), rankdata(cb_preds)]
    opt_blend = sum(w * r for w, r in zip(weights, ranks))

    sample = pl.read_csv(DATA / 'sample_submit.csv')
    ts = datetime.now().strftime('%Y%m%d_%H%M')

    submissions = {
        'lgbm': lgbm_preds,
        'xgb': xgb_preds,
        'catboost': cb_preds,
        'blend': opt_blend,
    }

    for name, score in submissions.items():
        sub = pl.DataFrame({'event_id': event_ids, 'predict': score})
        sub = sample.select('event_id').join(sub, on='event_id', how='left')
        n_null = sub['predict'].is_null().sum()
        if n_null > 0:
            sub = sub.with_columns(pl.col('predict').fill_null(sub['predict'].drop_nulls().median()))
        path = SUBMIT_OUT / f'submit_v11_{name}_{ts}.csv'
        sub.write_csv(path)
        log(f'Saved: {path.name}')


if __name__ == '__main__':
    t0 = time.time()

    X_train, y_train, X_val, y_val, available_feats = load_data()

    # Train all models
    lgbm_preds = train_lgbm(X_train, y_train, X_val, y_val)
    xgb_preds = train_xgb(X_train, y_train, X_val, y_val)
    cb_preds = train_catboost(X_train, y_train, X_val, y_val)

    # Results
    log('\n=== INDIVIDUAL RESULTS ===')
    for name, p in [('LGBM', lgbm_preds), ('XGB', xgb_preds), ('CatBoost', cb_preds)]:
        log(f'{name}: val={average_precision_score(y_val, p):.6f}')

    equal = (rankdata(lgbm_preds) + rankdata(xgb_preds) + rankdata(cb_preds)) / 3
    log(f'Equal blend: val={average_precision_score(y_val, equal):.6f}')

    names = ['LGBM', 'XGB', 'CatBoost']
    weights, opt_score = optimize_blend([lgbm_preds, xgb_preds, cb_preds], y_val, names)

    # Save
    results = {
        'lgbm': float(average_precision_score(y_val, lgbm_preds)),
        'xgb': float(average_precision_score(y_val, xgb_preds)),
        'catboost': float(average_precision_score(y_val, cb_preds)),
        'equal_blend': float(average_precision_score(y_val, equal)),
        'optimized_blend': opt_score,
        'weights': dict(zip(names, weights.tolist())),
        'features': len(available_feats),
    }
    with open(MODELS_OUT / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    generate_submissions(available_feats, weights)

    log(f'\nTotal time: {(time.time()-t0)/60:.1f} min')
    log('DONE')
