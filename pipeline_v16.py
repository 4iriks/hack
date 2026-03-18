"""
Pipeline v16: Ablation — Pretest Signal + Drift.

Чистая ablation из 4 экспериментов:
A) v14-C baseline (reference, old profiles, 121 фич)
B) v14-C + 5 pretest anomaly фичей (additive, не трогаем old profiles)
C) v14-C + 3 drift фичи (additive)
D) Blended profiles (alpha * pretest + (1-alpha) * old)

Все эксперименты: LGBM + ES на полном val (523K) + 5 seeds.
Pretest deep profiles уже закэшированы.
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from pathlib import Path
from datetime import datetime
import gc, json, time, os, math

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


# ═══════════════════════════════════════════════════════════
# Import v9 + v14 utilities
# ═══════════════════════════════════════════════════════════

import importlib.util
spec9 = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
v9 = importlib.util.module_from_spec(spec9)
spec9.loader.exec_module(v9)

spec14 = importlib.util.spec_from_file_location("v14", ROOT / "pipeline_v14.py")
v14 = importlib.util.module_from_spec(spec14)
spec14.loader.exec_module(v14)

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


# ═══════════════════════════════════════════════════════════
# PRETEST FEATURES (additive — не трогаем old profiles)
# ═══════════════════════════════════════════════════════════

PRETEST_ANOMALY_FEATURES = [
    'pt_anom_amt_zscore',      # amount z-score vs PRETEST profile
    'pt_anom_amt_vs_median',   # amount / pretest median
    'pt_anom_hour_zscore',     # hour z-score vs pretest hour pattern
    'pt_anom_mcc_novel',       # new MCC in pretest period?
    'pt_anom_frequency',       # tx count vs pretest daily rate
]

DRIFT_FEATURES = [
    'drift_amt_mean_ratio',    # pretest_mean / old_mean
    'drift_frequency_ratio',   # pretest_tx_per_day / old_tx_per_day
    'drift_hour_shift',        # |pretest_hour_mean - old_hour_mean|
]


def add_pretest_anomaly_features(df, pretest_profs, pretest_mcc):
    """Add anomaly features computed from PRETEST profiles (additive to v14).

    These are EXTRA features on top of v14-C anomalies.
    v14-C anomalies use old profiles (177M rows).
    These use pretest profiles (14M rows, same epoch as test).
    """
    # Prefix pretest profile columns to avoid collision with old profiles
    pt_cols = {c: f'_pt_{c}' for c in pretest_profs.columns if c != 'customer_id'}
    pt_renamed = pretest_profs.rename(pt_cols)

    df = df.join(pt_renamed, on='customer_id', how='left')

    # Amount z-score vs pretest
    df = df.with_columns([
        ((pl.col('operaton_amt') - pl.col('_pt_prof_amt_mean').fill_null(pl.col('prof_amt_mean'))) /
         pl.col('_pt_prof_amt_std').fill_null(pl.col('prof_amt_std')).clip(lower_bound=1.0)
        ).alias('pt_anom_amt_zscore'),

        (pl.col('operaton_amt') /
         pl.col('_pt_prof_amt_median').fill_null(pl.col('prof_amt_median')).clip(lower_bound=1.0)
        ).alias('pt_anom_amt_vs_median'),
    ])

    # Hour z-score vs pretest
    if 'hour' not in df.columns:
        df = df.with_columns(
            pl.col('event_dttm').str.slice(11, 2).cast(pl.Int32).alias('hour')
        )
    df = df.with_columns([
        (((pl.col('hour') - pl.col('_pt_prof_hour_mean').fill_null(pl.col('prof_hour_mean'))).abs())
         .clip(upper_bound=12.0) /
         pl.col('_pt_prof_hour_std').fill_null(pl.col('prof_hour_std')).clip(lower_bound=0.1)
        ).alias('pt_anom_hour_zscore'),
    ])

    # MCC novelty in pretest
    if 'mcc_code' in df.columns:
        pt_mcc_cols = pretest_mcc.select(['customer_id', 'mcc_code',
            pl.col('mcc_n_tx_total').alias('_pt_mcc_n_tx')])
        df = df.join(pt_mcc_cols, on=['customer_id', 'mcc_code'], how='left')
        df = df.with_columns([
            pl.col('_pt_mcc_n_tx').is_null().cast(pl.Int8).alias('pt_anom_mcc_novel'),
        ])
        df = df.drop('_pt_mcc_n_tx')

    # Frequency anomaly vs pretest
    if 'cnt_24h' in df.columns:
        df = df.with_columns([
            (pl.col('cnt_24h') /
             pl.col('_pt_prof_tx_per_day').fill_null(pl.col('prof_tx_per_day')).clip(lower_bound=0.5)
            ).alias('pt_anom_frequency'),
        ])

    # Drop temp pretest profile columns
    pt_temp_cols = [c for c in df.columns if c.startswith('_pt_')]
    df = df.drop(pt_temp_cols)

    return df


def add_drift_features(df, pretest_profs, old_profs):
    """Add per-customer drift features (pretest vs old behavior)."""
    # Compute drift table
    pt_sub = pretest_profs.select([
        'customer_id',
        pl.col('prof_amt_mean').alias('_pt_amt_mean'),
        pl.col('prof_tx_per_day').alias('_pt_tx_per_day'),
        pl.col('prof_hour_mean').alias('_pt_hour_mean'),
    ])
    old_sub = old_profs.select([
        'customer_id',
        pl.col('prof_amt_mean').alias('_old_amt_mean'),
        pl.col('prof_tx_per_day').alias('_old_tx_per_day'),
        pl.col('prof_hour_mean').alias('_old_hour_mean'),
    ])

    drift = pt_sub.join(old_sub, on='customer_id', how='inner')
    drift = drift.with_columns([
        (pl.col('_pt_amt_mean') / pl.col('_old_amt_mean').clip(lower_bound=1.0))
            .alias('drift_amt_mean_ratio'),
        (pl.col('_pt_tx_per_day') / pl.col('_old_tx_per_day').clip(lower_bound=0.1))
            .alias('drift_frequency_ratio'),
        (pl.col('_pt_hour_mean') - pl.col('_old_hour_mean')).abs()
            .clip(upper_bound=12.0).alias('drift_hour_shift'),
    ]).select(['customer_id'] + DRIFT_FEATURES)

    df = df.join(drift, on='customer_id', how='left')
    # Fill nulls: ratio=1.0 (no change), shift=0.0
    df = df.with_columns([
        pl.col('drift_amt_mean_ratio').fill_null(1.0),
        pl.col('drift_frequency_ratio').fill_null(1.0),
        pl.col('drift_hour_shift').fill_null(0.0),
    ])
    return df


def build_blended_profiles(pretest_profs, old_profs):
    """Blend pretest and old profiles: alpha weighted by pretest sample size.

    alpha = min(pretest_n_tx / 200, 0.6) — more pretest data → more weight, max 60%
    Gives stability of old profiles + recency of pretest.
    """
    log('Building blended profiles...')

    # Columns to blend (numeric profile columns)
    blend_cols = [
        'prof_amt_mean', 'prof_amt_std', 'prof_amt_median',
        'prof_amt_p10', 'prof_amt_p25', 'prof_amt_p75', 'prof_amt_p90',
        'prof_amt_p95', 'prof_amt_p99', 'prof_amt_max', 'prof_amt_min',
        'prof_hour_mean', 'prof_hour_std', 'prof_pct_night', 'prof_pct_weekend',
        'prof_voip_rate', 'prof_rdp_rate', 'prof_tx_per_day', 'prof_amt_cv',
        'prof_gap_mean', 'prof_gap_std',
    ]

    # Rename for join
    pt_rename = {c: f'_pt_{c}' for c in pretest_profs.columns if c != 'customer_id'}
    pt = pretest_profs.rename(pt_rename)

    merged = old_profs.join(pt, on='customer_id', how='left')

    # Alpha = min(pretest_n_tx / 200, 0.6), 0 if no pretest
    merged = merged.with_columns(
        pl.when(pl.col('_pt_prof_n_tx').is_not_null())
        .then((pl.col('_pt_prof_n_tx') / 200.0).clip(upper_bound=0.6))
        .otherwise(0.0)
        .alias('_alpha')
    )

    n_blended = merged.filter(pl.col('_alpha') > 0).height
    log(f'  Customers with pretest data: {n_blended:,} / {merged.height:,}')
    log(f'  Alpha stats: mean={merged["_alpha"].mean():.3f}, max={merged["_alpha"].max():.3f}')

    # Blend each column
    for col in blend_cols:
        pt_col = f'_pt_{col}'
        if pt_col in merged.columns:
            merged = merged.with_columns(
                pl.when(pl.col(pt_col).is_not_null())
                .then(pl.col('_alpha') * pl.col(pt_col) + (1.0 - pl.col('_alpha')) * pl.col(col))
                .otherwise(pl.col(col))
                .alias(col)
            )

    # Keep non-blended columns from old (entropy, top_mcc, etc.)
    # Drop temp columns
    drop_cols = [c for c in merged.columns if c.startswith('_pt_') or c == '_alpha']
    blended = merged.drop(drop_cols)

    log(f'  Blended profiles: {blended.height:,} customers, {len(blended.columns)} cols')
    return blended


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════

def train_model(name, X_train, y_train, X_val, y_val, n_trees=10000, patience=300):
    """Train 5-seed LGBM ensemble with early stopping on full val."""
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    spw = min(spw, 50.0)

    log(f'\n{"="*60}')
    log(f'Training: {name}')
    log(f'Features: {X_train.shape[1]}, Trees: {n_trees}, spw: {spw:.1f}')
    log(f'{"="*60}')

    preds = np.zeros(len(X_val), dtype=np.float64)
    seed_results = []

    for i, seed in enumerate(seeds):
        m = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed,
            scale_pos_weight=spw, n_estimators=n_trees)
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)])
        best_iter = m.best_iteration_
        p = m.predict_proba(X_val)[:, 1]
        preds += p
        prauc = average_precision_score(y_val, p)
        seed_results.append(prauc)
        log(f'  Seed {seed}: iter={best_iter}, val={prauc:.6f}')

        model_path = MODELS_OUT / f'{name}_s{seed}.txt'
        m.booster_.save_model(str(model_path))
        del m; gc.collect()

    preds /= len(seeds)
    ensemble_prauc = average_precision_score(y_val, preds)
    log(f'  Ensemble ({name}): val={ensemble_prauc:.6f}')

    return ensemble_prauc, preds, seed_results


def generate_submission(name, feats):
    """Generate submission for an experiment."""
    test_df = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    test_df = test_df.unique(subset=['event_id'], keep='first')
    test_df = v9.add_features(test_df)
    test_df = v9.add_customer_profiles(test_df, old_profiles)

    # Add anomaly features based on experiment
    if name.startswith('D_'):
        # Blended profiles
        test_df = v14.add_anomaly_features(test_df, blended_profs, old_mcc_profs)
    else:
        # Old profiles (standard v14-C)
        test_df = v14.add_anomaly_features(test_df, old_deep_profs, old_mcc_profs)

    # Add pretest anomaly features if needed
    if any(f.startswith('pt_anom_') for f in feats):
        test_df = add_pretest_anomaly_features(test_df, pretest_profs, pretest_mcc)

    # Add drift features if needed
    if any(f.startswith('drift_') for f in feats):
        test_df = add_drift_features(test_df, pretest_profs, old_deep_profs)

    available_feats = [f for f in feats if f in test_df.columns]
    X_test = test_df.select(available_feats).to_pandas().astype(np.float32)
    event_ids = test_df['event_id'].to_numpy()

    preds = np.mean([
        lgb.Booster(model_file=str(MODELS_OUT / f'{name}_s{s}.txt')).predict(X_test)
        for s in seeds], axis=0)

    sample = pl.read_csv(DATA / 'sample_submit.csv')
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    sub = pl.DataFrame({'event_id': event_ids, 'predict': preds.astype(np.float64)})
    sub = sample.select('event_id').join(sub, on='event_id', how='left')
    n_null = sub['predict'].is_null().sum()
    if n_null > 0:
        sub = sub.with_columns(pl.col('predict').fill_null(sub['predict'].drop_nulls().median()))

    path = SUBMIT_OUT / f'submit_v16_{name}_{ts}.csv'
    sub.write_csv(path)
    log(f'Saved: {path.name}')
    return path


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_start = time.time()
    log('=== PIPELINE V16: ABLATION — PRETEST SIGNAL + DRIFT ===')

    # ── Load all profiles ──
    log('Loading profiles...')
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

    # ── Blended profiles ──
    blended_profs = build_blended_profiles(pretest_profs, old_deep_profs)
    gc.collect()

    # ── Load train/val ──
    log('\nLoading data...')
    val_df = pl.read_parquet(FEATURES_IN / 'val_proper.parquet')
    val_df = val_df.with_columns(
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'))
    val_df = v9.add_features(val_df)
    val_df = v9.add_customer_profiles(val_df, old_profiles)

    val_ids = set(pl.read_parquet(FEATURES_IN / 'val_event_ids.parquet')['event_id'].to_list())
    train_df = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    train_df = train_df.filter(~pl.col('event_id').is_in(val_ids))
    train_df = v9.add_features(train_df)
    train_df = v9.add_customer_profiles(train_df, old_profiles)

    y_train = train_df['target'].to_numpy().astype(int)
    y_val = val_df['is_fraud'].to_numpy().astype(int)

    log(f'Train: {len(train_df):,}, Val: {len(val_df):,}, Val fraud: {y_val.sum()}')
    log(f'RAM: {_ram_gb():.1f}GB')

    # ═══════════════════════════════════════════════════════
    # SEQUENTIAL EXPERIMENTS (no .clone() — save RAM)
    # Skip baseline: v14-C already known (val=0.044, LB=0.101)
    # ═══════════════════════════════════════════════════════

    # Add v14 anomaly features IN-PLACE (needed for all experiments)
    log('\n>>> Adding v14 anomaly features <<<')
    train_df = v14.add_anomaly_features(train_df, old_deep_profs, old_mcc_profs)
    val_df = v14.add_anomaly_features(val_df, old_deep_profs, old_mcc_profs)
    gc.collect()

    base_feats = [c for c in v9.FEATURE_COLS if c in train_df.columns and c in val_df.columns]
    anom_feats = [c for c in v14.ANOMALY_FEATURES if c in train_df.columns and c in val_df.columns]
    feats_a = base_feats + anom_feats
    log(f'  v14-C feature set: {len(feats_a)} features')

    results = {}
    prauc_ref = 0.044  # v14-C val reference

    # ── EXP B: v14-C + 5 pretest anomaly features ──
    log('\n>>> EXP B: + PRETEST ANOMALY FEATURES <<<')
    train_df = add_pretest_anomaly_features(train_df, pretest_profs, pretest_mcc)
    val_df = add_pretest_anomaly_features(val_df, pretest_profs, pretest_mcc)

    pt_feats = [c for c in PRETEST_ANOMALY_FEATURES if c in train_df.columns and c in val_df.columns]
    feats_b = feats_a + pt_feats
    log(f'  Features: {len(feats_a)} base+anom + {len(pt_feats)} pretest = {len(feats_b)}')

    X_tr = train_df.select(feats_b).to_pandas().astype(np.float32)
    X_va = val_df.select(feats_b).to_pandas().astype(np.float32)
    prauc_b, _, _ = train_model('B_plus_pretest', X_tr, y_train, X_va, y_val)
    results['B_plus_pretest'] = prauc_b
    del X_tr, X_va; gc.collect()

    # ── EXP C: v14-C + 3 drift features ──
    log('\n>>> EXP C: + DRIFT FEATURES <<<')
    train_df = add_drift_features(train_df, pretest_profs, old_deep_profs)
    val_df = add_drift_features(val_df, pretest_profs, old_deep_profs)

    dr_feats = [c for c in DRIFT_FEATURES if c in train_df.columns and c in val_df.columns]
    feats_c = feats_a + dr_feats
    log(f'  Features: {len(feats_a)} base+anom + {len(dr_feats)} drift = {len(feats_c)}')

    X_tr = train_df.select(feats_c).to_pandas().astype(np.float32)
    X_va = val_df.select(feats_c).to_pandas().astype(np.float32)
    prauc_c, _, _ = train_model('C_plus_drift', X_tr, y_train, X_va, y_val)
    results['C_plus_drift'] = prauc_c
    del X_tr, X_va; gc.collect()

    # ── EXP BC: v14-C + pretest + drift (all additive) ──
    log('\n>>> EXP BC: + PRETEST + DRIFT <<<')
    feats_bc = feats_a + pt_feats + dr_feats
    log(f'  Features: {len(feats_bc)} total')

    X_tr = train_df.select(feats_bc).to_pandas().astype(np.float32)
    X_va = val_df.select(feats_bc).to_pandas().astype(np.float32)
    prauc_bc, _, _ = train_model('BC_pretest_drift', X_tr, y_train, X_va, y_val)
    results['BC_pretest_drift'] = prauc_bc
    del X_tr, X_va; gc.collect()

    # Free old DataFrames, rebuild for D
    del train_df, val_df; gc.collect()

    # ── EXP D: Blended profiles (rebuild from scratch) ──
    log('\n>>> EXP D: BLENDED PROFILES <<<')
    val_df_d = pl.read_parquet(FEATURES_IN / 'val_proper.parquet')
    val_df_d = val_df_d.with_columns(
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'))
    val_df_d = v9.add_features(val_df_d)
    val_df_d = v9.add_customer_profiles(val_df_d, old_profiles)
    val_df_d = v14.add_anomaly_features(val_df_d, blended_profs, old_mcc_profs)

    train_df_d = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    train_df_d = train_df_d.filter(~pl.col('event_id').is_in(val_ids))
    train_df_d = v9.add_features(train_df_d)
    train_df_d = v9.add_customer_profiles(train_df_d, old_profiles)
    train_df_d = v14.add_anomaly_features(train_df_d, blended_profs, old_mcc_profs)

    feats_d = [c for c in feats_a if c in train_df_d.columns and c in val_df_d.columns]
    log(f'  Features: {len(feats_d)} (same as v14-C but blended profiles)')

    y_train_d = train_df_d['target'].to_numpy().astype(int)
    y_val_d = val_df_d['is_fraud'].to_numpy().astype(int)
    X_tr = train_df_d.select(feats_d).to_pandas().astype(np.float32)
    X_va = val_df_d.select(feats_d).to_pandas().astype(np.float32)
    prauc_d, _, _ = train_model('D_blended', X_tr, y_train_d, X_va, y_val_d)
    results['D_blended'] = prauc_d
    del X_tr, X_va, train_df_d, val_df_d; gc.collect()

    # ═══════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════
    results['A_baseline_ref'] = prauc_ref  # v14-C known value

    log(f'\n{"="*60}')
    log('ИТОГИ V16 ABLATION')
    log(f'{"="*60}')

    sorted_res = sorted(results.items(), key=lambda x: -x[1])
    for name, prauc in sorted_res:
        delta = 100 * (prauc / prauc_ref - 1)
        marker = ' <<<' if prauc == sorted_res[0][1] else ''
        log(f'{name:25s}: val={prauc:.6f} ({delta:+.1f}% vs v14-C ref){marker}')

    with open(MODELS_OUT / 'results_v16.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate submissions for trained experiments
    log('\n=== ГЕНЕРАЦИЯ САБМИТОВ ===')
    feat_map = {
        'B_plus_pretest': feats_b,
        'C_plus_drift': feats_c,
        'BC_pretest_drift': feats_bc,
        'D_blended': feats_d,
    }
    for name, prauc in sorted_res:
        if name in feat_map:
            generate_submission(name, feat_map[name])

    total_min = (time.time() - t_start) / 60
    log(f'\nВсего: {total_min:.0f} мин')
    log('DONE')
