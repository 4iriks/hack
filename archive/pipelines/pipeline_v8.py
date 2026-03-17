"""
Pipeline v8: Full Data Training — 2.6M rows (8x more than v6/v7).

KEY CHANGE: Training on 50:1 green:fraud ratio (2.6M rows) instead of 5:1 (344K).
Model sees 10x more "normal" transaction patterns → better discrimination.

Features: v3f base (41) + best customer profile features (14) + proven interactions = ~60
Strategy: labeled+green (proven on LB)
Models: LGBM + XGB + CatBoost, 5 seeds, GPU
Trees: n_estimators=10000, early_stopping=300 (more epochs, let model converge)
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize
from scipy.stats import rankdata
from pathlib import Path
from datetime import datetime
import gc, json, time, os

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_OUT  = ROOT / 'models_v8'
SUBMIT_OUT  = ROOT / 'submissions'

for d in [MODELS_OUT, SUBMIT_OUT]:
    d.mkdir(exist_ok=True)


def _ram_gb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1048576
    except:
        return 0.0
    return 0.0

def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")} RAM:{_ram_gb():.1f}GB] {msg}', flush=True)

def check_resources():
    """Проверка ресурсов перед тяжёлой операцией."""
    ram = _ram_gb()
    if ram > 20:
        log(f'WARNING: RAM usage {ram:.1f}GB > 20GB, running gc...')
        gc.collect()
    return ram


# ═══════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════

def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Feature engineering — proven features only."""
    new_cols = []

    # --- Device/session flags ---
    if 'browser_language' in df.columns:
        new_cols.append(pl.col('browser_language').is_not_null().cast(pl.Int8).alias('has_browser_lang'))
    if 'accept_language' in df.columns:
        new_cols.append(pl.col('accept_language').is_not_null().cast(pl.Int8).alias('has_accept_lang'))
    if 'screen_size' in df.columns:
        new_cols.extend([
            pl.col('screen_size').str.split('x').list.first().cast(pl.Int32, strict=False).alias('screen_w'),
            pl.col('screen_size').str.split('x').list.last().cast(pl.Int32, strict=False).alias('screen_h'),
            pl.col('screen_size').is_null().cast(pl.Int8).alias('is_no_screen'),
        ])
    if 'device_system_version' in df.columns:
        new_cols.append(pl.col('device_system_version').is_not_null().cast(pl.Int8).alias('has_device_ver'))
    if 'session_id' in df.columns:
        new_cols.append(pl.col('session_id').is_not_null().cast(pl.Int8).alias('has_session'))
    if new_cols:
        df = df.with_columns(new_cols)

    # --- POS code features ---
    if 'pos_cd' in df.columns:
        df = df.with_columns([
            (pl.col('pos_cd') == 1).fill_null(False).cast(pl.Int8).alias('is_manual_entry'),
            (pl.col('pos_cd') == 3).fill_null(False).cast(pl.Int8).alias('is_pos_cd_3'),
            pl.col('pos_cd').is_not_null().cast(pl.Int8).alias('has_pos_data'),
        ])

    # --- timezone null ---
    if 'timezone' in df.columns:
        df = df.with_columns([
            pl.col('timezone').is_null().cast(pl.Int8).alias('is_no_timezone'),
        ])

    # --- Velocity ratios ---
    ratio_cols = []
    for short, long in [('1h','24h'), ('6h','24h'), ('24h','7d'), ('7d','30d'), ('1h','6h')]:
        cs, cl = f'cnt_{short}', f'cnt_{long}'
        if cs in df.columns and cl in df.columns:
            ratio_cols.append((pl.col(cs) / (pl.col(cl) + 1)).alias(f'cnt_ratio_{short}_{long}'))
    for short, long in [('1h','24h'), ('24h','7d'), ('7d','30d')]:
        cs, cl = f'amt_sum_{short}', f'amt_sum_{long}'
        if cs in df.columns and cl in df.columns:
            ratio_cols.append((pl.col(cs) / (pl.col(cl).abs() + 1)).alias(f'amt_ratio_{short}_{long}'))
    if 'operaton_amt' in df.columns:
        for w in ['24h', '7d', '30d']:
            c = f'amt_sum_{w}'
            if c in df.columns:
                ratio_cols.append((pl.col('operaton_amt') / (pl.col(c).abs() + 1)).alias(f'amt_cur_ratio_{w}'))
    if ratio_cols:
        df = df.with_columns(ratio_cols)

    # --- Time features ---
    time_cols = []
    if 'hour' in df.columns and 'weekday' in df.columns:
        time_cols.append((pl.col('hour') * 7 + pl.col('weekday')).alias('hour_weekday'))
    if 'hour' in df.columns:
        time_cols.extend([
            (np.pi * 2 * pl.col('hour').cast(pl.Float32) / 24).sin().alias('hour_sin'),
            (np.pi * 2 * pl.col('hour').cast(pl.Float32) / 24).cos().alias('hour_cos'),
        ])
    if 'event_dttm' in df.columns:
        time_cols.append(pl.col('event_dttm').dt.day().cast(pl.Int8).alias('day_of_month'))
    if time_cols:
        df = df.with_columns(time_cols)

    # --- Security ---
    if all(c in df.columns for c in ['compromised','web_rdp_connection','phone_voip_call_state','developer_tools']):
        df = df.with_columns([
            (pl.col('compromised') * 3 + pl.col('web_rdp_connection') * 2 +
             pl.col('phone_voip_call_state') * 2 + pl.col('developer_tools')).alias('security_risk_score'),
        ])

    # --- Behavioral deviation ---
    v4_cols = []
    if 'secs_since_last' in df.columns:
        v4_cols.extend([
            (pl.col('secs_since_last') < 60).cast(pl.Int8).alias('is_fast_60s'),
            (pl.col('secs_since_last') < 300).cast(pl.Int8).alias('is_fast_300s'),
            pl.col('secs_since_last').log1p().alias('log_secs_since_last'),
        ])
    if 'amt_sum_30d' in df.columns and 'cnt_30d' in df.columns:
        v4_cols.append((pl.col('amt_sum_30d') / (pl.col('cnt_30d') + 1)).alias('avg_amt_30d'))
    if 'amt_sum_7d' in df.columns and 'cnt_7d' in df.columns:
        v4_cols.append((pl.col('amt_sum_7d') / (pl.col('cnt_7d') + 1)).alias('avg_amt_7d'))
    if v4_cols:
        df = df.with_columns(v4_cols)

    dev_cols = []
    if 'avg_amt_30d' in df.columns and 'operaton_amt' in df.columns:
        dev_cols.append((pl.col('operaton_amt') / (pl.col('avg_amt_30d') + 1)).alias('amt_deviation_30d'))
    if 'avg_amt_7d' in df.columns and 'operaton_amt' in df.columns:
        dev_cols.append((pl.col('operaton_amt') / (pl.col('avg_amt_7d') + 1)).alias('amt_deviation_7d'))
    if dev_cols:
        df = df.with_columns(dev_cols)

    if 'cnt_1h' in df.columns:
        df = df.with_columns([
            (pl.col('cnt_1h') > 3).cast(pl.Int8).alias('activity_burst_1h'),
        ])
    if 'cnt_1h' in df.columns and 'cnt_24h' in df.columns:
        df = df.with_columns([
            (pl.col('cnt_1h') / (pl.col('cnt_24h') / 24 + 0.01)).alias('hourly_activity_ratio'),
        ])

    # --- Interactions (only proven ones) ---
    if 'phone_voip_call_state' in df.columns and 'log_amount' in df.columns:
        df = df.with_columns([
            (pl.col('phone_voip_call_state') * pl.col('log_amount')).alias('voip_x_logamt'),
        ])
    if 'is_high_risk_type' in df.columns and 'log_amount' in df.columns:
        df = df.with_columns([
            (pl.col('is_high_risk_type') * pl.col('log_amount')).alias('high_risk_x_logamt'),
        ])
    if 'session_ops_before' in df.columns and 'log_amount' in df.columns:
        df = df.with_columns([
            (pl.col('session_ops_before') * pl.col('log_amount')).alias('session_ops_x_log_amt'),
        ])
    if 'operaton_amt' in df.columns and 'amt_sum_30d' in df.columns:
        df = df.with_columns([
            (pl.col('operaton_amt') / (pl.col('amt_sum_30d') + 1)).alias('amt_pct_of_30d'),
        ])
    if 'session_ops_before' in df.columns:
        df = df.with_columns([
            (pl.col('session_ops_before') == 0).cast(pl.Int8).alias('is_first_in_session'),
        ])
    if 'cnt_30d' in df.columns:
        df = df.with_columns([
            pl.col('cnt_30d').cast(pl.Float32).log1p().alias('log_cnt_30d'),
        ])

    # --- v6 PDF: only manual_entry_x_amt (the only one with real importance) ---
    if 'is_manual_entry' in df.columns and 'log_amount' in df.columns:
        df = df.with_columns([
            (pl.col('is_manual_entry') * pl.col('log_amount')).alias('manual_entry_x_amt'),
        ])

    return df


def add_customer_profile_features(df: pl.DataFrame, profiles: pl.DataFrame) -> pl.DataFrame:
    """Join customer profiles — only features with proven importance."""
    df = df.join(profiles, on='customer_id', how='left')

    deviation_cols = []
    if 'cust_avg_amt' in df.columns and 'operaton_amt' in df.columns:
        deviation_cols.extend([
            ((pl.col('operaton_amt') - pl.col('cust_avg_amt')) /
             (pl.col('cust_std_amt') + 1)).alias('amt_zscore'),
            (pl.col('operaton_amt') / (pl.col('cust_med_amt') + 1)).alias('amt_vs_median'),
        ])

    if deviation_cols:
        df = df.with_columns(deviation_cols)

    # Dormancy
    if 'cust_last_epoch' in df.columns and 'event_dttm' in df.columns:
        df = df.with_columns([
            ((pl.col('event_dttm').dt.epoch('s') - pl.col('cust_last_epoch')) / 86400)
                .alias('dormancy_days'),
        ])

    # Hour deviation
    if 'cust_avg_hour' in df.columns and 'hour' in df.columns:
        df = df.with_columns([
            (pl.col('hour').cast(pl.Float64) - pl.col('cust_avg_hour')).abs().alias('hour_deviation'),
        ])

    # Activity frequency ratio
    if 'cnt_30d' in df.columns and 'cust_avg_gap_sec' in df.columns:
        df = df.with_columns([
            (pl.col('cnt_30d').cast(pl.Float64) /
             (2592000 / (pl.col('cust_avg_gap_sec') + 1) + 0.01)).alias('freq_vs_historical'),
        ])

    # Fill nulls
    profile_cols = [c for c in df.columns if c.startswith('cust_') or
                    c in ('amt_zscore', 'amt_vs_median', 'dormancy_days',
                           'hour_deviation', 'freq_vs_historical')]
    for c in profile_cols:
        if c in df.columns:
            df = df.with_columns(pl.col(c).fill_null(0.0))

    return df


# Features to use — curated list, no zero-importance junk
FEATURE_COLS = [
    # Original base (from run_features.py)
    'hour', 'weekday', 'month', 'is_night',
    'log_amount', 'operaton_amt',
    'phone_voip_call_state', 'web_rdp_connection', 'compromised',
    'developer_tools', 'security_flags_sum',
    'event_type_nm', 'is_high_risk_type',
    'mcc_code', 'is_null_mcc',
    'channel_indicator_type', 'channel_indicator_sub_type', 'currency_iso_cd',
    'operating_system_type', 'pos_cd',
    'cnt_1h', 'cnt_6h', 'cnt_24h', 'cnt_7d', 'cnt_30d',
    'amt_sum_1h', 'amt_sum_6h', 'amt_sum_24h', 'amt_sum_7d', 'amt_sum_30d',
    'secs_since_last', 'voip_cnt_24h',
    'is_new_mcc_code', 'is_new_channel_indicator_type',
    'cum_unique_mcc_approx',
    'session_ops_before', 'session_amt_before',
    'timezone',
    # v2 enriched
    'event_desc',
    'has_browser_lang', 'has_accept_lang', 'screen_w', 'screen_h',
    'has_device_ver', 'has_session',
    'cnt_ratio_1h_24h', 'cnt_ratio_6h_24h', 'cnt_ratio_24h_7d',
    'cnt_ratio_7d_30d', 'cnt_ratio_1h_6h',
    'amt_ratio_1h_24h', 'amt_ratio_24h_7d', 'amt_ratio_7d_30d',
    'amt_cur_ratio_24h', 'amt_cur_ratio_7d', 'amt_cur_ratio_30d',
    'hour_weekday', 'hour_sin', 'hour_cos', 'day_of_month',
    'security_risk_score', 'session_ops_x_log_amt',
    # v5 behavioral
    'is_fast_60s', 'is_fast_300s', 'log_secs_since_last',
    'avg_amt_30d', 'avg_amt_7d',
    'amt_deviation_30d', 'amt_deviation_7d',
    'activity_burst_1h', 'hourly_activity_ratio',
    'voip_x_logamt', 'high_risk_x_logamt',
    'amt_pct_of_30d', 'is_first_in_session', 'log_cnt_30d',
    'is_manual_entry', 'is_pos_cd_3', 'has_pos_data',
    'is_no_screen', 'is_no_timezone',
    # v6 (only the one that works)
    'manual_entry_x_amt',
    # Customer profiles (top importance from v7)
    'cust_avg_amt', 'cust_std_amt', 'cust_med_amt', 'cust_max_amt', 'cust_p95_amt',
    'cust_n_tx', 'cust_n_unique_mcc', 'cust_n_unique_channel',
    'cust_avg_hour', 'cust_hour_std', 'cust_pct_night', 'cust_pct_high_risk',
    'cust_tenure_days', 'cust_avg_gap_sec',
    # Customer deviation features
    'amt_zscore', 'amt_vs_median',
    'dormancy_days', 'hour_deviation', 'freq_vs_historical',
]


def optimize_rank_weights(preds_list, y_true, names):
    n = len(preds_list)
    ranks = [rankdata(p) for p in preds_list]
    def neg_prauc(w):
        w = np.abs(w); w = w / w.sum()
        return -average_precision_score(y_true, sum(w[i]*ranks[i] for i in range(n)))
    best_s, best_w = -1, None
    for seed in range(50):
        w0 = np.random.RandomState(seed).dirichlet(np.ones(n))
        res = minimize(neg_prauc, w0, method='Nelder-Mead', options={'maxiter': 3000})
        if -res.fun > best_s:
            best_s = -res.fun; best_w = np.abs(np.array(res.x))
    best_w = best_w / best_w.sum()
    log(f'Rank weights: {dict(zip(names, best_w.round(4)))}  PR-AUC={best_s:.4f}')
    return dict(zip(names, best_w.tolist())), best_s


def main():
    t_start = time.time()
    print('\n' + '='*60)
    print('PIPELINE V8: Full Data Training (2.6M rows)')
    print('='*60)

    check_resources()

    # ─── Load customer profiles ───
    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    log(f'Profiles: {len(profiles):,} customers')

    # ─── Load FULL training data ───
    log('Loading full train features (2.6M rows)...')
    df_all = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    log(f'Loaded: {len(df_all):,} rows, {df_all.estimated_size("mb"):.0f} MB')
    check_resources()

    df_all = add_features(df_all)
    df_all = add_customer_profile_features(df_all, profiles)

    available_feats = [c for c in FEATURE_COLS if c in df_all.columns]
    log(f'Features: {len(available_feats)} / {len(FEATURE_COLS)}')

    # ─── Train/val split (temporal) ───
    val_dt = datetime(2025, 4, 1)
    df_tr = df_all.filter(pl.col('event_dttm') < val_dt)
    df_val = df_all.filter(pl.col('event_dttm') >= val_dt)

    # Convert to pandas for training
    log('Converting to pandas...')
    X_train = df_tr.select(available_feats).to_pandas().astype(np.float32)
    y_train = df_tr['target'].to_numpy().astype(int)
    X_val = df_val.select(available_feats).to_pandas().astype(np.float32)
    y_val = df_val['target'].to_numpy().astype(int)

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    spw = n_neg / n_pos
    log(f'Train: {len(X_train):,} (pos={n_pos:,}, neg={n_neg:,}, spw={spw:.1f})')
    log(f'Val: {len(X_val):,} (fraud={y_val.sum():,})')

    del df_tr, df_val; gc.collect()
    check_resources()

    # ─── Multi-seed training ───
    seeds = [42, 123, 777, 2024, 31337]
    all_lgbm_p, all_xgb_p, all_cat_p = [], [], []
    best_iters = {'l': [], 'x': [], 'c': []}

    for si, seed in enumerate(seeds):
        log(f'\n--- Seed {seed} ({si+1}/{len(seeds)}) ---')
        check_resources()

        # LightGBM — GPU, 10K trees, patient early stopping
        m_l = lgb.LGBMClassifier(
            objective='binary', metric='average_precision',
            device='gpu', gpu_platform_id=0, gpu_device_id=0,
            scale_pos_weight=spw,
            n_estimators=10000, learning_rate=0.02,
            num_leaves=255, min_child_samples=50,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
            reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
            random_state=seed, n_jobs=4, verbose=-1,
        )
        m_l.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(300, verbose=True), lgb.log_evaluation(200)])
        lp = m_l.predict_proba(X_val)[:, 1]
        ls = average_precision_score(y_val, lp)
        log(f'LGBM[{seed}]: {ls:.4f} (iter={m_l.best_iteration_})')
        all_lgbm_p.append(lp)
        best_iters['l'].append(m_l.best_iteration_)
        del m_l; gc.collect()

        # XGBoost — GPU
        m_x = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='aucpr',
            tree_method='hist', device='cuda',
            scale_pos_weight=spw,
            n_estimators=10000, learning_rate=0.02,
            max_depth=8, min_child_weight=50,
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
            random_state=seed, n_jobs=4, verbosity=0,
            early_stopping_rounds=300,
        )
        m_x.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)
        xp = m_x.predict_proba(X_val)[:, 1]
        xs = average_precision_score(y_val, xp)
        log(f'XGB[{seed}]: {xs:.4f} (iter={m_x.best_iteration})')
        all_xgb_p.append(xp)
        best_iters['x'].append(m_x.best_iteration)
        del m_x; gc.collect()

        # CatBoost — GPU
        m_c = CatBoostClassifier(
            iterations=10000, learning_rate=0.02, depth=8,
            task_type='GPU', devices='0',
            loss_function='Logloss', eval_metric='PRAUC',
            auto_class_weights='Balanced', l2_leaf_reg=5.0,
            random_seed=seed, verbose=200, early_stopping_rounds=300,
        )
        m_c.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        cp_pred = m_c.predict_proba(X_val)[:, 1]
        cs = average_precision_score(y_val, cp_pred)
        log(f'CAT[{seed}]: {cs:.4f} (iter={m_c.best_iteration_})')
        all_cat_p.append(cp_pred)
        best_iters['c'].append(m_c.best_iteration_)
        del m_c; gc.collect()

    # ─── Ensemble optimization ───
    avg_l = np.mean(all_lgbm_p, axis=0)
    avg_x = np.mean(all_xgb_p, axis=0)
    avg_c = np.mean(all_cat_p, axis=0)

    log(f'\nAveraged: LGBM={average_precision_score(y_val, avg_l):.4f}'
        f'  XGB={average_precision_score(y_val, avg_x):.4f}'
        f'  CAT={average_precision_score(y_val, avg_c):.4f}')

    weights, ens_score = optimize_rank_weights(
        [avg_l, avg_x, avg_c], y_val, ['lgbm', 'xgb', 'catboost'])

    config = {
        'strategy': 'labeled_green_full_data_v8',
        'green_ratio': '50:1',
        'train_rows': len(X_train),
        'weights': weights,
        'ensemble_prauc': ens_score,
        'best_iters': best_iters,
        'n_features': len(available_feats),
        'n_seeds': len(seeds),
    }
    with open(MODELS_OUT / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    log(f'Config saved')

    del X_train, y_train, X_val, y_val; gc.collect()
    check_resources()

    # ─── Final models on ALL data ───
    log('\nTraining final models on ALL data...')

    X_full = df_all.select(available_feats).to_pandas().astype(np.float32)
    y_full = df_all['target'].to_numpy().astype(int)

    n_pos_full = (y_full == 1).sum()
    n_neg_full = (y_full == 0).sum()
    spw_full = n_neg_full / n_pos_full
    log(f'Full data: {len(X_full):,} (pos={n_pos_full:,}, neg={n_neg_full:,})')

    del df_all; gc.collect()
    check_resources()

    # Use avg best_iter * 1.1 for final training (no early stopping)
    avg_li = int(np.mean(best_iters['l']) * 1.1)
    avg_xi = int(np.mean(best_iters['x']) * 1.1)
    avg_ci = int(np.mean(best_iters['c']) * 1.1)
    log(f'Final iters: LGBM={avg_li}, XGB={avg_xi}, CAT={avg_ci}')

    for si, seed in enumerate(seeds):
        log(f'Final seed {seed} ({si+1}/{len(seeds)})...')
        check_resources()

        fl = lgb.LGBMClassifier(
            objective='binary', metric='average_precision',
            device='gpu', gpu_platform_id=0, gpu_device_id=0,
            scale_pos_weight=spw_full, n_estimators=avg_li, learning_rate=0.02,
            num_leaves=255, min_child_samples=50,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
            reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
            random_state=seed, n_jobs=4, verbose=-1)
        fl.fit(X_full, y_full)
        fl.booster_.save_model(str(MODELS_OUT / f'lgbm_s{seed}.txt'))
        del fl; gc.collect()

        fx = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='aucpr',
            tree_method='hist', device='cuda',
            scale_pos_weight=spw_full, n_estimators=avg_xi, learning_rate=0.02,
            max_depth=8, min_child_weight=50,
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
            random_state=seed, n_jobs=4, verbosity=0)
        fx.fit(X_full, y_full)
        fx.save_model(str(MODELS_OUT / f'xgb_s{seed}.json'))
        del fx; gc.collect()

        fc = CatBoostClassifier(
            iterations=avg_ci, learning_rate=0.02, depth=8,
            task_type='GPU', devices='0',
            loss_function='Logloss', auto_class_weights='Balanced',
            l2_leaf_reg=5.0, random_seed=seed, verbose=0)
        fc.fit(X_full, y_full)
        fc.save_model(str(MODELS_OUT / f'cat_s{seed}.cbm'))
        del fc; gc.collect()

    del X_full, y_full; gc.collect()
    check_resources()

    # ─── Predict test ───
    log('\nPredicting test...')
    df_test = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    n_before = len(df_test)
    df_test = df_test.unique(subset=['event_id'], keep='first')
    log(f'Dedup: {n_before} -> {len(df_test)} (-{n_before - len(df_test)})')

    df_test = add_features(df_test)
    df_test = add_customer_profile_features(df_test, profiles)
    X_test = df_test.select(available_feats).to_pandas().astype(np.float32)
    event_ids = df_test['event_id'].to_numpy()
    log(f'X_test: {X_test.shape}')

    lp, xp, cp_preds = [], [], []
    for seed in seeds:
        b = lgb.Booster(model_file=str(MODELS_OUT / f'lgbm_s{seed}.txt'))
        lp.append(b.predict(X_test))
        m = xgb.XGBClassifier(); m.load_model(str(MODELS_OUT / f'xgb_s{seed}.json'))
        xp.append(m.predict_proba(X_test)[:, 1])
        m = CatBoostClassifier(); m.load_model(str(MODELS_OUT / f'cat_s{seed}.cbm'))
        cp_preds.append(m.predict_proba(X_test)[:, 1])

    avg_l = np.mean(lp, axis=0)
    avg_x = np.mean(xp, axis=0)
    avg_c = np.mean(cp_preds, axis=0)

    ens = (rankdata(avg_l) * weights['lgbm'] +
           rankdata(avg_x) * weights['xgb'] +
           rankdata(avg_c) * weights['catboost'])
    log(f'Predictions: min={ens.min():.1f}  max={ens.max():.1f}  mean={ens.mean():.1f}')

    # ─── Submit ───
    sample = pl.read_csv(DATA / 'sample_submit.csv')
    submit = pl.DataFrame({'event_id': event_ids, 'predict': ens})
    submit = sample.select('event_id').join(submit, on='event_id', how='left')
    assert len(submit) == len(sample), f"Size mismatch: {len(submit)} vs {len(sample)}"

    n_null = submit['predict'].is_null().sum()
    if n_null > 0:
        submit = submit.with_columns(pl.col('predict').fill_null(submit['predict'].drop_nulls().median()))
        log(f'Filled {n_null} nulls')

    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = SUBMIT_OUT / f'submit_v8_{ts}.csv'
    submit.write_csv(out_path)
    log(f'Saved: {out_path}  ({len(submit):,} rows)')

    # Feature importance
    log('\nTop 30 features:')
    b = lgb.Booster(model_file=str(MODELS_OUT / f'lgbm_s42.txt'))
    imp = pd.DataFrame({'feature': available_feats, 'importance': b.feature_importance()})
    print(imp.sort_values('importance', ascending=False).head(30).to_string(index=False))

    log(f'\nTotal: {(time.time()-t_start)/60:.1f} min')
    log('ALL DONE')


if __name__ == '__main__':
    main()
