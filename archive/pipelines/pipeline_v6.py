"""
Pipeline v6: V5 features (84) + 6 new PDF features (90 total) + labeled+green strategy.

Key idea: v5 features are better but labeled-only strategy failed on LB (0.068 vs 0.094).
Combining best features with proven strategy should beat v3f's 0.094.

Changes vs v3f (LB=0.094):
  - 90 features (was 41): +pos_cd signals, +is_no_screen/timezone, +behavioral, +PDF research
  - Rank-based blending (more robust for PR-AUC)
  - 5 seeds for averaging (was 3)
  - num_leaves=255 (v3f value, better for large dataset)

Changes vs v5 (LB=0.068):
  - Strategy: labeled+green (NOT labeled-only)
  - 6 new features from PDF fraud research

New v6 features from PDF "Маркеры мошенничества в банковских транзакциях":
  1. is_high_risk_mcc — MCC 6051(крипто),7995(гемблинг),4829(переводы),5094(ювелирка),5967(adult)
  2. voip_and_rdp — VoIP + RDP одновременно (социальная инженерия)
  3. n_security_flags — простой count активных security флагов
  4. mcc_novelty_x_amt — новый MCC * сумма (красный флаг)
  5. compromised_x_amt — root устройство * сумма
  6. manual_entry_x_amt — ручной ввод карты * сумма (CNP fraud)
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
import gc, json, time

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_OUT  = ROOT / 'models_v6'
SUBMIT_OUT  = ROOT / 'submissions'

for d in [MODELS_OUT, SUBMIT_OUT]:
    d.mkdir(exist_ok=True)

def _ram_gb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1048576
    except Exception:
        return 0.0
    return 0.0

def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")} RAM:{_ram_gb():.1f}GB] {msg}', flush=True)


# ─── Feature Engineering (v5 features — superset of v3f) ───

def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Feature engineering v6 (90 features: v5 base + 6 from PDF research)."""
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

    # --- POS code features (pos_cd=1 = 5.7x lift for fraud!) ---
    if 'pos_cd' in df.columns:
        df = df.with_columns([
            (pl.col('pos_cd') == 1).fill_null(False).cast(pl.Int8).alias('is_manual_entry'),
            (pl.col('pos_cd') == 3).fill_null(False).cast(pl.Int8).alias('is_pos_cd_3'),
            pl.col('pos_cd').is_not_null().cast(pl.Int8).alias('has_pos_data'),
        ])

    # --- timezone null = fraud signal ---
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

    # --- Interactions ---
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

    # ─── v6 NEW: 6 features from PDF fraud research ───

    # 1. High-risk MCC codes (crypto, gambling, money transfers, jewelry, adult)
    if 'mcc_code' in df.columns:
        high_risk_mccs = [6051, 7995, 4829, 5094, 5967]
        df = df.with_columns([
            pl.col('mcc_code').is_in(high_risk_mccs).fill_null(False).cast(pl.Int8).alias('is_high_risk_mcc'),
        ])

    # 2. VoIP + RDP simultaneous — classic social engineering pattern
    if 'phone_voip_call_state' in df.columns and 'web_rdp_connection' in df.columns:
        df = df.with_columns([
            (pl.col('phone_voip_call_state') & pl.col('web_rdp_connection')).cast(pl.Int8).alias('voip_and_rdp'),
        ])

    # 3. Count of active security flags (simpler than weighted risk_score)
    if all(c in df.columns for c in ['compromised','web_rdp_connection','phone_voip_call_state','developer_tools']):
        df = df.with_columns([
            (pl.col('compromised').cast(pl.Int8) + pl.col('web_rdp_connection').cast(pl.Int8) +
             pl.col('phone_voip_call_state').cast(pl.Int8) + pl.col('developer_tools').cast(pl.Int8)
            ).alias('n_security_flags'),
        ])

    # 4. New MCC + high amount — red flag per PDF
    if 'is_new_mcc_code' in df.columns and 'log_amount' in df.columns:
        df = df.with_columns([
            (pl.col('is_new_mcc_code') * pl.col('log_amount')).alias('mcc_novelty_x_amt'),
        ])

    # 5. Root/compromised device + high amount
    if 'compromised' in df.columns and 'log_amount' in df.columns:
        df = df.with_columns([
            (pl.col('compromised') * pl.col('log_amount')).alias('compromised_x_amt'),
        ])

    # 6. Manual POS entry + high amount — card-not-present fraud
    if 'is_manual_entry' in df.columns and 'log_amount' in df.columns:
        df = df.with_columns([
            (pl.col('is_manual_entry') * pl.col('log_amount')).alias('manual_entry_x_amt'),
        ])

    return df


FEATURE_COLS = [
    # Original (from run_features.py)
    'hour', 'weekday', 'month', 'is_night',
    'log_amount', 'operaton_amt', 'is_null_amount',
    'phone_voip_call_state', 'web_rdp_connection', 'compromised',
    'developer_tools', 'security_flags_sum',
    'event_type_nm', 'is_high_risk_type',
    'mcc_code', 'is_null_mcc',
    'channel_indicator_type', 'channel_indicator_sub_type', 'currency_iso_cd',
    'battery', 'operating_system_type', 'pos_cd',
    'cnt_1h', 'cnt_6h', 'cnt_24h', 'cnt_7d', 'cnt_30d',
    'amt_sum_1h', 'amt_sum_6h', 'amt_sum_24h', 'amt_sum_7d', 'amt_sum_30d',
    'secs_since_last', 'voip_cnt_24h',
    'is_new_mcc_code', 'is_new_channel_indicator_type', 'is_new_currency_iso_cd',
    'cum_unique_mcc_approx',
    'session_ops_before', 'session_amt_before',
    'timezone',
    # v2 features
    'event_desc',
    'has_browser_lang', 'has_accept_lang', 'screen_w', 'screen_h',
    'has_device_ver', 'has_session',
    'cnt_ratio_1h_24h', 'cnt_ratio_6h_24h', 'cnt_ratio_24h_7d',
    'cnt_ratio_7d_30d', 'cnt_ratio_1h_6h',
    'amt_ratio_1h_24h', 'amt_ratio_24h_7d', 'amt_ratio_7d_30d',
    'amt_cur_ratio_24h', 'amt_cur_ratio_7d', 'amt_cur_ratio_30d',
    'hour_weekday', 'hour_sin', 'hour_cos', 'day_of_month',
    'security_risk_score', 'session_ops_x_log_amt',
    # v5 NEW features
    'is_fast_60s', 'is_fast_300s', 'log_secs_since_last',
    'avg_amt_30d', 'avg_amt_7d',
    'amt_deviation_30d', 'amt_deviation_7d',
    'activity_burst_1h', 'hourly_activity_ratio',
    'voip_x_logamt', 'high_risk_x_logamt',
    'amt_pct_of_30d', 'is_first_in_session', 'log_cnt_30d',
    'is_manual_entry', 'is_pos_cd_3', 'has_pos_data',
    'is_no_screen', 'is_no_timezone',
    # v6 NEW: from PDF research on fraud markers
    'is_high_risk_mcc',         # MCC 6051/7995/4829/5094/5967
    'voip_and_rdp',             # VoIP + RDP simultaneous
    'n_security_flags',         # count of active security flags
    'mcc_novelty_x_amt',        # new MCC * amount
    'compromised_x_amt',        # root device * amount
    'manual_entry_x_amt',       # manual POS entry * amount
]


def optimize_rank_weights(preds_list, y_true, names):
    """Optimize rank-based blending weights for PR-AUC."""
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
    print('PIPELINE V6: v5 features + labeled+green strategy')
    print('='*60)

    # ─── Load data ───
    log('Loading train features...')
    df_all = pl.read_parquet(FEATURES_IN / 'train_features.parquet')
    df_all = add_features(df_all)

    available_feats = [c for c in FEATURE_COLS if c in df_all.columns]
    log(f'Features: {len(available_feats)} / {len(FEATURE_COLS)}')

    # ─── Train/val split ───
    val_dt = datetime(2025, 4, 1)
    df_tr = df_all.filter(pl.col('event_dttm') < val_dt)
    df_val = df_all.filter(pl.col('event_dttm') >= val_dt)

    X_train = df_tr.select(available_feats).to_pandas()
    y_train = df_tr['target'].to_numpy().astype(int)
    X_val = df_val.select(available_feats).to_pandas()
    y_val = df_val['target'].to_numpy().astype(int)

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    spw = n_neg / n_pos
    log(f'Train: {len(X_train):,} (pos={n_pos:,}, neg={n_neg:,}, spw={spw:.1f})')
    log(f'Val: {len(X_val):,} (fraud={y_val.sum():,})')

    del df_tr, df_val; gc.collect()

    # ─── Multi-seed training ───
    seeds = [42, 123, 777, 2024, 31337]
    all_lgbm_p, all_xgb_p, all_cat_p = [], [], []
    best_iters = {'l': [], 'x': [], 'c': []}

    for si, seed in enumerate(seeds):
        log(f'\n--- Seed {seed} ({si+1}/{len(seeds)}) ---')

        # LightGBM (num_leaves=255 from v3f)
        m_l = lgb.LGBMClassifier(
            objective='binary', metric='average_precision',
            device='gpu', gpu_platform_id=0, gpu_device_id=0,
            scale_pos_weight=spw,
            n_estimators=5000, learning_rate=0.03,
            num_leaves=255, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0, max_bin=255,
            random_state=seed, n_jobs=4, verbose=-1,
        )
        m_l.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(100)])
        lp = m_l.predict_proba(X_val)[:, 1]
        ls = average_precision_score(y_val, lp)
        log(f'LGBM[{seed}]: {ls:.4f} (iter={m_l.best_iteration_})')
        all_lgbm_p.append(lp)
        best_iters['l'].append(m_l.best_iteration_)

        # XGBoost
        m_x = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='aucpr',
            tree_method='hist', device='cuda',
            scale_pos_weight=spw,
            n_estimators=5000, learning_rate=0.03,
            max_depth=8, min_child_weight=30,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0, max_bin=255,
            random_state=seed, n_jobs=4, verbosity=0,
            early_stopping_rounds=200,
        )
        m_x.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
        xp = m_x.predict_proba(X_val)[:, 1]
        xs = average_precision_score(y_val, xp)
        log(f'XGB[{seed}]: {xs:.4f} (iter={m_x.best_iteration})')
        all_xgb_p.append(xp)
        best_iters['x'].append(m_x.best_iteration)

        # CatBoost
        m_c = CatBoostClassifier(
            iterations=5000, learning_rate=0.03, depth=8,
            task_type='GPU', devices='0',
            loss_function='Logloss', eval_metric='AUC',
            auto_class_weights='Balanced', l2_leaf_reg=3.0,
            random_seed=seed, verbose=100, early_stopping_rounds=200,
        )
        m_c.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        cp = m_c.predict_proba(X_val)[:, 1]
        cs = average_precision_score(y_val, cp)
        log(f'CAT[{seed}]: {cs:.4f} (iter={m_c.best_iteration_})')
        all_cat_p.append(cp)
        best_iters['c'].append(m_c.best_iteration_)

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
        'strategy': 'labeled_green_v5features',
        'weights': weights,
        'ensemble_prauc': ens_score,
        'best_iters': best_iters,
        'n_features': len(available_feats),
        'n_seeds': len(seeds),
    }
    with open(MODELS_OUT / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    del X_train, y_train, X_val, y_val; gc.collect()

    # ─── Final models on ALL data ───
    log('\nTraining final models on ALL data...')

    X_full = df_all.select(available_feats).to_pandas()
    y_full = df_all['target'].to_numpy().astype(int)
    del df_all; gc.collect()

    avg_li = int(np.mean(best_iters['l']) * 1.1)
    avg_xi = int(np.mean(best_iters['x']) * 1.1)
    avg_ci = int(np.mean(best_iters['c']) * 1.1)
    log(f'Final iters: LGBM={avg_li}, XGB={avg_xi}, CAT={avg_ci}')

    for si, seed in enumerate(seeds):
        log(f'Final seed {seed} ({si+1}/{len(seeds)})...')

        fl = lgb.LGBMClassifier(
            objective='binary', metric='average_precision',
            device='gpu', gpu_platform_id=0, gpu_device_id=0,
            scale_pos_weight=spw, n_estimators=avg_li, learning_rate=0.03,
            num_leaves=255, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0, max_bin=255,
            random_state=seed, n_jobs=4, verbose=-1)
        fl.fit(X_full, y_full)
        fl.booster_.save_model(str(MODELS_OUT / f'lgbm_s{seed}.txt'))

        fx = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='aucpr',
            tree_method='hist', device='cuda',
            scale_pos_weight=spw, n_estimators=avg_xi, learning_rate=0.03,
            max_depth=8, min_child_weight=30,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0, max_bin=255,
            random_state=seed, n_jobs=4, verbosity=0)
        fx.fit(X_full, y_full)
        fx.save_model(str(MODELS_OUT / f'xgb_s{seed}.json'))

        fc = CatBoostClassifier(
            iterations=avg_ci, learning_rate=0.03, depth=8,
            task_type='GPU', devices='0',
            loss_function='Logloss', auto_class_weights='Balanced',
            l2_leaf_reg=3.0, random_seed=seed, verbose=0)
        fc.fit(X_full, y_full)
        fc.save_model(str(MODELS_OUT / f'cat_s{seed}.cbm'))

    del X_full, y_full; gc.collect()

    # ─── Predict test ───
    log('\nPredicting test...')
    df_test = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    n_before = len(df_test)
    df_test = df_test.unique(subset=['event_id'], keep='first')
    log(f'Dedup: {n_before} -> {len(df_test)} (-{n_before - len(df_test)})')

    df_test = add_features(df_test)
    X_test = df_test.select(available_feats).to_pandas()
    event_ids = df_test['event_id'].to_numpy()
    log(f'X_test: {X_test.shape}')

    lp, xp, cp = [], [], []
    for seed in seeds:
        b = lgb.Booster(model_file=str(MODELS_OUT / f'lgbm_s{seed}.txt'))
        lp.append(b.predict(X_test))
        m = xgb.XGBClassifier(); m.load_model(str(MODELS_OUT / f'xgb_s{seed}.json'))
        xp.append(m.predict_proba(X_test)[:, 1])
        m = CatBoostClassifier(); m.load_model(str(MODELS_OUT / f'cat_s{seed}.cbm'))
        cp.append(m.predict_proba(X_test)[:, 1])

    avg_l = np.mean(lp, axis=0)
    avg_x = np.mean(xp, axis=0)
    avg_c = np.mean(cp, axis=0)

    # Rank-based blending
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
    out_path = SUBMIT_OUT / f'submit_v6_{ts}.csv'
    submit.write_csv(out_path)
    log(f'Saved: {out_path}  ({len(submit):,} rows)')

    # Feature importance
    log('\nTop 20 features:')
    b = lgb.Booster(model_file=str(MODELS_OUT / f'lgbm_s42.txt'))
    imp = pd.DataFrame({'feature': available_feats, 'importance': b.feature_importance()})
    print(imp.sort_values('importance', ascending=False).head(20).to_string(index=False))

    log(f'\nTotal: {(time.time()-t_start)/60:.1f} min')
    log('ALL DONE')


if __name__ == '__main__':
    main()
