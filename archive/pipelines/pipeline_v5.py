"""
Pipeline v5: ПРАВИЛЬНАЯ постановка задачи.

Ключевые изменения:
1. Тренируем на labeled-only (🔴 vs 🟡) — модель учит РЕАЛЬНУЮ границу фрода
2. Также пробуем labeled + green (взвешенный)
3. Новые фичи: pos_cd_1 (5.7x lift!), is_no_screen, is_no_timezone
4. Dedup test, multi-seed, rank blending
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
MODELS_OUT  = ROOT / 'models_v5'
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


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Feature engineering v5."""
    new_cols = []

    # --- Из неиспользованных колонок ---
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

    # --- NEW v5: POS code features (pos_cd=1 = 5.7x lift!) ---
    if 'pos_cd' in df.columns:
        df = df.with_columns([
            (pl.col('pos_cd') == 1).fill_null(False).cast(pl.Int8).alias('is_manual_entry'),
            (pl.col('pos_cd') == 3).fill_null(False).cast(pl.Int8).alias('is_pos_cd_3'),
            pl.col('pos_cd').is_not_null().cast(pl.Int8).alias('has_pos_data'),
        ])

    # --- NEW v5: timezone null = fraud signal ---
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

    # --- Behavioral deviation (v4) ---
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

    return df


FEATURE_COLS = [
    # Original
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
    # v2
    'event_desc',
    'has_browser_lang', 'has_accept_lang', 'screen_w', 'screen_h',
    'has_device_ver', 'has_session',
    'cnt_ratio_1h_24h', 'cnt_ratio_6h_24h', 'cnt_ratio_24h_7d',
    'cnt_ratio_7d_30d', 'cnt_ratio_1h_6h',
    'amt_ratio_1h_24h', 'amt_ratio_24h_7d', 'amt_ratio_7d_30d',
    'amt_cur_ratio_24h', 'amt_cur_ratio_7d', 'amt_cur_ratio_30d',
    'hour_weekday', 'hour_sin', 'hour_cos', 'day_of_month',
    'security_risk_score', 'session_ops_x_log_amt',
    # v4 behavioral
    'is_fast_60s', 'is_fast_300s', 'log_secs_since_last',
    'avg_amt_30d', 'avg_amt_7d',
    'amt_deviation_30d', 'amt_deviation_7d',
    'activity_burst_1h', 'hourly_activity_ratio',
    'voip_x_logamt', 'high_risk_x_logamt',
    'amt_pct_of_30d', 'is_first_in_session', 'log_cnt_30d',
    # v5 NEW
    'is_manual_entry', 'is_pos_cd_3', 'has_pos_data',
    'is_no_screen', 'is_no_timezone',
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
        res = minimize(neg_prauc, w0, method='Nelder-Mead', options={'maxiter':3000})
        if -res.fun > best_s:
            best_s = -res.fun; best_w = np.abs(np.array(res.x))
    best_w = best_w / best_w.sum()
    log(f'Rank weights: {dict(zip(names, best_w.round(4)))}  PR-AUC={best_s:.4f}')
    return dict(zip(names, best_w.tolist())), best_s


def main():
    t_start = time.time()
    print('\n' + '='*60)
    print('PIPELINE V5: Labeled-only + new fraud signals')
    print('='*60)

    # Load ALL train features + add v5 features
    log('Loading data...')
    df_all = pl.read_parquet(FEATURES_IN / 'train_features.parquet')
    df_all = add_features(df_all)

    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    red_eids = set(labels.filter(pl.col('target')==1)['event_id'].to_list())
    yellow_eids = set(labels.filter(pl.col('target')==0)['event_id'].to_list())
    label_eids = red_eids | yellow_eids

    available_feats = [c for c in FEATURE_COLS if c in df_all.columns]
    log(f'Features: {len(available_feats)}')

    val_dt = datetime(2025, 4, 1)

    # ═══════════════════════════════════════════
    # EXPERIMENT: Compare training strategies
    # ═══════════════════════════════════════════
    print('\n' + '='*60)
    print('EXPERIMENT: Training strategy comparison')
    print('='*60)

    # Full val for evaluation (same as before for comparison)
    df_val_full = df_all.filter(pl.col('event_dttm') >= val_dt)
    X_val_full = df_val_full.select(available_feats).to_pandas()
    y_val_full = df_val_full['target'].to_numpy().astype(int)

    # Labeled-only val
    df_val_labeled = df_val_full.filter(pl.col('event_id').is_in(list(label_eids)))
    X_val_lab = df_val_labeled.select(available_feats).to_pandas()
    y_val_lab = df_val_labeled['target'].to_numpy().astype(int)
    log(f'Val labeled: {len(X_val_lab):,} (🔴={y_val_lab.sum():,}  🟡={(y_val_lab==0).sum():,})')

    strategies = {}

    # --- Strategy A: Labeled-only (🔴 vs 🟡) ---
    log('\n--- A: Labeled-only (🔴 vs 🟡) ---')
    df_tr_lab = df_all.filter(
        (pl.col('event_dttm') < val_dt) & (pl.col('event_id').is_in(list(label_eids)))
    )
    X_a = df_tr_lab.select(available_feats).to_pandas()
    y_a = df_tr_lab['target'].to_numpy().astype(int)
    n_pos_a, n_neg_a = y_a.sum(), (y_a==0).sum()
    log(f'Train A: {len(X_a):,} (🔴={n_pos_a:,}  🟡={n_neg_a:,})')

    m_a = lgb.LGBMClassifier(
        objective='binary', metric='average_precision',
        device='gpu', gpu_platform_id=0, gpu_device_id=0,
        scale_pos_weight=n_neg_a / n_pos_a,
        n_estimators=3000, learning_rate=0.03,
        num_leaves=127, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=2.0,
        random_state=42, n_jobs=4, verbose=-1,
    )
    m_a.fit(X_a, y_a, eval_set=[(X_val_lab, y_val_lab)],
            callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(100)])
    p_a_lab = m_a.predict_proba(X_val_lab)[:, 1]
    p_a_full = m_a.predict_proba(X_val_full)[:, 1]
    s_a_lab = average_precision_score(y_val_lab, p_a_lab)
    s_a_full = average_precision_score(y_val_full, p_a_full)
    log(f'A labeled-val: {s_a_lab:.4f}  full-val: {s_a_full:.4f}  (iter={m_a.best_iteration_})')
    strategies['A_labeled_only'] = {'lab': s_a_lab, 'full': s_a_full}

    # --- Strategy B: Labeled + green (current approach, 5:1) ---
    log('\n--- B: Labeled + green (5:1, old approach) ---')
    df_tr_full = df_all.filter(pl.col('event_dttm') < val_dt)
    X_b = df_tr_full.select(available_feats).to_pandas()
    y_b = df_tr_full['target'].to_numpy().astype(int)
    n_pos_b, n_neg_b = y_b.sum(), (y_b==0).sum()
    log(f'Train B: {len(X_b):,} (pos={n_pos_b:,}  neg={n_neg_b:,})')

    m_b = lgb.LGBMClassifier(
        objective='binary', metric='average_precision',
        device='gpu', gpu_platform_id=0, gpu_device_id=0,
        scale_pos_weight=n_neg_b / n_pos_b,
        n_estimators=3000, learning_rate=0.03,
        num_leaves=127, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=2.0,
        random_state=42, n_jobs=4, verbose=-1,
    )
    m_b.fit(X_b, y_b, eval_set=[(X_val_full, y_val_full)],
            callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(100)])
    p_b_lab = m_b.predict_proba(X_val_lab)[:, 1]
    p_b_full = m_b.predict_proba(X_val_full)[:, 1]
    s_b_lab = average_precision_score(y_val_lab, p_b_lab)
    s_b_full = average_precision_score(y_val_full, p_b_full)
    log(f'B labeled-val: {s_b_lab:.4f}  full-val: {s_b_full:.4f}  (iter={m_b.best_iteration_})')
    strategies['B_labeled_green'] = {'lab': s_b_lab, 'full': s_b_full}

    # --- Strategy C: Labeled + small green (1:1 yellow:green) ---
    log('\n--- C: Labeled + small green (🔴 + 🟡 + 🟢 sampled 1:1 with 🟡) ---')
    df_tr_labeled = df_all.filter(
        (pl.col('event_dttm') < val_dt) & (pl.col('event_id').is_in(list(label_eids)))
    )
    df_tr_green = df_all.filter(
        (pl.col('event_dttm') < val_dt) & (~pl.col('event_id').is_in(list(label_eids)))
    )
    n_yellow_tr = df_tr_labeled.filter(pl.col('target')==0).height
    df_tr_green_sample = df_tr_green.sample(n=min(n_yellow_tr, len(df_tr_green)), seed=42)
    df_tr_green_sample = df_tr_green_sample.with_columns(pl.lit(0).cast(pl.Int32).alias('target'))
    df_tr_c = pl.concat([df_tr_labeled, df_tr_green_sample])
    X_c = df_tr_c.select(available_feats).to_pandas()
    y_c = df_tr_c['target'].to_numpy().astype(int)
    n_pos_c, n_neg_c = y_c.sum(), (y_c==0).sum()
    log(f'Train C: {len(X_c):,} (🔴={n_pos_c:,}  🟡+🟢={n_neg_c:,})')

    m_c = lgb.LGBMClassifier(
        objective='binary', metric='average_precision',
        device='gpu', gpu_platform_id=0, gpu_device_id=0,
        scale_pos_weight=n_neg_c / n_pos_c,
        n_estimators=3000, learning_rate=0.03,
        num_leaves=127, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=2.0,
        random_state=42, n_jobs=4, verbose=-1,
    )
    m_c.fit(X_c, y_c, eval_set=[(X_val_lab, y_val_lab)],
            callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(100)])
    p_c_lab = m_c.predict_proba(X_val_lab)[:, 1]
    p_c_full = m_c.predict_proba(X_val_full)[:, 1]
    s_c_lab = average_precision_score(y_val_lab, p_c_lab)
    s_c_full = average_precision_score(y_val_full, p_c_full)
    log(f'C labeled-val: {s_c_lab:.4f}  full-val: {s_c_full:.4f}  (iter={m_c.best_iteration_})')
    strategies['C_labeled_small_green'] = {'lab': s_c_lab, 'full': s_c_full}

    # --- Strategy D: Labeled with sample_weight (🟡 weight 5x, 🟢 weight 1x) ---
    log('\n--- D: Weighted (🟡 5x weight, 🟢 1x weight) ---')
    sample_weights = np.ones(len(X_b))
    for i, eid in enumerate(df_tr_full['event_id'].to_list()):
        if eid in yellow_eids:
            sample_weights[i] = 5.0
        elif eid in red_eids:
            sample_weights[i] = 1.0  # Will be handled by scale_pos_weight

    m_d = lgb.LGBMClassifier(
        objective='binary', metric='average_precision',
        device='gpu', gpu_platform_id=0, gpu_device_id=0,
        scale_pos_weight=n_neg_b / n_pos_b,
        n_estimators=3000, learning_rate=0.03,
        num_leaves=127, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=2.0,
        random_state=42, n_jobs=4, verbose=-1,
    )
    m_d.fit(X_b, y_b, sample_weight=sample_weights,
            eval_set=[(X_val_full, y_val_full)],
            callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(100)])
    p_d_lab = m_d.predict_proba(X_val_lab)[:, 1]
    p_d_full = m_d.predict_proba(X_val_full)[:, 1]
    s_d_lab = average_precision_score(y_val_lab, p_d_lab)
    s_d_full = average_precision_score(y_val_full, p_d_full)
    log(f'D labeled-val: {s_d_lab:.4f}  full-val: {s_d_full:.4f}  (iter={m_d.best_iteration_})')
    strategies['D_weighted'] = {'lab': s_d_lab, 'full': s_d_full}

    # Print comparison
    print('\n' + '='*60)
    print('STRATEGY COMPARISON')
    print('='*60)
    print(f'{"Strategy":30s}  {"Lab-val":>8s}  {"Full-val":>8s}')
    for k, v in strategies.items():
        print(f'{k:30s}  {v["lab"]:8.4f}  {v["full"]:8.4f}')

    # ═══════════════════════════════════════════
    # Pick best strategy and do full training
    # ═══════════════════════════════════════════

    # For LB, labeled-val is more representative (closer to actual fraud vs non-fraud boundary)
    best_key = max(strategies, key=lambda k: strategies[k]['lab'])
    log(f'\nBest strategy by labeled-val: {best_key}')

    # Use best strategy for final models
    # Map strategy to train data
    if best_key == 'A_labeled_only':
        df_final_tr = df_all.filter(pl.col('event_id').is_in(list(label_eids)))
        best_model = m_a
    elif best_key == 'C_labeled_small_green':
        labeled_all = df_all.filter(pl.col('event_id').is_in(list(label_eids)))
        green_all = df_all.filter(~pl.col('event_id').is_in(list(label_eids)))
        n_yellow_all = labeled_all.filter(pl.col('target')==0).height
        green_sample = green_all.sample(n=min(n_yellow_all, len(green_all)), seed=42)
        green_sample = green_sample.with_columns(pl.lit(0).cast(pl.Int32).alias('target'))
        df_final_tr = pl.concat([labeled_all, green_sample])
        best_model = m_c
    elif best_key == 'D_weighted':
        df_final_tr = df_all
        best_model = m_d
    else:  # B
        df_final_tr = df_all
        best_model = m_b

    # ═══════════════════════════════════════════
    # Multi-seed full training with best strategy
    # ═══════════════════════════════════════════
    print('\n' + '='*60)
    print(f'FULL TRAINING: {best_key}')
    print('='*60)

    # Val predictions for ensemble optimization
    all_lgbm_p, all_xgb_p, all_cat_p = [], [], []
    seeds = [42, 123, 777]
    best_iters = {'lgbm': [], 'xgb': [], 'cat': []}

    # Prepare train data
    X_final = df_final_tr.select(available_feats).to_pandas()
    y_final = df_final_tr['target'].to_numpy().astype(int)
    n_pos_f = y_final.sum()
    n_neg_f = (y_final == 0).sum()
    spw = n_neg_f / n_pos_f

    # Prepare sample weights if using weighted strategy
    sw_final = None
    if best_key == 'D_weighted':
        sw_final = np.ones(len(X_final))
        final_eids = df_final_tr['event_id'].to_list()
        for i, eid in enumerate(final_eids):
            if eid in yellow_eids:
                sw_final[i] = 5.0

    # Train split for val predictions
    X_tr_split = df_final_tr.filter(pl.col('event_dttm') < val_dt).select(available_feats).to_pandas()
    y_tr_split = df_final_tr.filter(pl.col('event_dttm') < val_dt)['target'].to_numpy().astype(int)

    sw_tr_split = None
    if sw_final is not None:
        mask_tr = (df_final_tr['event_dttm'] < val_dt).to_numpy()
        sw_tr_split = sw_final[mask_tr]

    # Use labeled val for evaluation
    for si, seed in enumerate(seeds):
        log(f'\n--- Seed {seed} ({si+1}/{len(seeds)}) ---')

        # LightGBM
        lgbm = lgb.LGBMClassifier(
            objective='binary', metric='average_precision',
            device='gpu', gpu_platform_id=0, gpu_device_id=0,
            scale_pos_weight=spw,
            n_estimators=5000, learning_rate=0.03,
            num_leaves=127, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0,
            random_state=seed, n_jobs=4, verbose=-1,
        )
        fit_kw = {'eval_set': [(X_val_lab, y_val_lab)],
                   'callbacks': [lgb.early_stopping(200, verbose=True), lgb.log_evaluation(100)]}
        if sw_tr_split is not None:
            fit_kw['sample_weight'] = sw_tr_split
        lgbm.fit(X_tr_split, y_tr_split, **fit_kw)
        lp = lgbm.predict_proba(X_val_lab)[:, 1]
        ls = average_precision_score(y_val_lab, lp)
        log(f'LGBM[{seed}]: {ls:.4f} (iter={lgbm.best_iteration_})')
        all_lgbm_p.append(lp)
        best_iters['lgbm'].append(lgbm.best_iteration_)

        # XGBoost
        xgbm = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='aucpr',
            tree_method='hist', device='cuda',
            scale_pos_weight=spw,
            n_estimators=5000, learning_rate=0.03,
            max_depth=8, min_child_weight=20,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0,
            random_state=seed, n_jobs=4, verbosity=0,
            early_stopping_rounds=200,
        )
        xgb_fit_kw = {'eval_set': [(X_val_lab, y_val_lab)], 'verbose': 100}
        if sw_tr_split is not None:
            xgb_fit_kw['sample_weight'] = sw_tr_split
        xgbm.fit(X_tr_split, y_tr_split, **xgb_fit_kw)
        xp = xgbm.predict_proba(X_val_lab)[:, 1]
        xs = average_precision_score(y_val_lab, xp)
        log(f'XGB[{seed}]: {xs:.4f} (iter={xgbm.best_iteration})')
        all_xgb_p.append(xp)
        best_iters['xgb'].append(xgbm.best_iteration)

        # CatBoost
        cat = CatBoostClassifier(
            iterations=5000, learning_rate=0.03, depth=8,
            task_type='GPU', devices='0',
            loss_function='Logloss', eval_metric='AUC',
            auto_class_weights='Balanced', l2_leaf_reg=3.0,
            random_seed=seed, verbose=100, early_stopping_rounds=200,
        )
        cat_fit_kw = {'eval_set': (X_val_lab, y_val_lab), 'use_best_model': True}
        if sw_tr_split is not None:
            from catboost import Pool
            cat_fit_kw = {}
            train_pool = Pool(X_tr_split, y_tr_split, weight=sw_tr_split)
            val_pool = Pool(X_val_lab, y_val_lab)
            cat.fit(train_pool, eval_set=val_pool, use_best_model=True)
        else:
            cat.fit(X_tr_split, y_tr_split, **cat_fit_kw)
        cp = cat.predict_proba(X_val_lab)[:, 1]
        cs = average_precision_score(y_val_lab, cp)
        log(f'CAT[{seed}]: {cs:.4f} (iter={cat.best_iteration_})')
        all_cat_p.append(cp)
        best_iters['cat'].append(cat.best_iteration_)

    # Average and optimize ensemble
    avg_l = np.mean(all_lgbm_p, axis=0)
    avg_x = np.mean(all_xgb_p, axis=0)
    avg_c = np.mean(all_cat_p, axis=0)

    log(f'\nAveraged: LGBM={average_precision_score(y_val_lab, avg_l):.4f}'
        f'  XGB={average_precision_score(y_val_lab, avg_x):.4f}'
        f'  CAT={average_precision_score(y_val_lab, avg_c):.4f}')

    weights, ens_score = optimize_rank_weights(
        [avg_l, avg_x, avg_c], y_val_lab, ['lgbm', 'xgb', 'catboost'])

    with open(MODELS_OUT / 'config.json', 'w') as f:
        json.dump({
            'strategy': best_key,
            'weights': weights,
            'ensemble_prauc': ens_score,
            'features': available_feats,
            'strategies_comparison': strategies,
        }, f, indent=2, default=str)

    del X_tr_split, y_tr_split, X_val_lab, y_val_lab, X_val_full, y_val_full
    del df_val_full, df_val_labeled, df_tr_full; gc.collect()

    # ═══════════════════════════════════════════
    # Final models on ALL data
    # ═══════════════════════════════════════════
    log('\nTraining final models on all data...')

    avg_li = int(np.mean(best_iters['lgbm']) * 1.1)
    avg_xi = int(np.mean(best_iters['xgb']) * 1.1)
    avg_ci = int(np.mean(best_iters['cat']) * 1.1)

    for si, seed in enumerate(seeds):
        log(f'Final seed {seed}...')
        fl = lgb.LGBMClassifier(
            objective='binary', metric='average_precision',
            device='gpu', gpu_platform_id=0, gpu_device_id=0,
            scale_pos_weight=spw, n_estimators=avg_li, learning_rate=0.03,
            num_leaves=127, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0,
            random_state=seed, n_jobs=4, verbose=-1)
        if sw_final is not None:
            fl.fit(X_final, y_final, sample_weight=sw_final)
        else:
            fl.fit(X_final, y_final)
        fl.booster_.save_model(str(MODELS_OUT / f'lgbm_s{seed}.txt'))

        fx = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='aucpr',
            tree_method='hist', device='cuda',
            scale_pos_weight=spw, n_estimators=avg_xi, learning_rate=0.03,
            max_depth=8, min_child_weight=20,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0,
            random_state=seed, n_jobs=4, verbosity=0)
        if sw_final is not None:
            fx.fit(X_final, y_final, sample_weight=sw_final)
        else:
            fx.fit(X_final, y_final)
        fx.save_model(str(MODELS_OUT / f'xgb_s{seed}.json'))

        fc = CatBoostClassifier(
            iterations=avg_ci, learning_rate=0.03, depth=8,
            task_type='GPU', devices='0',
            loss_function='Logloss', auto_class_weights='Balanced',
            l2_leaf_reg=3.0, random_seed=seed, verbose=0)
        if sw_final is not None:
            from catboost import Pool
            fc.fit(Pool(X_final, y_final, weight=sw_final))
        else:
            fc.fit(X_final, y_final)
        fc.save_model(str(MODELS_OUT / f'cat_s{seed}.cbm'))

    del X_final, y_final; gc.collect()

    # ═══════════════════════════════════════════
    # Predict test (with dedup)
    # ═══════════════════════════════════════════
    log('\nPredicting test...')
    df_test = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    n_before = len(df_test)
    df_test = df_test.unique(subset=['event_id'], keep='first')
    log(f'Dedup: {n_before} -> {len(df_test)} (-{n_before-len(df_test)})')

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

    # Submit
    sample = pl.read_csv(DATA / 'sample_submit.csv')
    submit = pl.DataFrame({'event_id': event_ids, 'predict': ens})
    submit = sample.select('event_id').join(submit, on='event_id', how='left')
    assert len(submit) == len(sample)

    n_null = submit['predict'].is_null().sum()
    if n_null > 0:
        submit = submit.with_columns(pl.col('predict').fill_null(submit['predict'].drop_nulls().median()))
        log(f'Filled {n_null} nulls')

    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = SUBMIT_OUT / f'submit_v5_{ts}.csv'
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
