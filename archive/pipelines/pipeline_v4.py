"""
Pipeline v4: Customer behavior deviation features + better fraud signals.

Ключевые улучшения на основе EDA:
1. Фичи отклонения от нормы клиента (amount deviation, activity burst)
2. Быстрые транзакции (is_fast_60s, is_fast_300s)
3. Interaction features (voip*amount, high_risk*channel)
4. Эксперимент: labeled-only vs labeled+unlabeled
5. Rank-based blending для ансамбля
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
MODELS_OUT  = ROOT / 'models_v4'
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


def add_v4_features(df: pl.DataFrame) -> pl.DataFrame:
    """All features including v2 + new v4 behavioral deviation features."""

    # ─── v2 features (proven) ───
    new_cols = []
    if 'browser_language' in df.columns:
        new_cols.append(pl.col('browser_language').is_not_null().cast(pl.Int8).alias('has_browser_lang'))
    if 'accept_language' in df.columns:
        new_cols.append(pl.col('accept_language').is_not_null().cast(pl.Int8).alias('has_accept_lang'))
    if 'screen_size' in df.columns:
        new_cols.extend([
            pl.col('screen_size').str.split('x').list.first().cast(pl.Int32, strict=False).alias('screen_w'),
            pl.col('screen_size').str.split('x').list.last().cast(pl.Int32, strict=False).alias('screen_h'),
        ])
    if 'device_system_version' in df.columns:
        new_cols.append(pl.col('device_system_version').is_not_null().cast(pl.Int8).alias('has_device_ver'))
    if 'session_id' in df.columns:
        new_cols.append(pl.col('session_id').is_not_null().cast(pl.Int8).alias('has_session'))
    if new_cols:
        df = df.with_columns(new_cols)

    # Velocity ratios
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

    # Time features
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

    if all(c in df.columns for c in ['compromised','web_rdp_connection','phone_voip_call_state','developer_tools']):
        df = df.with_columns([
            (pl.col('compromised') * 3 + pl.col('web_rdp_connection') * 2 +
             pl.col('phone_voip_call_state') * 2 + pl.col('developer_tools')).alias('security_risk_score'),
        ])
    if 'session_ops_before' in df.columns and 'log_amount' in df.columns:
        df = df.with_columns([
            (pl.col('session_ops_before') * pl.col('log_amount')).alias('session_ops_x_log_amt'),
        ])

    # ─── NEW v4 features: Customer behavior deviation ───

    v4_cols = []

    # 1. Fast transaction flags (EDA: fraud median gap = 148s, legit = 343s)
    if 'secs_since_last' in df.columns:
        v4_cols.extend([
            (pl.col('secs_since_last') < 60).cast(pl.Int8).alias('is_fast_60s'),
            (pl.col('secs_since_last') < 120).cast(pl.Int8).alias('is_fast_120s'),
            (pl.col('secs_since_last') < 300).cast(pl.Int8).alias('is_fast_300s'),
            pl.col('secs_since_last').log1p().alias('log_secs_since_last'),
        ])

    # 2. Average transaction amount from rolling windows → deviation
    if 'amt_sum_30d' in df.columns and 'cnt_30d' in df.columns:
        v4_cols.append(
            (pl.col('amt_sum_30d') / (pl.col('cnt_30d') + 1)).alias('avg_amt_30d')
        )
    if 'amt_sum_7d' in df.columns and 'cnt_7d' in df.columns:
        v4_cols.append(
            (pl.col('amt_sum_7d') / (pl.col('cnt_7d') + 1)).alias('avg_amt_7d')
        )

    if v4_cols:
        df = df.with_columns(v4_cols)

    # 3. Amount deviation from customer's rolling average
    dev_cols = []
    if 'avg_amt_30d' in df.columns and 'operaton_amt' in df.columns:
        dev_cols.extend([
            (pl.col('operaton_amt') / (pl.col('avg_amt_30d') + 1)).alias('amt_deviation_30d'),
            (pl.col('operaton_amt') - pl.col('avg_amt_30d')).alias('amt_diff_30d'),
        ])
    if 'avg_amt_7d' in df.columns and 'operaton_amt' in df.columns:
        dev_cols.append(
            (pl.col('operaton_amt') / (pl.col('avg_amt_7d') + 1)).alias('amt_deviation_7d'),
        )
    if dev_cols:
        df = df.with_columns(dev_cols)

    # 4. Activity burst detection
    burst_cols = []
    if 'cnt_1h' in df.columns:
        burst_cols.extend([
            (pl.col('cnt_1h') > 3).cast(pl.Int8).alias('activity_burst_1h'),
            (pl.col('cnt_1h') > 5).cast(pl.Int8).alias('activity_burst_1h_5'),
        ])
    if 'cnt_1h' in df.columns and 'cnt_24h' in df.columns:
        burst_cols.append(
            (pl.col('cnt_1h') / (pl.col('cnt_24h') / 24 + 0.01)).alias('hourly_activity_ratio')
        )
    if 'cnt_6h' in df.columns:
        burst_cols.append(
            (pl.col('cnt_6h') > 10).cast(pl.Int8).alias('activity_burst_6h'),
        )
    if burst_cols:
        df = df.with_columns(burst_cols)

    # 5. Interaction features (combine strongest signals)
    inter_cols = []
    if 'phone_voip_call_state' in df.columns and 'log_amount' in df.columns:
        inter_cols.append(
            (pl.col('phone_voip_call_state') * pl.col('log_amount')).alias('voip_x_logamt')
        )
    if 'phone_voip_call_state' in df.columns and 'is_high_risk_type' in df.columns:
        inter_cols.append(
            (pl.col('phone_voip_call_state') * pl.col('is_high_risk_type')).cast(pl.Int8).alias('voip_x_high_risk')
        )
    if 'is_high_risk_type' in df.columns and 'log_amount' in df.columns:
        inter_cols.append(
            (pl.col('is_high_risk_type') * pl.col('log_amount')).alias('high_risk_x_logamt')
        )
    if 'cnt_1h' in df.columns and 'operaton_amt' in df.columns:
        inter_cols.append(
            (pl.col('cnt_1h') * pl.col('log_amount')).alias('cnt1h_x_logamt')
        )
    if inter_cols:
        df = df.with_columns(inter_cols)

    # 6. Amount as fraction of 30d total (how big is this single txn)
    if 'operaton_amt' in df.columns and 'amt_sum_30d' in df.columns:
        df = df.with_columns([
            (pl.col('operaton_amt') / (pl.col('amt_sum_30d') + 1)).alias('amt_pct_of_30d'),
        ])

    # 7. Session features
    if 'session_ops_before' in df.columns:
        df = df.with_columns([
            (pl.col('session_ops_before') == 0).cast(pl.Int8).alias('is_first_in_session'),
        ])

    # 8. Log-transformed counts for better splits
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
    # v2 features
    'event_desc',
    'has_browser_lang', 'has_accept_lang',
    'screen_w', 'screen_h',
    'has_device_ver', 'has_session',
    'cnt_ratio_1h_24h', 'cnt_ratio_6h_24h', 'cnt_ratio_24h_7d',
    'cnt_ratio_7d_30d', 'cnt_ratio_1h_6h',
    'amt_ratio_1h_24h', 'amt_ratio_24h_7d', 'amt_ratio_7d_30d',
    'amt_cur_ratio_24h', 'amt_cur_ratio_7d', 'amt_cur_ratio_30d',
    'hour_weekday', 'hour_sin', 'hour_cos', 'day_of_month',
    'security_risk_score',
    'session_ops_x_log_amt',
    # v4 NEW features
    'is_fast_60s', 'is_fast_120s', 'is_fast_300s', 'log_secs_since_last',
    'avg_amt_30d', 'avg_amt_7d',
    'amt_deviation_30d', 'amt_diff_30d', 'amt_deviation_7d',
    'activity_burst_1h', 'activity_burst_1h_5', 'hourly_activity_ratio',
    'activity_burst_6h',
    'voip_x_logamt', 'voip_x_high_risk', 'high_risk_x_logamt', 'cnt1h_x_logamt',
    'amt_pct_of_30d',
    'is_first_in_session',
    'log_cnt_30d',
]


def optimize_weights_rank(preds_list, y_true, names):
    """Optimize ensemble using rank-averaged predictions."""
    n = len(preds_list)

    # Convert to ranks
    ranks_list = [rankdata(p) for p in preds_list]

    def neg_prauc(w):
        w = np.abs(w)
        w = w / w.sum()
        ens = sum(w[i] * ranks_list[i] for i in range(n))
        return -average_precision_score(y_true, ens)

    best_score = -1
    best_w = None
    for seed in range(50):
        rng = np.random.RandomState(seed)
        w0 = rng.dirichlet(np.ones(n))
        res = minimize(neg_prauc, w0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-8})
        if -res.fun > best_score:
            best_score = -res.fun
            best_w = np.abs(np.array(res.x))

    best_w = best_w / best_w.sum()
    log(f'Rank-optimized weights: {dict(zip(names, best_w.round(4)))}')
    log(f'Rank-optimized ensemble PR-AUC: {best_score:.4f}')
    return dict(zip(names, best_w.tolist()))


def main():
    t_start = time.time()

    print('\n' + '='*60)
    print('PIPELINE V4: Behavioral deviation features + rank blend')
    print('='*60)

    # ═══════════════════════════════════════════
    # Load data
    # ═══════════════════════════════════════════
    log('Loading train features...')
    df_dataset = pl.read_parquet(FEATURES_IN / 'train_features.parquet')
    df_dataset = add_v4_features(df_dataset)

    available_feats = [c for c in FEATURE_COLS if c in df_dataset.columns]
    log(f'Features: {len(available_feats)}')

    val_dt = datetime(2025, 4, 1)
    df_tr = df_dataset.filter(pl.col('event_dttm') < val_dt)
    df_val = df_dataset.filter(pl.col('event_dttm') >= val_dt)

    log(f'Train: {len(df_tr):,}  Val: {len(df_val):,}')
    log(f'Train fraud: {df_tr.filter(pl.col("target")==1).height:,}')
    log(f'Val fraud: {df_val.filter(pl.col("target")==1).height:,}')

    def to_xy(d):
        X = d.select(available_feats).to_pandas()
        y = d['target'].to_numpy().astype(int)
        return X, y

    X_train, y_train = to_xy(df_tr)
    X_val, y_val = to_xy(df_val)

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos = n_neg / n_pos
    log(f'scale_pos_weight={scale_pos:.2f}  pos={n_pos:,}  neg={n_neg:,}')

    # ═══════════════════════════════════════════
    # Experiment: Labeled-only vs Labeled+Unlabeled
    # ═══════════════════════════════════════════
    print('\n--- Experiment: Labeled-only vs Full ---')

    # Labeled-only
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    label_eids = set(labels['event_id'].to_list())

    df_tr_labeled = df_tr.filter(pl.col('event_id').is_in(list(label_eids)))
    df_val_labeled = df_val.filter(pl.col('event_id').is_in(list(label_eids)))

    X_tr_lab = df_tr_labeled.select(available_feats).to_pandas()
    y_tr_lab = df_tr_labeled['target'].to_numpy().astype(int)
    X_val_lab = df_val_labeled.select(available_feats).to_pandas()
    y_val_lab = df_val_labeled['target'].to_numpy().astype(int)

    n_pos_lab = (y_tr_lab == 1).sum()
    n_neg_lab = (y_tr_lab == 0).sum()
    log(f'Labeled-only train: {len(X_tr_lab):,} (pos={n_pos_lab:,}, neg={n_neg_lab:,})')
    log(f'Labeled-only val:   {len(X_val_lab):,}')

    if len(X_val_lab) > 0 and n_pos_lab > 0 and n_neg_lab > 0:
        scale_lab = n_neg_lab / n_pos_lab
        m_lab = lgb.LGBMClassifier(
            objective='binary', metric='average_precision',
            device='gpu', gpu_platform_id=0, gpu_device_id=0,
            scale_pos_weight=scale_lab,
            n_estimators=3000, learning_rate=0.03,
            num_leaves=255, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0,
            random_state=42, n_jobs=4, verbose=-1,
        )
        m_lab.fit(X_tr_lab, y_tr_lab, eval_set=[(X_val_lab, y_val_lab)],
                  callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        p_lab = m_lab.predict_proba(X_val_lab)[:, 1]
        score_lab = average_precision_score(y_val_lab, p_lab)
        log(f'Labeled-only LGBM PR-AUC (labeled val): {score_lab:.4f} (iter={m_lab.best_iteration_})')

        # Also evaluate on FULL val set
        p_full_from_lab = m_lab.predict_proba(X_val)[:, 1]
        score_lab_full = average_precision_score(y_val, p_full_from_lab)
        log(f'Labeled-only LGBM PR-AUC (full val): {score_lab_full:.4f}')
    else:
        log(f'Not enough labeled data in val split')
        score_lab_full = 0

    # Full data model
    m_full = lgb.LGBMClassifier(
        objective='binary', metric='average_precision',
        device='gpu', gpu_platform_id=0, gpu_device_id=0,
        scale_pos_weight=scale_pos,
        n_estimators=3000, learning_rate=0.03,
        num_leaves=255, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=2.0,
        random_state=42, n_jobs=4, verbose=-1,
    )
    m_full.fit(X_train, y_train, eval_set=[(X_val, y_val)],
               callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
    p_full = m_full.predict_proba(X_val)[:, 1]
    score_full = average_precision_score(y_val, p_full)
    log(f'Full-data LGBM PR-AUC (full val): {score_full:.4f} (iter={m_full.best_iteration_})')

    log(f'\n>>> Labeled-only: {score_lab_full:.4f}  Full: {score_full:.4f}')

    del df_tr_labeled, df_val_labeled, X_tr_lab, y_tr_lab, X_val_lab, y_val_lab; gc.collect()

    # ═══════════════════════════════════════════
    # Main training: multi-seed 3-model ensemble
    # ═══════════════════════════════════════════
    print('\n' + '='*60)
    print('Main training: 3 models x 3 seeds')
    print('='*60)

    all_lgbm_preds = []
    all_xgb_preds = []
    all_cat_preds = []
    seeds = [42, 123, 777]
    best_iters = {'lgbm': [], 'xgb': [], 'cat': []}

    for si, seed in enumerate(seeds):
        log(f'\n--- Seed {seed} ({si+1}/{len(seeds)}) ---')

        # LightGBM
        lgbm = lgb.LGBMClassifier(
            objective='binary', metric='average_precision',
            device='gpu', gpu_platform_id=0, gpu_device_id=0,
            scale_pos_weight=scale_pos,
            n_estimators=5000, learning_rate=0.03,
            num_leaves=255, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0, max_bin=255,
            random_state=seed, n_jobs=4, verbose=-1,
        )
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(100)])
        lgbm_p = lgbm.predict_proba(X_val)[:, 1]
        lgbm_s = average_precision_score(y_val, lgbm_p)
        log(f'LGBM[{seed}]: {lgbm_s:.4f} (iter={lgbm.best_iteration_})')
        all_lgbm_preds.append(lgbm_p)
        best_iters['lgbm'].append(lgbm.best_iteration_)

        # XGBoost
        xgbm = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='aucpr',
            tree_method='hist', device='cuda',
            scale_pos_weight=scale_pos,
            n_estimators=5000, learning_rate=0.03,
            max_depth=8, min_child_weight=30,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0, max_bin=255,
            random_state=seed, n_jobs=4, verbosity=0,
            early_stopping_rounds=200,
        )
        xgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
        xgb_p = xgbm.predict_proba(X_val)[:, 1]
        xgb_s = average_precision_score(y_val, xgb_p)
        log(f'XGB[{seed}]: {xgb_s:.4f} (iter={xgbm.best_iteration})')
        all_xgb_preds.append(xgb_p)
        best_iters['xgb'].append(xgbm.best_iteration)

        # CatBoost
        cat = CatBoostClassifier(
            iterations=5000, learning_rate=0.03, depth=8,
            task_type='GPU', devices='0',
            loss_function='Logloss', eval_metric='AUC',
            auto_class_weights='Balanced', l2_leaf_reg=3.0,
            random_seed=seed, verbose=100, early_stopping_rounds=200,
        )
        cat.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        cat_p = cat.predict_proba(X_val)[:, 1]
        cat_s = average_precision_score(y_val, cat_p)
        log(f'CAT[{seed}]: {cat_s:.4f} (iter={cat.best_iteration_})')
        all_cat_preds.append(cat_p)
        best_iters['cat'].append(cat.best_iteration_)

    # Average predictions per model
    avg_lgbm = np.mean(all_lgbm_preds, axis=0)
    avg_xgb = np.mean(all_xgb_preds, axis=0)
    avg_cat = np.mean(all_cat_preds, axis=0)

    log(f'\nAveraged: LGBM={average_precision_score(y_val, avg_lgbm):.4f}'
        f'  XGB={average_precision_score(y_val, avg_xgb):.4f}'
        f'  CAT={average_precision_score(y_val, avg_cat):.4f}')

    # Optimize using RANK blending (better for PR-AUC)
    log('Optimizing rank-based ensemble...')
    weights = optimize_weights_rank(
        [avg_lgbm, avg_xgb, avg_cat], y_val,
        ['lgbm', 'xgb', 'catboost']
    )

    # Also try probability-based
    def neg_prauc_prob(w):
        w = np.abs(w); w = w / w.sum()
        return -average_precision_score(y_val, w[0]*avg_lgbm + w[1]*avg_xgb + w[2]*avg_cat)

    best_prob_score = -1
    best_prob_w = None
    for seed in range(50):
        rng = np.random.RandomState(seed)
        w0 = rng.dirichlet(np.ones(3))
        res = minimize(neg_prauc_prob, w0, method='Nelder-Mead')
        if -res.fun > best_prob_score:
            best_prob_score = -res.fun
            best_prob_w = np.abs(np.array(res.x))
    best_prob_w = best_prob_w / best_prob_w.sum()
    log(f'Prob-based weights: lgbm={best_prob_w[0]:.4f} xgb={best_prob_w[1]:.4f} cat={best_prob_w[2]:.4f}')
    log(f'Prob-based ensemble: {best_prob_score:.4f}')

    # Use whichever is better
    rank_ens = average_precision_score(y_val,
        rankdata(avg_lgbm) * weights['lgbm'] +
        rankdata(avg_xgb) * weights['xgb'] +
        rankdata(avg_cat) * weights['catboost'])
    log(f'Rank ensemble: {rank_ens:.4f}  Prob ensemble: {best_prob_score:.4f}')
    use_rank = rank_ens > best_prob_score
    log(f'Using: {"rank" if use_rank else "prob"} blending')

    prob_weights = {'lgbm': best_prob_w[0], 'xgb': best_prob_w[1], 'catboost': best_prob_w[2]}
    with open(MODELS_OUT / 'weights.json', 'w') as f:
        json.dump({'rank_weights': weights, 'prob_weights': prob_weights,
                   'use_rank': use_rank}, f, indent=2)

    del X_train, y_train, X_val, y_val, df_tr, df_val; gc.collect()

    # ═══════════════════════════════════════════
    # Final models on ALL data
    # ═══════════════════════════════════════════
    log('\nTraining final models on all data...')
    X_full, y_full = to_xy(df_dataset)
    del df_dataset; gc.collect()

    avg_lgbm_iter = int(np.mean(best_iters['lgbm']) * 1.1)
    avg_xgb_iter = int(np.mean(best_iters['xgb']) * 1.1)
    avg_cat_iter = int(np.mean(best_iters['cat']) * 1.1)

    for si, seed in enumerate(seeds):
        log(f'Final seed {seed}...')

        fl = lgb.LGBMClassifier(
            objective='binary', metric='average_precision',
            device='gpu', gpu_platform_id=0, gpu_device_id=0,
            scale_pos_weight=scale_pos,
            n_estimators=avg_lgbm_iter, learning_rate=0.03,
            num_leaves=255, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0, max_bin=255,
            random_state=seed, n_jobs=4, verbose=-1,
        )
        fl.fit(X_full, y_full)
        fl.booster_.save_model(str(MODELS_OUT / f'lgbm_s{seed}.txt'))

        fx = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='aucpr',
            tree_method='hist', device='cuda',
            scale_pos_weight=scale_pos,
            n_estimators=avg_xgb_iter, learning_rate=0.03,
            max_depth=8, min_child_weight=30,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0, max_bin=255,
            random_state=seed, n_jobs=4, verbosity=0,
        )
        fx.fit(X_full, y_full)
        fx.save_model(str(MODELS_OUT / f'xgb_s{seed}.json'))

        fc = CatBoostClassifier(
            iterations=avg_cat_iter, learning_rate=0.03, depth=8,
            task_type='GPU', devices='0',
            loss_function='Logloss', auto_class_weights='Balanced',
            l2_leaf_reg=3.0, random_seed=seed, verbose=0,
        )
        fc.fit(X_full, y_full)
        fc.save_model(str(MODELS_OUT / f'cat_s{seed}.cbm'))

    del X_full, y_full; gc.collect()

    # ═══════════════════════════════════════════
    # Predict test (with dedup)
    # ═══════════════════════════════════════════
    log('\nLoading test features...')
    df_test = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    n_before = len(df_test)
    df_test = df_test.unique(subset=['event_id'], keep='first')
    log(f'Dedup: {n_before} -> {len(df_test)} (-{n_before - len(df_test)})')

    df_test = add_v4_features(df_test)
    X_test = df_test.select(available_feats).to_pandas()
    event_ids = df_test['event_id'].to_numpy()
    log(f'X_test: {X_test.shape}')

    # Average predictions across seeds
    lgbm_preds = []
    xgb_preds = []
    cat_preds = []

    for seed in seeds:
        b = lgb.Booster(model_file=str(MODELS_OUT / f'lgbm_s{seed}.txt'))
        lgbm_preds.append(b.predict(X_test))

        m = xgb.XGBClassifier()
        m.load_model(str(MODELS_OUT / f'xgb_s{seed}.json'))
        xgb_preds.append(m.predict_proba(X_test)[:, 1])

        m = CatBoostClassifier()
        m.load_model(str(MODELS_OUT / f'cat_s{seed}.cbm'))
        cat_preds.append(m.predict_proba(X_test)[:, 1])

    avg_l = np.mean(lgbm_preds, axis=0)
    avg_x = np.mean(xgb_preds, axis=0)
    avg_c = np.mean(cat_preds, axis=0)

    if use_rank:
        ens_test = (rankdata(avg_l) * weights['lgbm'] +
                    rankdata(avg_x) * weights['xgb'] +
                    rankdata(avg_c) * weights['catboost'])
    else:
        ens_test = (avg_l * prob_weights['lgbm'] +
                    avg_x * prob_weights['xgb'] +
                    avg_c * prob_weights['catboost'])

    log(f'Predictions: min={ens_test.min():.4f}  max={ens_test.max():.4f}  mean={ens_test.mean():.4f}')

    # Build submission
    sample = pl.read_csv(DATA / 'sample_submit.csv')
    submit = pl.DataFrame({'event_id': event_ids, 'predict': ens_test})
    submit = sample.select('event_id').join(submit, on='event_id', how='left')

    assert len(submit) == len(sample), f"Size mismatch: {len(submit)} vs {len(sample)}"

    n_null = submit['predict'].is_null().sum()
    if n_null > 0:
        median_pred = submit['predict'].drop_nulls().median()
        submit = submit.with_columns(pl.col('predict').fill_null(median_pred))
        log(f'Filled {n_null} missing with median')

    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = SUBMIT_OUT / f'submit_v4_{ts}.csv'
    submit.write_csv(out_path)
    log(f'Saved: {out_path}')
    log(f'Rows: {len(submit):,}')

    # Feature importance
    log('\nTop 20 features (final LGBM):')
    b = lgb.Booster(model_file=str(MODELS_OUT / f'lgbm_s42.txt'))
    imp = pd.DataFrame({
        'feature': available_feats,
        'importance': b.feature_importance(),
    }).sort_values('importance', ascending=False)
    print(imp.head(20).to_string(index=False))

    log(f'\nTotal time: {(time.time()-t_start)/60:.1f} min')
    log('ALL DONE')


if __name__ == '__main__':
    main()
