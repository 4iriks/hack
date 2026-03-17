"""
Pipeline v9: Two-Stage Fraud Detection.

INSIGHT: One model can't optimally do two things at once:
  (a) separate suspicious from normal (85M green vs 87K labeled)
  (b) separate fraud from confirmed (51K vs 36K)

APPROACH:
  Stage 1 — Suspicious Detector: green(0) vs labeled(1). Learns "normal" patterns.
  Stage 2 — Fraud Classifier: confirmed(0) vs fraud(1). Learns fraud-specific patterns.
  Combine: rank(suspicious) × rank(fraud) → final score.

Also tests:
  Method A: rank_suspicious × rank_fraud
  Method B: suspicious_score as extra feature in labeled+green model
  Method C: stacking (meta-model on Stage 1 + Stage 2 predictions)
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from scipy.optimize import minimize
from pathlib import Path
from datetime import datetime
import gc, json, time, os

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_OUT  = ROOT / 'models_v9'
SUBMIT_OUT  = ROOT / 'submissions'

for d in [MODELS_OUT, SUBMIT_OUT]:
    d.mkdir(exist_ok=True)

PY = '/home/vadim/PyPr/hak/venv/bin/python'

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


# ═══════════════════════════════════════════
# Feature Engineering (same as v8)
# ═══════════════════════════════════════════

def add_features(df: pl.DataFrame) -> pl.DataFrame:
    new_cols = []
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

    if 'pos_cd' in df.columns:
        df = df.with_columns([
            (pl.col('pos_cd') == 1).fill_null(False).cast(pl.Int8).alias('is_manual_entry'),
            pl.col('pos_cd').is_not_null().cast(pl.Int8).alias('has_pos_data'),
        ])
    if 'timezone' in df.columns:
        df = df.with_columns(pl.col('timezone').is_null().cast(pl.Int8).alias('is_no_timezone'))

    ratio_cols = []
    for short, long in [('1h','24h'), ('24h','7d'), ('7d','30d')]:
        cs, cl = f'cnt_{short}', f'cnt_{long}'
        if cs in df.columns and cl in df.columns:
            ratio_cols.append((pl.col(cs) / (pl.col(cl) + 1)).alias(f'cnt_ratio_{short}_{long}'))
        cs2, cl2 = f'amt_sum_{short}', f'amt_sum_{long}'
        if cs2 in df.columns and cl2 in df.columns:
            ratio_cols.append((pl.col(cs2) / (pl.col(cl2).abs() + 1)).alias(f'amt_ratio_{short}_{long}'))
    if 'operaton_amt' in df.columns:
        for w in ['24h', '7d', '30d']:
            c = f'amt_sum_{w}'
            if c in df.columns:
                ratio_cols.append((pl.col('operaton_amt') / (pl.col(c).abs() + 1)).alias(f'amt_cur_ratio_{w}'))
    if ratio_cols:
        df = df.with_columns(ratio_cols)

    time_cols = []
    if 'hour' in df.columns and 'weekday' in df.columns:
        time_cols.append((pl.col('hour') * 7 + pl.col('weekday')).alias('hour_weekday'))
    if 'event_dttm' in df.columns:
        time_cols.append(pl.col('event_dttm').dt.day().cast(pl.Int8).alias('day_of_month'))
    if time_cols:
        df = df.with_columns(time_cols)

    if all(c in df.columns for c in ['compromised','web_rdp_connection','phone_voip_call_state','developer_tools']):
        df = df.with_columns(
            (pl.col('compromised') * 3 + pl.col('web_rdp_connection') * 2 +
             pl.col('phone_voip_call_state') * 2 + pl.col('developer_tools')).alias('security_risk_score'))

    if 'secs_since_last' in df.columns:
        df = df.with_columns([
            (pl.col('secs_since_last') < 60).cast(pl.Int8).alias('is_fast_60s'),
            pl.col('secs_since_last').log1p().alias('log_secs_since_last'),
        ])
    if 'amt_sum_30d' in df.columns and 'cnt_30d' in df.columns:
        df = df.with_columns((pl.col('amt_sum_30d') / (pl.col('cnt_30d') + 1)).alias('avg_amt_30d'))
    if 'amt_sum_7d' in df.columns and 'cnt_7d' in df.columns:
        df = df.with_columns((pl.col('amt_sum_7d') / (pl.col('cnt_7d') + 1)).alias('avg_amt_7d'))

    if 'avg_amt_30d' in df.columns and 'operaton_amt' in df.columns:
        df = df.with_columns((pl.col('operaton_amt') / (pl.col('avg_amt_30d') + 1)).alias('amt_deviation_30d'))
    if 'cnt_1h' in df.columns and 'cnt_24h' in df.columns:
        df = df.with_columns((pl.col('cnt_1h') / (pl.col('cnt_24h') / 24 + 0.01)).alias('hourly_activity_ratio'))
    if 'phone_voip_call_state' in df.columns and 'log_amount' in df.columns:
        df = df.with_columns((pl.col('phone_voip_call_state') * pl.col('log_amount')).alias('voip_x_logamt'))
    if 'operaton_amt' in df.columns and 'amt_sum_30d' in df.columns:
        df = df.with_columns((pl.col('operaton_amt') / (pl.col('amt_sum_30d') + 1)).alias('amt_pct_of_30d'))
    if 'session_ops_before' in df.columns:
        df = df.with_columns((pl.col('session_ops_before') == 0).cast(pl.Int8).alias('is_first_in_session'))
    if 'cnt_30d' in df.columns:
        df = df.with_columns(pl.col('cnt_30d').cast(pl.Float32).log1p().alias('log_cnt_30d'))
    if 'is_manual_entry' in df.columns and 'log_amount' in df.columns:
        df = df.with_columns((pl.col('is_manual_entry') * pl.col('log_amount')).alias('manual_entry_x_amt'))
    return df


def add_customer_profiles(df: pl.DataFrame, profiles: pl.DataFrame) -> pl.DataFrame:
    df = df.join(profiles, on='customer_id', how='left')
    dev = []
    if 'cust_avg_amt' in df.columns and 'operaton_amt' in df.columns:
        dev.extend([
            ((pl.col('operaton_amt') - pl.col('cust_avg_amt')) / (pl.col('cust_std_amt') + 1)).alias('amt_zscore'),
            (pl.col('operaton_amt') / (pl.col('cust_med_amt') + 1)).alias('amt_vs_median'),
        ])
    if dev:
        df = df.with_columns(dev)
    if 'cust_last_epoch' in df.columns and 'event_dttm' in df.columns:
        df = df.with_columns(
            ((pl.col('event_dttm').dt.epoch('s') - pl.col('cust_last_epoch')) / 86400).alias('dormancy_days'))
    if 'cust_avg_hour' in df.columns and 'hour' in df.columns:
        df = df.with_columns(
            (pl.col('hour').cast(pl.Float64) - pl.col('cust_avg_hour')).abs().alias('hour_deviation'))
    if 'cnt_30d' in df.columns and 'cust_avg_gap_sec' in df.columns:
        df = df.with_columns(
            (pl.col('cnt_30d').cast(pl.Float64) / (2592000 / (pl.col('cust_avg_gap_sec') + 1) + 0.01)).alias('freq_vs_historical'))
    # Fill nulls
    for c in df.columns:
        if c.startswith('cust_') or c in ('amt_zscore','amt_vs_median','dormancy_days','hour_deviation','freq_vs_historical'):
            df = df.with_columns(pl.col(c).fill_null(0.0))
    return df


FEATURE_COLS = [
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
    'session_ops_before', 'session_amt_before', 'timezone',
    'event_desc',
    'has_browser_lang', 'has_accept_lang', 'screen_w', 'screen_h',
    'has_device_ver', 'has_session',
    'cnt_ratio_1h_24h', 'cnt_ratio_24h_7d', 'cnt_ratio_7d_30d',
    'amt_ratio_1h_24h', 'amt_ratio_24h_7d', 'amt_ratio_7d_30d',
    'amt_cur_ratio_24h', 'amt_cur_ratio_7d', 'amt_cur_ratio_30d',
    'hour_weekday', 'day_of_month',
    'security_risk_score',
    'is_fast_60s', 'log_secs_since_last',
    'avg_amt_30d', 'avg_amt_7d', 'amt_deviation_30d',
    'hourly_activity_ratio', 'voip_x_logamt',
    'amt_pct_of_30d', 'is_first_in_session', 'log_cnt_30d',
    'is_manual_entry', 'has_pos_data', 'is_no_screen', 'is_no_timezone',
    'manual_entry_x_amt',
    'cust_avg_amt', 'cust_std_amt', 'cust_med_amt', 'cust_max_amt', 'cust_p95_amt',
    'cust_n_tx', 'cust_n_unique_mcc', 'cust_n_unique_channel',
    'cust_avg_hour', 'cust_hour_std', 'cust_pct_night', 'cust_pct_high_risk',
    'cust_tenure_days', 'cust_avg_gap_sec',
    'amt_zscore', 'amt_vs_median', 'dormancy_days',
    'hour_deviation', 'freq_vs_historical',
]


LGBM_BASE = dict(
    objective='binary', metric='average_precision',
    device='gpu', gpu_platform_id=0, gpu_device_id=0,
    n_estimators=10000, learning_rate=0.02,
    num_leaves=127, min_child_samples=200,
    subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
    n_jobs=4, verbose=-1,
)


def train_lgbm(X_tr, y_tr, X_val, y_val, seed, spw=None, tag=''):
    """Train one LGBM with early stopping."""
    params = {**LGBM_BASE, 'random_state': seed}
    if spw is not None:
        params['scale_pos_weight'] = spw
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(500)])
    p = m.predict_proba(X_val)[:, 1]
    s = average_precision_score(y_val, p)
    log(f'  {tag}[{seed}]: PR-AUC={s:.4f} iter={m.best_iteration_}')
    return m, p, s


def main():
    t_start = time.time()
    print('\n' + '='*60)
    print('PIPELINE V9: Two-Stage Fraud Detection')
    print('='*60)

    # ─── Load data ───
    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    label_ids = set(labels['event_id'].to_list())
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())
    confirmed_ids = set(labels.filter(pl.col('target') == 0)['event_id'].to_list())
    log(f'Labels: {len(labels):,} (fraud={len(fraud_ids):,}, confirmed={len(confirmed_ids):,})')

    log('Loading full data (2.6M rows)...')
    df = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    df = add_features(df)
    df = add_customer_profiles(df, profiles)
    log(f'Data: {len(df):,} rows, RAM={_ram_gb():.1f}GB')

    available_feats = [c for c in FEATURE_COLS if c in df.columns]
    log(f'Features: {len(available_feats)}')

    # ─── Create targets ───
    # is_suspicious: 1 if labeled (fraud or confirmed), 0 if green
    df = df.with_columns(
        pl.col('event_id').is_in(label_ids).cast(pl.Int8).alias('is_suspicious'),
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'),
    )

    # ─── Time split ───
    val_dt = datetime(2025, 4, 1)
    tr = df.filter(pl.col('event_dttm') < val_dt)
    va = df.filter(pl.col('event_dttm') >= val_dt)
    log(f'Train: {len(tr):,}, Val: {len(va):,}')

    # ─── Stage 1: Suspicious Detector (green vs labeled) ───
    print('\n' + '─'*40)
    print('STAGE 1: Suspicious Detector')
    print('─'*40)

    X_tr1 = tr.select(available_feats).to_pandas().astype(np.float32)
    y_tr1 = tr['is_suspicious'].to_numpy()
    X_va1 = va.select(available_feats).to_pandas().astype(np.float32)
    y_va1 = va['is_suspicious'].to_numpy()

    spw1 = (y_tr1 == 0).sum() / (y_tr1 == 1).sum()
    log(f'Stage1 train: {len(X_tr1):,} (suspicious={y_tr1.sum():,}, green={len(X_tr1)-y_tr1.sum():,}, spw={spw1:.1f})')

    seeds = [42, 123, 777, 2024, 31337]
    s1_preds_val, s1_models, s1_iters = [], [], []
    for seed in seeds:
        m, p, s = train_lgbm(X_tr1, y_tr1, X_va1, y_va1, seed, spw=spw1, tag='S1')
        s1_preds_val.append(p)
        s1_models.append(m)
        s1_iters.append(m.best_iteration_)
    s1_avg = np.mean(s1_preds_val, axis=0)
    log(f'Stage1 avg: PR-AUC={average_precision_score(y_va1, s1_avg):.4f}')

    del X_tr1, y_tr1; gc.collect()

    # ─── Stage 2: Fraud Classifier (fraud vs confirmed, labeled only) ───
    print('\n' + '─'*40)
    print('STAGE 2: Fraud Classifier (labeled only)')
    print('─'*40)

    tr_labeled = tr.filter(pl.col('event_id').is_in(label_ids))
    va_labeled = va.filter(pl.col('event_id').is_in(label_ids))

    X_tr2 = tr_labeled.select(available_feats).to_pandas().astype(np.float32)
    y_tr2 = tr_labeled['is_fraud'].to_numpy()
    X_va2 = va_labeled.select(available_feats).to_pandas().astype(np.float32)
    y_va2 = va_labeled['is_fraud'].to_numpy()

    spw2 = (y_tr2 == 0).sum() / max((y_tr2 == 1).sum(), 1)
    log(f'Stage2 train: {len(X_tr2):,} (fraud={y_tr2.sum():,}, confirmed={len(X_tr2)-y_tr2.sum():,}, spw={spw2:.2f})')

    s2_preds_val_labeled, s2_models, s2_iters = [], [], []
    for seed in seeds:
        m, p, s = train_lgbm(X_tr2, y_tr2, X_va2, y_va2, seed, spw=spw2, tag='S2')
        s2_preds_val_labeled.append(p)
        s2_models.append(m)
        s2_iters.append(m.best_iteration_)
    s2_avg_labeled = np.mean(s2_preds_val_labeled, axis=0)
    log(f'Stage2 avg (labeled-only val): PR-AUC={average_precision_score(y_va2, s2_avg_labeled):.4f}')

    # Stage 2 predictions on FULL val (including green)
    s2_preds_full = []
    for m in s2_models:
        s2_preds_full.append(m.predict_proba(X_va1)[:, 1])
    s2_avg = np.mean(s2_preds_full, axis=0)

    del X_tr2, y_tr2, X_va2, y_va2, tr_labeled, va_labeled; gc.collect()

    # ─── Stage 3 (bonus): labeled+green model with suspicious score as feature ───
    print('\n' + '─'*40)
    print('STAGE 3: Main model + suspicious score as feature')
    print('─'*40)

    # Get Stage 1 OOF predictions for train
    s1_train_preds = []
    for m in s1_models:
        s1_train_preds.append(m.predict_proba(
            tr.select(available_feats).to_pandas().astype(np.float32))[:, 1])
    s1_train_avg = np.mean(s1_train_preds, axis=0)

    # Add suspicious score as feature
    feats_plus = available_feats + ['susp_score']
    X_tr3 = tr.select(available_feats).to_pandas().astype(np.float32)
    X_tr3['susp_score'] = s1_train_avg.astype(np.float32)
    y_tr3 = tr['target'].to_numpy().astype(int)
    X_va3 = X_va1.copy()
    X_va3['susp_score'] = s1_avg.astype(np.float32)
    y_va3 = va['target'].to_numpy().astype(int)

    spw3 = (y_tr3 == 0).sum() / max((y_tr3 == 1).sum(), 1)

    s3_preds_val, s3_models, s3_iters = [], [], []
    for seed in seeds:
        m, p, s = train_lgbm(X_tr3, y_tr3, X_va3, y_va3, seed, spw=spw3, tag='S3')
        s3_preds_val.append(p)
        s3_models.append(m)
        s3_iters.append(m.best_iteration_)
    s3_avg = np.mean(s3_preds_val, axis=0)
    log(f'Stage3 avg: PR-AUC={average_precision_score(y_va3, s3_avg):.4f}')

    del X_tr3, y_tr3; gc.collect()

    # ─── Evaluate combination methods on val ───
    print('\n' + '─'*40)
    print('EVALUATION: Combining methods')
    print('─'*40)

    # True fraud labels for full val
    y_val_fraud = va['target'].to_numpy().astype(int)

    # Method A: rank(suspicious) × rank(fraud)
    score_A = rankdata(s1_avg) * rankdata(s2_avg)
    prauc_A = average_precision_score(y_val_fraud, score_A)

    # Method B: just suspicious score
    prauc_S1 = average_precision_score(y_val_fraud, s1_avg)

    # Method C: Stage 3 (main + susp feature)
    prauc_S3 = average_precision_score(y_val_fraud, s3_avg)

    # Method D: rank blend of all three
    score_D = rankdata(s1_avg) * 0.3 + rankdata(s2_avg) * 0.3 + rankdata(s3_avg) * 0.4
    prauc_D = average_precision_score(y_val_fraud, score_D)

    # Method E: optimized weights
    all_scores = [s1_avg, s2_avg, s3_avg]
    all_ranks = [rankdata(s) for s in all_scores]
    names = ['suspicious', 'fraud_clf', 'main+susp']

    def neg_prauc(w):
        w = np.abs(w); w /= w.sum()
        return -average_precision_score(y_val_fraud, sum(w[i]*all_ranks[i] for i in range(3)))
    best_s, best_w = -1, None
    for seed in range(100):
        w0 = np.random.RandomState(seed).dirichlet(np.ones(3))
        res = minimize(neg_prauc, w0, method='Nelder-Mead', options={'maxiter': 3000})
        if -res.fun > best_s:
            best_s = -res.fun; best_w = np.abs(np.array(res.x))
    best_w = best_w / best_w.sum()
    prauc_E = best_s

    log(f'\n=== VAL RESULTS ===')
    log(f'  Method A (susp × fraud):     {prauc_A:.4f}')
    log(f'  Method B (susp only):         {prauc_S1:.4f}')
    log(f'  Method C (main + susp feat):  {prauc_S3:.4f}')
    log(f'  Method D (equal blend):       {prauc_D:.4f}')
    log(f'  Method E (optimized blend):   {prauc_E:.4f}')
    log(f'  Weights E: {dict(zip(names, best_w.round(4)))}')
    log(f'  (v8 baseline was:             0.3145)')

    # Pick best method
    results = {'A': prauc_A, 'B': prauc_S1, 'C': prauc_S3, 'D': prauc_D, 'E': prauc_E}
    best_method = max(results, key=results.get)
    log(f'\n  BEST: Method {best_method} = {results[best_method]:.4f}')

    # Save config
    config = {
        'val_results': {k: float(v) for k, v in results.items()},
        'best_method': best_method,
        'weights_E': dict(zip(names, best_w.tolist())),
        's1_iters': s1_iters, 's2_iters': s2_iters, 's3_iters': s3_iters,
        'n_features': len(available_feats),
    }
    with open(MODELS_OUT / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    del X_va1, X_va3, s1_preds_val, s2_preds_full, s3_preds_val; gc.collect()

    # ─── Final: train on ALL data, predict test ───
    print('\n' + '─'*40)
    print('FINAL: Training on ALL data')
    print('─'*40)

    X_full = df.select(available_feats).to_pandas().astype(np.float32)
    y_susp_full = df['is_suspicious'].to_numpy()
    y_fraud_full = df['is_fraud'].to_numpy()
    y_target_full = df['target'].to_numpy().astype(int)
    log(f'Full data: {len(X_full):,}')

    # Final Stage 1
    avg_s1_iter = int(np.mean(s1_iters) * 1.1)
    spw1_full = (y_susp_full == 0).sum() / max((y_susp_full == 1).sum(), 1)
    log(f'Final Stage 1: {avg_s1_iter} iters, spw={spw1_full:.1f}')
    final_s1 = []
    for seed in seeds:
        m = lgb.LGBMClassifier(**{**LGBM_BASE, 'random_state': seed,
            'scale_pos_weight': spw1_full, 'n_estimators': avg_s1_iter})
        m.fit(X_full, y_susp_full)
        m.booster_.save_model(str(MODELS_OUT / f's1_lgbm_s{seed}.txt'))
        final_s1.append(m)
        log(f'  S1 final [{seed}] done')
    del y_susp_full; gc.collect()

    # Final Stage 2 (labeled only)
    labeled_mask = df['event_id'].is_in(label_ids).to_numpy()
    X_labeled = X_full[labeled_mask]
    y_labeled = y_fraud_full[labeled_mask]
    avg_s2_iter = int(np.mean(s2_iters) * 1.1)
    spw2_full = (y_labeled == 0).sum() / max((y_labeled == 1).sum(), 1)
    log(f'Final Stage 2: {avg_s2_iter} iters, {len(X_labeled):,} rows, spw={spw2_full:.2f}')
    final_s2 = []
    for seed in seeds:
        m = lgb.LGBMClassifier(**{**LGBM_BASE, 'random_state': seed,
            'scale_pos_weight': spw2_full, 'n_estimators': avg_s2_iter})
        m.fit(X_labeled, y_labeled)
        m.booster_.save_model(str(MODELS_OUT / f's2_lgbm_s{seed}.txt'))
        final_s2.append(m)
        log(f'  S2 final [{seed}] done')
    del X_labeled, y_labeled; gc.collect()

    # Final Stage 3 (main + susp score)
    s1_full_preds = np.mean([m.predict_proba(X_full)[:, 1] for m in final_s1], axis=0)
    X_full_plus = X_full.copy()
    X_full_plus['susp_score'] = s1_full_preds.astype(np.float32)
    avg_s3_iter = int(np.mean(s3_iters) * 1.1)
    spw3_full = (y_target_full == 0).sum() / max((y_target_full == 1).sum(), 1)
    log(f'Final Stage 3: {avg_s3_iter} iters, spw={spw3_full:.1f}')
    final_s3 = []
    for seed in seeds:
        m = lgb.LGBMClassifier(**{**LGBM_BASE, 'random_state': seed,
            'scale_pos_weight': spw3_full, 'n_estimators': avg_s3_iter})
        m.fit(X_full_plus, y_target_full)
        m.booster_.save_model(str(MODELS_OUT / f's3_lgbm_s{seed}.txt'))
        final_s3.append(m)
        log(f'  S3 final [{seed}] done')
    del X_full, X_full_plus, y_target_full, y_fraud_full, df; gc.collect()

    # ─── Predict test ───
    print('\n' + '─'*40)
    print('TEST PREDICTIONS')
    print('─'*40)

    df_test = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    df_test = df_test.unique(subset=['event_id'], keep='first')
    df_test = add_features(df_test)
    df_test = add_customer_profiles(df_test, profiles)
    X_test = df_test.select(available_feats).to_pandas().astype(np.float32)
    event_ids = df_test['event_id'].to_numpy()
    log(f'Test: {X_test.shape}')

    # Stage 1 predictions
    test_s1 = np.mean([lgb.Booster(model_file=str(MODELS_OUT / f's1_lgbm_s{s}.txt')).predict(X_test)
                        for s in seeds], axis=0)
    # Stage 2 predictions
    test_s2 = np.mean([lgb.Booster(model_file=str(MODELS_OUT / f's2_lgbm_s{s}.txt')).predict(X_test)
                        for s in seeds], axis=0)
    # Stage 3 predictions
    X_test_plus = X_test.copy()
    X_test_plus['susp_score'] = test_s1.astype(np.float32)
    test_s3 = np.mean([lgb.Booster(model_file=str(MODELS_OUT / f's3_lgbm_s{s}.txt')).predict(X_test_plus)
                        for s in seeds], axis=0)

    # Apply best combination method
    if best_method == 'A':
        final_score = rankdata(test_s1) * rankdata(test_s2)
    elif best_method == 'B':
        final_score = test_s1
    elif best_method == 'C':
        final_score = test_s3
    elif best_method == 'D':
        final_score = rankdata(test_s1) * 0.3 + rankdata(test_s2) * 0.3 + rankdata(test_s3) * 0.4
    else:  # E
        final_score = sum(best_w[i] * rankdata(s) for i, s in enumerate([test_s1, test_s2, test_s3]))

    log(f'Final score: min={final_score.min():.1f} max={final_score.max():.1f}')

    # Save ALL submissions (each method) for flexibility
    sample = pl.read_csv(DATA / 'sample_submit.csv')
    ts = datetime.now().strftime('%Y%m%d_%H%M')

    for method_name, score in [
        ('A_susp_x_fraud', rankdata(test_s1) * rankdata(test_s2)),
        ('C_main_plus_susp', test_s3),
        ('E_optimized', sum(best_w[i] * rankdata(s) for i, s in enumerate([test_s1, test_s2, test_s3]))),
    ]:
        sub = pl.DataFrame({'event_id': event_ids, 'predict': score})
        sub = sample.select('event_id').join(sub, on='event_id', how='left')
        n_null = sub['predict'].is_null().sum()
        if n_null > 0:
            sub = sub.with_columns(pl.col('predict').fill_null(sub['predict'].drop_nulls().median()))
        path = SUBMIT_OUT / f'submit_v9_{method_name}_{ts}.csv'
        sub.write_csv(path)
        log(f'Saved: {path.name}')

    # Feature importance for Stage 1
    log('\nStage 1 top features (what makes tx suspicious):')
    b = lgb.Booster(model_file=str(MODELS_OUT / 's1_lgbm_s42.txt'))
    imp = pd.DataFrame({'feature': available_feats, 'importance': b.feature_importance()})
    print(imp.sort_values('importance', ascending=False).head(20).to_string(index=False))

    log('\nStage 2 top features (fraud vs confirmed):')
    b = lgb.Booster(model_file=str(MODELS_OUT / 's2_lgbm_s42.txt'))
    imp2 = pd.DataFrame({'feature': available_feats, 'importance': b.feature_importance()})
    print(imp2.sort_values('importance', ascending=False).head(20).to_string(index=False))

    log(f'\nTotal: {(time.time()-t_start)/60:.1f} min')
    log('ALL DONE')


if __name__ == '__main__':
    main()
