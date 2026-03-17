"""
Pipeline v3: исправлены дубли в test, улучшенные фичи, 3-модельный ансамбль.
Ключевые фиксы:
  - Дедупликация test event_ids (12,405 дублей в v1/v2)
  - Target encoding с proper LOO
  - Frequency encoding
  - Более агрессивная регуляризация (борьба с переобучением)
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize
from pathlib import Path
from datetime import datetime
import gc, json, time

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_OUT  = ROOT / 'models_v3'
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


# ─── Feature Engineering ─────────────────────

def add_extra_features(df: pl.DataFrame) -> pl.DataFrame:
    """Добавляем фичи из неиспользованных колонок + ratio features."""
    new_cols = []

    # Неиспользованные колонки
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

    # Security risk score (weighted)
    if all(c in df.columns for c in ['compromised','web_rdp_connection','phone_voip_call_state','developer_tools']):
        df = df.with_columns([
            (pl.col('compromised') * 3 + pl.col('web_rdp_connection') * 2 +
             pl.col('phone_voip_call_state') * 2 + pl.col('developer_tools')).alias('security_risk_score'),
        ])

    # Session interaction
    if 'session_ops_before' in df.columns and 'log_amount' in df.columns:
        df = df.with_columns([
            (pl.col('session_ops_before') * pl.col('log_amount')).alias('session_ops_x_log_amt'),
        ])

    return df


# All feature columns
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
    # Added in v2/v3
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
]

# Target encoding columns
TE_COLS = ['mcc_code', 'event_type_nm', 'event_desc', 'channel_indicator_type',
           'channel_indicator_sub_type', 'currency_iso_cd', 'operating_system_type', 'pos_cd']

# Frequency encoding columns
FREQ_COLS = ['mcc_code', 'event_type_nm', 'event_desc', 'channel_indicator_type']


def compute_target_encoding_stats(df_train: pl.DataFrame, cols: list,
                                  target_col='target', smoothing=20):
    """Вычисляем статистики target encoding из тренировочного набора."""
    global_mean = df_train[target_col].mean()
    stats = {'global_mean': global_mean}

    for col in cols:
        col_stats = df_train.group_by(col).agg([
            pl.col(target_col).sum().alias('_sum'),
            pl.col(target_col).count().alias('_cnt'),
        ])
        col_stats = col_stats.with_columns([
            (pl.col('_sum') / (pl.col('_cnt') + smoothing) +
             global_mean * smoothing / (pl.col('_cnt') + smoothing)).alias(f'te_{col}')
        ]).select([col, f'te_{col}'])
        stats[col] = col_stats

    return stats


def apply_target_encoding_train(df: pl.DataFrame, cols: list,
                                target_col='target', smoothing=20):
    """LOO target encoding для train (anti-leakage)."""
    global_mean = df[target_col].mean()

    for col in cols:
        agg = df.group_by(col).agg([
            pl.col(target_col).sum().alias('_te_sum'),
            pl.col(target_col).count().alias('_te_cnt'),
        ])
        df = df.join(agg, on=col, how='left')
        df = df.with_columns([
            pl.when(pl.col('_te_cnt') > 1)
            .then(
                (pl.col('_te_sum') - pl.col(target_col)) / (pl.col('_te_cnt') - 1 + smoothing) +
                global_mean * smoothing / (pl.col('_te_cnt') - 1 + smoothing)
            )
            .otherwise(global_mean)
            .alias(f'te_{col}')
        ]).drop(['_te_sum', '_te_cnt'])

    return df


def apply_target_encoding_test(df: pl.DataFrame, te_stats: dict, cols: list):
    """Применяем target encoding к test/val."""
    for col in cols:
        col_stats = te_stats[col]
        df = df.join(col_stats, on=col, how='left')
        df = df.with_columns(
            pl.col(f'te_{col}').fill_null(te_stats['global_mean'])
        )
    return df


def add_frequency_encoding(df: pl.DataFrame, cols: list):
    """Частотное кодирование категорий."""
    total = len(df)
    for col in cols:
        if col in df.columns:
            counts = df.group_by(col).agg(pl.len().alias(f'freq_{col}'))
            df = df.join(counts, on=col, how='left')
            df = df.with_columns((pl.col(f'freq_{col}') / total).alias(f'freq_{col}'))
    return df


def compute_freq_stats(df: pl.DataFrame, cols: list):
    """Вычисляем частотные статистики из train для применения к test."""
    total = len(df)
    stats = {}
    for col in cols:
        if col in df.columns:
            counts = df.group_by(col).agg(pl.len().alias(f'freq_{col}'))
            counts = counts.with_columns((pl.col(f'freq_{col}') / total).alias(f'freq_{col}'))
            stats[col] = counts
    return stats


def apply_freq_encoding(df: pl.DataFrame, freq_stats: dict, cols: list):
    """Применяем частотное кодирование из train."""
    for col in cols:
        if col in freq_stats:
            df = df.join(freq_stats[col], on=col, how='left')
            df = df.with_columns(pl.col(f'freq_{col}').fill_null(0.0))
    return df


def optimize_ensemble_weights(preds_list, y_true, names):
    """Оптимизируем веса ансамбля."""
    n = len(preds_list)
    def neg_prauc(w):
        w = np.array(w)
        w = w / w.sum()
        ens = sum(w[i] * preds_list[i] for i in range(n))
        return -average_precision_score(y_true, ens)

    best_score = -1
    best_w = None
    for seed in range(30):
        rng = np.random.RandomState(seed)
        w0 = rng.dirichlet(np.ones(n))
        res = minimize(neg_prauc, w0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-7})
        if -res.fun > best_score:
            best_score = -res.fun
            best_w = np.array(res.x)

    best_w = best_w / best_w.sum()
    log(f'Optimized weights: {dict(zip(names, best_w.round(4)))}')
    log(f'Optimized ensemble PR-AUC: {best_score:.4f}')
    return dict(zip(names, best_w.tolist()))


def main():
    t_start = time.time()
    findings = []  # Document findings

    print('\n' + '='*60)
    print('PIPELINE V3: Fixed duplicates + TE + freq + 3-model ensemble')
    print('='*60)

    # ══════════════════════════════════════════
    # Load and prepare data
    # ══════════════════════════════════════════
    log('Loading train features...')
    df_train_raw = pl.read_parquet(FEATURES_IN / 'train_features.parquet')
    df_train_raw = add_extra_features(df_train_raw)

    log(f'Train shape: {df_train_raw.shape}')
    log(f'Fraud: {df_train_raw.filter(pl.col("target")==1).height:,}')
    log(f'Legit: {df_train_raw.filter(pl.col("target")==0).height:,}')

    # Temporal split
    val_dt = datetime(2025, 4, 1)
    df_tr = df_train_raw.filter(pl.col('event_dttm') < val_dt)
    df_val = df_train_raw.filter(pl.col('event_dttm') >= val_dt)
    del df_train_raw; gc.collect()

    log(f'Train: {len(df_tr):,}  Val: {len(df_val):,}')
    log(f'Train fraud: {df_tr.filter(pl.col("target")==1).height:,}')
    log(f'Val fraud: {df_val.filter(pl.col("target")==1).height:,}')

    # Target encoding (LOO on train, global on val)
    te_cols = [c for c in TE_COLS if c in df_tr.columns]
    log(f'Computing target encoding for: {te_cols}')
    te_stats = compute_target_encoding_stats(df_tr, te_cols)
    df_tr = apply_target_encoding_train(df_tr, te_cols)
    df_val = apply_target_encoding_test(df_val, te_stats, te_cols)

    # Frequency encoding
    freq_cols = [c for c in FREQ_COLS if c in df_tr.columns]
    log(f'Computing frequency encoding for: {freq_cols}')
    freq_stats = compute_freq_stats(df_tr, freq_cols)
    df_tr = add_frequency_encoding(df_tr, freq_cols)
    df_val = apply_freq_encoding(df_val, freq_stats, freq_cols)

    # Build feature list
    te_feat_names = [f'te_{c}' for c in te_cols]
    freq_feat_names = [f'freq_{c}' for c in freq_cols]
    all_feats = FEATURE_COLS + te_feat_names + freq_feat_names
    available_feats = [c for c in all_feats if c in df_tr.columns and c in df_val.columns]
    log(f'Total features: {len(available_feats)}')

    def to_xy(d):
        X = d.select(available_feats).to_pandas()
        # Replace inf with NaN, then fill NaN with 0 (LightGBM GPU requires no NaN in some cols)
        X = X.replace([np.inf, -np.inf], np.nan)
        # Fill NaN in TE/freq cols with global mean / 0
        for col in X.columns:
            if col.startswith('te_') or col.startswith('freq_'):
                X[col] = X[col].fillna(0.0)
        y = d['target'].to_numpy().astype(int)
        return X, y

    X_train, y_train = to_xy(df_tr)
    X_val, y_val = to_xy(df_val)

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos = n_neg / n_pos
    log(f'scale_pos_weight={scale_pos:.2f}  (pos={n_pos:,}  neg={n_neg:,})')

    # ══════════════════════════════════════════
    # Train models
    # ══════════════════════════════════════════

    # ── LightGBM GPU ──
    log('Training LightGBM GPU...')
    lgbm_model = lgb.LGBMClassifier(
        objective='binary', metric='average_precision',
        device='gpu', gpu_platform_id=0, gpu_device_id=0,
        scale_pos_weight=scale_pos,
        n_estimators=5000, learning_rate=0.02,
        num_leaves=127, max_depth=8,
        min_child_samples=100,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0,
        max_bin=255,
        random_state=42, n_jobs=4, verbose=-1,
    )
    lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                   callbacks=[lgb.early_stopping(300, verbose=True), lgb.log_evaluation(100)])
    lgbm_preds = lgbm_model.predict_proba(X_val)[:, 1]
    lgbm_score = average_precision_score(y_val, lgbm_preds)
    log(f'>>> LightGBM PR-AUC: {lgbm_score:.4f} (best_iter={lgbm_model.best_iteration_})')
    findings.append(f'LightGBM: PR-AUC={lgbm_score:.4f}, best_iter={lgbm_model.best_iteration_}')

    # Feature importance
    imp = pd.DataFrame({
        'feature': available_feats,
        'importance': lgbm_model.feature_importances_,
    }).sort_values('importance', ascending=False)
    log(f'Top 15 features:\n{imp.head(15).to_string(index=False)}')

    # ── XGBoost GPU ──
    log('Training XGBoost GPU...')
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='aucpr',
        tree_method='hist', device='cuda',
        scale_pos_weight=scale_pos,
        n_estimators=5000, learning_rate=0.02,
        max_depth=7, min_child_weight=50,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0,
        max_bin=255, gamma=0.1,
        random_state=42, n_jobs=4, verbosity=0,
        early_stopping_rounds=300,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
    xgb_score = average_precision_score(y_val, xgb_preds)
    log(f'>>> XGBoost PR-AUC: {xgb_score:.4f} (best_iter={xgb_model.best_iteration})')
    findings.append(f'XGBoost: PR-AUC={xgb_score:.4f}, best_iter={xgb_model.best_iteration}')

    # ── CatBoost GPU ──
    log('Training CatBoost GPU...')
    cat_model = CatBoostClassifier(
        iterations=5000, learning_rate=0.02, depth=7,
        task_type='GPU', devices='0',
        loss_function='Logloss', eval_metric='AUC',
        auto_class_weights='Balanced',
        l2_leaf_reg=5.0, min_data_in_leaf=50,
        random_seed=42, verbose=100, early_stopping_rounds=300,
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    cat_preds = cat_model.predict_proba(X_val)[:, 1]
    cat_score = average_precision_score(y_val, cat_preds)
    log(f'>>> CatBoost PR-AUC: {cat_score:.4f} (best_iter={cat_model.best_iteration_})')
    findings.append(f'CatBoost: PR-AUC={cat_score:.4f}, best_iter={cat_model.best_iteration_}')

    # ── Ensemble ──
    log('Optimizing ensemble weights...')
    weights = optimize_ensemble_weights(
        [lgbm_preds, xgb_preds, cat_preds], y_val,
        ['lgbm', 'xgb', 'catboost']
    )
    with open(MODELS_OUT / 'weights_v3.json', 'w') as f:
        json.dump(weights, f, indent=2)

    del X_train, y_train, X_val, y_val; gc.collect()

    # ══════════════════════════════════════════
    # Final models on ALL data
    # ══════════════════════════════════════════
    log('Rebuilding full dataset for final training...')

    # Recompute TE/freq on ALL train data
    df_full = pl.read_parquet(FEATURES_IN / 'train_features.parquet')
    df_full = add_extra_features(df_full)
    te_stats_full = compute_target_encoding_stats(df_full, te_cols)
    df_full = apply_target_encoding_train(df_full, te_cols)
    freq_stats_full = compute_freq_stats(df_full, freq_cols)
    df_full = add_frequency_encoding(df_full, freq_cols)

    X_full = df_full.select(available_feats).to_pandas().replace([np.inf, -np.inf], np.nan)
    for col in X_full.columns:
        if col.startswith('te_') or col.startswith('freq_'):
            X_full[col] = X_full[col].fillna(0.0)
    y_full = df_full['target'].to_numpy().astype(int)
    del df_full; gc.collect()

    log(f'Final training data: {X_full.shape}')

    # LightGBM final
    lgbm_best = lgbm_model.best_iteration_
    log(f'Training final LightGBM (iters={int(lgbm_best*1.1)})...')
    final_lgbm = lgb.LGBMClassifier(
        objective='binary', metric='average_precision',
        device='gpu', gpu_platform_id=0, gpu_device_id=0,
        scale_pos_weight=scale_pos,
        n_estimators=int(lgbm_best * 1.1), learning_rate=0.02,
        num_leaves=127, max_depth=8, min_child_samples=100,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0, max_bin=255,
        random_state=42, n_jobs=4, verbose=-1,
    )
    final_lgbm.fit(X_full, y_full)
    final_lgbm.booster_.save_model(str(MODELS_OUT / 'lgbm_final.txt'))

    # XGBoost final
    xgb_best = xgb_model.best_iteration
    log(f'Training final XGBoost (iters={int(xgb_best*1.1)})...')
    final_xgb = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='aucpr',
        tree_method='hist', device='cuda',
        scale_pos_weight=scale_pos,
        n_estimators=int(xgb_best * 1.1), learning_rate=0.02,
        max_depth=7, min_child_weight=50,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=5.0, max_bin=255, gamma=0.1,
        random_state=42, n_jobs=4, verbosity=0,
    )
    final_xgb.fit(X_full, y_full)
    final_xgb.save_model(str(MODELS_OUT / 'xgb_final.json'))

    # CatBoost final
    cat_best = cat_model.best_iteration_
    log(f'Training final CatBoost (iters={int(cat_best*1.1)})...')
    final_cat = CatBoostClassifier(
        iterations=int(cat_best * 1.1), learning_rate=0.02, depth=7,
        task_type='GPU', devices='0',
        loss_function='Logloss', auto_class_weights='Balanced',
        l2_leaf_reg=5.0, min_data_in_leaf=50,
        random_seed=42, verbose=100,
    )
    final_cat.fit(X_full, y_full)
    final_cat.save_model(str(MODELS_OUT / 'catboost_final.cbm'))

    del X_full, y_full; gc.collect()

    # ══════════════════════════════════════════
    # Predict test (WITH DEDUPLICATION)
    # ══════════════════════════════════════════
    log('Loading test features (with dedup)...')
    df_test = pl.read_parquet(FEATURES_IN / 'test_features.parquet')

    # CRITICAL FIX: deduplicate by event_id (keep first)
    n_before = len(df_test)
    df_test = df_test.unique(subset=['event_id'], keep='first')
    n_after = len(df_test)
    log(f'Dedup test: {n_before} -> {n_after} (removed {n_before - n_after} duplicates)')
    findings.append(f'Test dedup: removed {n_before - n_after} duplicate event_ids')

    df_test = add_extra_features(df_test)

    # Apply TE and freq from full train stats
    df_test = apply_target_encoding_test(df_test, te_stats_full, te_cols)
    df_test = apply_freq_encoding(df_test, freq_stats_full, freq_cols)

    X_test = df_test.select(available_feats).to_pandas().replace([np.inf, -np.inf], np.nan)
    for col in X_test.columns:
        if col.startswith('te_') or col.startswith('freq_'):
            X_test[col] = X_test[col].fillna(0.0)
    event_ids = df_test['event_id'].to_numpy()
    log(f'X_test: {X_test.shape}')

    # Predict
    lgbm_booster = lgb.Booster(model_file=str(MODELS_OUT / 'lgbm_final.txt'))
    lgbm_test = lgbm_booster.predict(X_test)

    xgb_loaded = xgb.XGBClassifier()
    xgb_loaded.load_model(str(MODELS_OUT / 'xgb_final.json'))
    xgb_test = xgb_loaded.predict_proba(X_test)[:, 1]

    cat_loaded = CatBoostClassifier()
    cat_loaded.load_model(str(MODELS_OUT / 'catboost_final.cbm'))
    cat_test = cat_loaded.predict_proba(X_test)[:, 1]

    ens_test = (lgbm_test * weights['lgbm'] +
                xgb_test * weights['xgb'] +
                cat_test * weights['catboost'])
    log(f'Predictions: min={ens_test.min():.4f}  max={ens_test.max():.4f}  mean={ens_test.mean():.4f}')

    # Build submission
    sample = pl.read_csv(DATA / 'sample_submit.csv')
    submit = pl.DataFrame({'event_id': event_ids, 'predict': ens_test})
    submit = sample.select('event_id').join(submit, on='event_id', how='left')

    assert len(submit) == len(sample), f"Submit size mismatch: {len(submit)} vs {len(sample)}"
    assert submit['event_id'].n_unique() == len(sample), "Duplicate event_ids in submit!"

    n_null = submit['predict'].is_null().sum()
    if n_null > 0:
        median_pred = submit['predict'].drop_nulls().median()
        submit = submit.with_columns(pl.col('predict').fill_null(median_pred))
        log(f'Filled {n_null} missing with median={median_pred:.4f}')

    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = SUBMIT_OUT / f'submit_v3_{ts}.csv'
    submit.write_csv(out_path)
    log(f'Saved: {out_path}')
    log(f'Rows: {len(submit):,} (unique event_ids: {submit["event_id"].n_unique():,})')

    # ══════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════
    print('\n' + '='*60)
    print('FINDINGS & SUMMARY')
    print('='*60)
    for f in findings:
        print(f'  * {f}')
    print(f'\n  Ensemble weights: {weights}')
    print(f'  Submit: {out_path}')
    print(f'  Total time: {(time.time()-t_start)/60:.1f} min')
    log('ALL DONE')


if __name__ == '__main__':
    main()
