"""
Pipeline v2: улучшенные фичи + XGBoost + оптимизированный ансамбль.
Работает поверх уже посчитанных features/train_features.parquet и test_features.parquet.
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
MODELS_OUT  = ROOT / 'models_v2'
SUBMIT_OUT  = ROOT / 'submissions'

for d in [MODELS_OUT, SUBMIT_OUT]:
    d.mkdir(exist_ok=True)


# ─── helpers ─────────────────────────────────

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


# ─── Feature Engineering v2 ──────────────────

def add_v2_features(df: pl.DataFrame) -> pl.DataFrame:
    """Добавляем новые фичи поверх уже существующих колонок."""

    new_cols = []

    # 1. event_desc — уже есть, 113 unique, 0 null, просто используем как фичу
    # (уже в df)

    # 2. browser_language → binary (чаще null, но если 2 unique - одна из них маркер)
    if 'browser_language' in df.columns:
        new_cols.append(
            pl.col('browser_language').is_not_null().cast(pl.Int8).alias('has_browser_lang')
        )

    # 3. accept_language → наличие
    if 'accept_language' in df.columns:
        new_cols.append(
            pl.col('accept_language').is_not_null().cast(pl.Int8).alias('has_accept_lang')
        )

    # 4. screen_size → width, height (format "WxH")
    if 'screen_size' in df.columns:
        new_cols.extend([
            pl.col('screen_size').str.split('x').list.first()
              .cast(pl.Int32, strict=False).alias('screen_w'),
            pl.col('screen_size').str.split('x').list.last()
              .cast(pl.Int32, strict=False).alias('screen_h'),
        ])

    # 5. device_system_version → hash (104 unique)
    if 'device_system_version' in df.columns:
        new_cols.append(
            pl.col('device_system_version').is_not_null().cast(pl.Int8).alias('has_device_ver')
        )

    # 6. session_id → наличие сессии
    if 'session_id' in df.columns:
        new_cols.append(
            pl.col('session_id').is_not_null().cast(pl.Int8).alias('has_session')
        )

    if new_cols:
        df = df.with_columns(new_cols)

    # 7. Velocity ratios (safe division)
    ratio_cols = []

    # cnt ratios
    for short, long in [('1h','24h'), ('6h','24h'), ('24h','7d'), ('7d','30d'), ('1h','6h')]:
        cs, cl = f'cnt_{short}', f'cnt_{long}'
        if cs in df.columns and cl in df.columns:
            ratio_cols.append(
                (pl.col(cs) / (pl.col(cl) + 1)).alias(f'cnt_ratio_{short}_{long}')
            )

    # amt ratios
    for short, long in [('1h','24h'), ('24h','7d'), ('7d','30d')]:
        cs, cl = f'amt_sum_{short}', f'amt_sum_{long}'
        if cs in df.columns and cl in df.columns:
            ratio_cols.append(
                (pl.col(cs) / (pl.col(cl).abs() + 1)).alias(f'amt_ratio_{short}_{long}')
            )

    # current amount / rolling sum
    if 'operaton_amt' in df.columns:
        for w in ['24h', '7d', '30d']:
            c = f'amt_sum_{w}'
            if c in df.columns:
                ratio_cols.append(
                    (pl.col('operaton_amt') / (pl.col(c).abs() + 1)).alias(f'amt_cur_ratio_{w}')
                )

    if ratio_cols:
        df = df.with_columns(ratio_cols)

    # 8. Time features
    time_cols = []
    if 'hour' in df.columns and 'weekday' in df.columns:
        time_cols.append(
            (pl.col('hour') * 7 + pl.col('weekday')).alias('hour_weekday')
        )
    if 'hour' in df.columns:
        # Sine/cosine encoding of hour (cyclical)
        time_cols.extend([
            (np.pi * 2 * pl.col('hour').cast(pl.Float32) / 24).sin().alias('hour_sin'),
            (np.pi * 2 * pl.col('hour').cast(pl.Float32) / 24).cos().alias('hour_cos'),
        ])
    if 'event_dttm' in df.columns:
        time_cols.append(
            pl.col('event_dttm').dt.day().cast(pl.Int8).alias('day_of_month')
        )

    if time_cols:
        df = df.with_columns(time_cols)

    # 9. Amount features
    amt_cols = []
    if 'operaton_amt' in df.columns:
        amt_cols.extend([
            (pl.col('operaton_amt') > 0).cast(pl.Int8).alias('is_positive_amt'),
            pl.col('operaton_amt').log1p().alias('log_amount_v2'),
        ])
    if amt_cols:
        df = df.with_columns(amt_cols)

    # 10. Security risk score (weighted)
    if all(c in df.columns for c in ['compromised','web_rdp_connection','phone_voip_call_state','developer_tools']):
        df = df.with_columns([
            (pl.col('compromised') * 3 + pl.col('web_rdp_connection') * 2 +
             pl.col('phone_voip_call_state') * 2 + pl.col('developer_tools'))
              .alias('security_risk_score'),
        ])

    # 11. session_ops_before * operaton_amt
    if 'session_ops_before' in df.columns and 'operaton_amt' in df.columns:
        df = df.with_columns([
            (pl.col('session_ops_before') * pl.col('log_amount')).alias('session_ops_x_log_amt'),
        ])

    return df


# Feature columns for model
FEATURE_COLS_V2 = [
    # Original 41
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
    # New in v2
    'event_desc',
    'has_browser_lang', 'has_accept_lang',
    'screen_w', 'screen_h',
    'has_device_ver', 'has_session',
    'cnt_ratio_1h_24h', 'cnt_ratio_6h_24h', 'cnt_ratio_24h_7d',
    'cnt_ratio_7d_30d', 'cnt_ratio_1h_6h',
    'amt_ratio_1h_24h', 'amt_ratio_24h_7d', 'amt_ratio_7d_30d',
    'amt_cur_ratio_24h', 'amt_cur_ratio_7d', 'amt_cur_ratio_30d',
    'hour_weekday', 'hour_sin', 'hour_cos', 'day_of_month',
    'is_positive_amt',
    'security_risk_score',
    'session_ops_x_log_amt',
]


# ─── Training ───────────────────────────────

def rebuild_training_data(neg_ratio: int = 10):
    """Пересобираем train с другим ratio neg/pos из чанков."""
    tmp_dir = FEATURES_IN / '_tmp_train'
    chunk_files = sorted(tmp_dir.glob('chunk_*.parquet'))
    if not chunk_files:
        raise RuntimeError("No train chunks found. Run pipeline.py first.")

    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    train_start = datetime(2024, 10, 1)

    n_fraud = labels.filter(pl.col('target') == 1).height
    n_null_target = n_fraud * neg_ratio

    label_eids = set(labels['event_id'].to_list())
    total_null = 0
    for cf in chunk_files:
        n = pl.scan_parquet(cf).filter(
            pl.col('event_dttm') >= train_start
        ).select(pl.len()).collect().item()
        total_null += n
    total_null -= len(labels)
    frac = min(n_null_target / max(total_null, 1), 1.0)
    log(f'Rebuilding train: fraud={n_fraud:,}  target_neg={n_null_target:,}  frac={frac:.4f}')

    results = []
    for i, cf in enumerate(chunk_files):
        df = pl.read_parquet(cf)
        df = df.filter(pl.col('event_dttm') >= train_start)
        df = df.join(labels.select(['event_id', 'target']), on='event_id', how='left')

        labeled = df.filter(pl.col('target').is_not_null())
        df_null = df.filter(pl.col('target').is_null())

        n_sample = max(1, round(len(df_null) * frac))
        null_sampled = df_null.sample(n=min(n_sample, len(df_null)), seed=42 + i)
        null_sampled = null_sampled.with_columns(pl.lit(0).cast(pl.Int32).alias('target'))

        results.append(pl.concat([labeled, null_sampled]))
        del df, labeled, df_null, null_sampled; gc.collect()

    df_train = pl.concat(results)
    del results; gc.collect()
    log(f'Train rows: {len(df_train):,}')
    log(f'Target dist:\n{df_train["target"].value_counts().sort("target")}')
    return df_train


def optimize_ensemble_weights(preds_list, y_true, names):
    """Оптимизируем веса ансамбля через scipy minimize."""
    n = len(preds_list)

    def neg_prauc(w):
        w = np.array(w)
        w = w / w.sum()
        ens = sum(w[i] * preds_list[i] for i in range(n))
        return -average_precision_score(y_true, ens)

    from scipy.optimize import minimize
    best_score = -1
    best_w = None
    # Multi-start optimization
    for seed in range(20):
        rng = np.random.RandomState(seed)
        w0 = rng.dirichlet(np.ones(n))
        res = minimize(neg_prauc, w0, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-6})
        if -res.fun > best_score:
            best_score = -res.fun
            best_w = np.array(res.x)

    best_w = best_w / best_w.sum()
    log(f'Optimized weights: {dict(zip(names, best_w.round(4)))}')
    log(f'Optimized ensemble PR-AUC: {best_score:.4f}')
    return dict(zip(names, best_w.tolist()))


def train_models_v2():
    # Rebuild training data with more negatives
    log('Loading/rebuilding training data...')
    train_feat_path = FEATURES_IN / 'train_features.parquet'
    df_dataset = pl.read_parquet(train_feat_path)

    # Add v2 features
    log('Adding v2 features...')
    df_dataset = add_v2_features(df_dataset)

    n_fraud = df_dataset.filter(pl.col('target') == 1).height
    n_neg   = df_dataset.filter(pl.col('target') == 0).height
    log(f'Dataset: {len(df_dataset):,}  fraud={n_fraud:,}  neg={n_neg:,}')

    # Validation split — temporal
    val_dt = datetime(2025, 4, 1)
    df_tr  = df_dataset.filter(pl.col('event_dttm') < val_dt)
    df_val = df_dataset.filter(pl.col('event_dttm') >= val_dt)
    log(f'Train split: {len(df_tr):,}  |  Val: {len(df_val):,}')
    log(f'Val fraud: {df_val.filter(pl.col("target")==1).height}')

    # Проверяем какие фичи реально есть
    available_feats = [c for c in FEATURE_COLS_V2 if c in df_dataset.columns]
    missing = set(FEATURE_COLS_V2) - set(available_feats)
    if missing:
        log(f'Warning: missing features (skipped): {missing}')

    def to_xy(d):
        X = d.select(available_feats).to_pandas()
        y = d['target'].to_numpy().astype(int)
        return X, y

    X_train, y_train = to_xy(df_tr)
    X_val,   y_val   = to_xy(df_val)
    del df_tr, df_val; gc.collect()
    log(f'Features: {X_train.shape[1]}')

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos = n_neg / n_pos
    log(f'scale_pos_weight={scale_pos:.2f}')

    # ── LightGBM GPU ──
    log('Training LightGBM GPU...')
    lgbm_model = lgb.LGBMClassifier(
        objective='binary',
        metric='average_precision',
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        scale_pos_weight=scale_pos,
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=255,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.3,
        reg_lambda=2.0,
        max_bin=255,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(200, verbose=True),
            lgb.log_evaluation(100),
        ],
    )
    lgbm_preds = lgbm_model.predict_proba(X_val)[:, 1]
    lgbm_score = average_precision_score(y_val, lgbm_preds)
    log(f'>>> LightGBM Val PR-AUC: {lgbm_score:.4f}')
    lgbm_model.booster_.save_model(str(MODELS_OUT / 'lgbm_v2.txt'))

    # ── XGBoost GPU ──
    log('Training XGBoost GPU...')
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        tree_method='hist',
        device='cuda',
        scale_pos_weight=scale_pos,
        n_estimators=5000,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=30,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.3,
        reg_lambda=2.0,
        max_bin=255,
        random_state=42,
        n_jobs=4,
        verbosity=0,
        early_stopping_rounds=200,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
    xgb_score = average_precision_score(y_val, xgb_preds)
    log(f'>>> XGBoost Val PR-AUC: {xgb_score:.4f}')
    xgb_model.save_model(str(MODELS_OUT / 'xgb_v2.json'))

    # ── CatBoost GPU ──
    log('Training CatBoost GPU...')
    cat_model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.03,
        depth=8,
        task_type='GPU',
        devices='0',
        loss_function='Logloss',
        eval_metric='AUC',
        auto_class_weights='Balanced',
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=200,
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    cat_preds = cat_model.predict_proba(X_val)[:, 1]
    cat_score = average_precision_score(y_val, cat_preds)
    log(f'>>> CatBoost Val PR-AUC: {cat_score:.4f}')
    cat_model.save_model(str(MODELS_OUT / 'catboost_v2.cbm'))

    # ── Optimized Ensemble ──
    log('Optimizing ensemble weights...')
    weights = optimize_ensemble_weights(
        [lgbm_preds, xgb_preds, cat_preds],
        y_val,
        ['lgbm', 'xgb', 'catboost']
    )
    with open(MODELS_OUT / 'weights_v2.json', 'w') as f:
        json.dump(weights, f, indent=2)

    # ── Final models on ALL data ──
    log('Training final LightGBM on all data...')
    X_full, y_full = to_xy(df_dataset)
    best_iter = lgbm_model.best_iteration_
    final_lgbm = lgb.LGBMClassifier(
        objective='binary',
        metric='average_precision',
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        scale_pos_weight=scale_pos,
        n_estimators=int(best_iter * 1.1),
        learning_rate=0.03,
        num_leaves=255,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.3,
        reg_lambda=2.0,
        max_bin=255,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    final_lgbm.fit(X_full, y_full)
    final_lgbm.booster_.save_model(str(MODELS_OUT / 'lgbm_final_v2.txt'))
    log('Saved lgbm_final_v2.txt')

    log('Training final XGBoost on all data...')
    xgb_best_iter = xgb_model.best_iteration if xgb_model.best_iteration else 600
    final_xgb = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        tree_method='hist',
        device='cuda',
        scale_pos_weight=scale_pos,
        n_estimators=int(xgb_best_iter * 1.1),
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=30,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.3,
        reg_lambda=2.0,
        max_bin=255,
        random_state=42,
        n_jobs=4,
        verbosity=0,
    )
    final_xgb.fit(X_full, y_full)
    final_xgb.save_model(str(MODELS_OUT / 'xgb_final_v2.json'))
    log('Saved xgb_final_v2.json')

    log('Training final CatBoost on all data...')
    cat_best_iter = cat_model.best_iteration_
    final_cat = CatBoostClassifier(
        iterations=int(cat_best_iter * 1.1),
        learning_rate=0.03,
        depth=8,
        task_type='GPU',
        devices='0',
        loss_function='Logloss',
        auto_class_weights='Balanced',
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=100,
    )
    final_cat.fit(X_full, y_full)
    final_cat.save_model(str(MODELS_OUT / 'catboost_final_v2.cbm'))
    log('Saved catboost_final_v2.cbm')

    del X_full, y_full, df_dataset; gc.collect()

    return weights, available_feats


# ─── Predict & Submit ────────────────────────

def predict_and_submit(weights, available_feats):
    test_feat_path = FEATURES_IN / 'test_features.parquet'
    log('Loading test features...')
    df_test = pl.read_parquet(test_feat_path)
    df_test = add_v2_features(df_test)
    X_test  = df_test.select(available_feats).to_pandas()
    event_ids = df_test['event_id'].to_numpy()
    log(f'X_test: {X_test.shape}')

    # Load final models
    lgbm_booster = lgb.Booster(model_file=str(MODELS_OUT / 'lgbm_final_v2.txt'))
    lgbm_test = lgbm_booster.predict(X_test)

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(MODELS_OUT / 'xgb_final_v2.json'))
    xgb_test = xgb_model.predict_proba(X_test)[:, 1]

    cat_model = CatBoostClassifier()
    cat_model.load_model(str(MODELS_OUT / 'catboost_final_v2.cbm'))
    cat_test = cat_model.predict_proba(X_test)[:, 1]

    ens_test = (lgbm_test * weights['lgbm'] +
                xgb_test * weights['xgb'] +
                cat_test * weights['catboost'])
    log(f'Predictions: min={ens_test.min():.4f}  max={ens_test.max():.4f}  mean={ens_test.mean():.4f}')

    sample = pl.read_csv(DATA / 'sample_submit.csv')
    submit = pl.DataFrame({'event_id': event_ids, 'predict': ens_test})
    submit = sample.select('event_id').join(submit, on='event_id', how='left')

    n_null = submit['predict'].is_null().sum()
    if n_null > 0:
        median_pred = submit['predict'].drop_nulls().median()
        submit = submit.with_columns(pl.col('predict').fill_null(median_pred))
        log(f'Filled {n_null} missing with median')

    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = SUBMIT_OUT / f'submit_v2_{ts}.csv'
    submit.write_csv(out_path)
    log(f'Saved: {out_path}')
    log(f'Rows: {len(submit):,}')
    return out_path


# ─── MAIN ────────────────────────────────────

if __name__ == '__main__':
    t_start = time.time()

    print('\n' + '='*60)
    print('PIPELINE V2: Enhanced features + 3-model ensemble')
    print('='*60)

    weights, available_feats = train_models_v2()

    print('\n' + '='*60)
    print('PREDICT & SUBMIT')
    print('='*60)
    predict_and_submit(weights, available_feats)

    log(f'Total time: {(time.time()-t_start)/60:.1f} min')
    log('ALL DONE')
