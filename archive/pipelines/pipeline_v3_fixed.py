"""
Pipeline v3-fixed:
  - Фичи из v2 (которые дали 0.608 на val) — БЕЗ target encoding (оно обрушило модель)
  - ФИКС: дедупликация test event_ids (12,405 дублей)
  - Больше сидов для стабильности (multi-seed averaging)
  - 3-модельный ансамбль с оптимизированными весами
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
MODELS_OUT  = ROOT / 'models_v3f'
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


# ─── Feature Engineering (same as v2 — proven to work) ───

def add_v2_features(df: pl.DataFrame) -> pl.DataFrame:
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

    return df


FEATURE_COLS = [
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


def optimize_ensemble_weights(preds_list, y_true, names):
    n = len(preds_list)
    def neg_prauc(w):
        w = np.array(w)
        w = np.abs(w)  # Force positive weights
        w = w / w.sum()
        ens = sum(w[i] * preds_list[i] for i in range(n))
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
    log(f'Optimized weights: {dict(zip(names, best_w.round(4)))}')
    log(f'Optimized ensemble PR-AUC: {best_score:.4f}')
    return dict(zip(names, best_w.tolist()))


def main():
    t_start = time.time()

    print('\n' + '='*60)
    print('PIPELINE V3-FIXED')
    print('='*60)

    # Load data
    log('Loading train features...')
    df_dataset = pl.read_parquet(FEATURES_IN / 'train_features.parquet')
    df_dataset = add_v2_features(df_dataset)

    available_feats = [c for c in FEATURE_COLS if c in df_dataset.columns]
    log(f'Features: {len(available_feats)}')

    val_dt = datetime(2025, 4, 1)
    df_tr  = df_dataset.filter(pl.col('event_dttm') < val_dt)
    df_val = df_dataset.filter(pl.col('event_dttm') >= val_dt)

    log(f'Train: {len(df_tr):,}  Val: {len(df_val):,}')
    log(f'Val fraud: {df_val.filter(pl.col("target")==1).height:,}')

    def to_xy(d):
        X = d.select(available_feats).to_pandas()
        y = d['target'].to_numpy().astype(int)
        return X, y

    X_train, y_train = to_xy(df_tr)
    X_val, y_val     = to_xy(df_val)
    del df_tr, df_val; gc.collect()

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos = n_neg / n_pos
    log(f'scale_pos_weight={scale_pos:.2f}')

    # ═══════════════════════════════════════════
    # Multi-seed training для стабильности
    # ═══════════════════════════════════════════

    all_lgbm_preds = []
    all_xgb_preds = []
    all_cat_preds = []
    seeds = [42, 123, 777]

    for seed_idx, seed in enumerate(seeds):
        log(f'\n--- Seed {seed} ({seed_idx+1}/{len(seeds)}) ---')

        # LightGBM
        lgbm_model = lgb.LGBMClassifier(
            objective='binary', metric='average_precision',
            device='gpu', gpu_platform_id=0, gpu_device_id=0,
            scale_pos_weight=scale_pos,
            n_estimators=5000, learning_rate=0.03,
            num_leaves=255, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0, max_bin=255,
            random_state=seed, n_jobs=4, verbose=-1,
        )
        lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                       callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(100)])
        lgbm_preds = lgbm_model.predict_proba(X_val)[:, 1]
        lgbm_score = average_precision_score(y_val, lgbm_preds)
        log(f'LightGBM[{seed}]: PR-AUC={lgbm_score:.4f} (iter={lgbm_model.best_iteration_})')
        all_lgbm_preds.append(lgbm_preds)

        # Save best LightGBM for final
        if seed_idx == 0:
            lgbm_best_iter = lgbm_model.best_iteration_
            lgbm_model.booster_.save_model(str(MODELS_OUT / f'lgbm_seed{seed}.txt'))

        # XGBoost
        xgb_model = xgb.XGBClassifier(
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
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
        xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
        xgb_score = average_precision_score(y_val, xgb_preds)
        log(f'XGBoost[{seed}]: PR-AUC={xgb_score:.4f} (iter={xgb_model.best_iteration})')
        all_xgb_preds.append(xgb_preds)

        if seed_idx == 0:
            xgb_best_iter = xgb_model.best_iteration

        # CatBoost
        cat_model = CatBoostClassifier(
            iterations=5000, learning_rate=0.03, depth=8,
            task_type='GPU', devices='0',
            loss_function='Logloss', eval_metric='AUC',
            auto_class_weights='Balanced', l2_leaf_reg=3.0,
            random_seed=seed, verbose=100, early_stopping_rounds=200,
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        cat_preds = cat_model.predict_proba(X_val)[:, 1]
        cat_score = average_precision_score(y_val, cat_preds)
        log(f'CatBoost[{seed}]: PR-AUC={cat_score:.4f} (iter={cat_model.best_iteration_})')
        all_cat_preds.append(cat_preds)

        if seed_idx == 0:
            cat_best_iter = cat_model.best_iteration_

    # Average across seeds
    avg_lgbm = np.mean(all_lgbm_preds, axis=0)
    avg_xgb  = np.mean(all_xgb_preds, axis=0)
    avg_cat  = np.mean(all_cat_preds, axis=0)

    lgbm_avg_score = average_precision_score(y_val, avg_lgbm)
    xgb_avg_score  = average_precision_score(y_val, avg_xgb)
    cat_avg_score  = average_precision_score(y_val, avg_cat)
    log(f'\nAveraged scores: LGBM={lgbm_avg_score:.4f}  XGB={xgb_avg_score:.4f}  CAT={cat_avg_score:.4f}')

    # Optimize ensemble on averaged predictions
    log('Optimizing ensemble weights on averaged predictions...')
    weights = optimize_ensemble_weights(
        [avg_lgbm, avg_xgb, avg_cat], y_val,
        ['lgbm', 'xgb', 'catboost']
    )
    with open(MODELS_OUT / 'weights.json', 'w') as f:
        json.dump(weights, f, indent=2)

    del X_train, y_train, X_val, y_val; gc.collect()

    # ═══════════════════════════════════════════
    # Final models (all seeds) on ALL data
    # ═══════════════════════════════════════════
    log('\nTraining final models on all data...')
    X_full, y_full = to_xy(df_dataset)
    del df_dataset; gc.collect()

    final_lgbm_models = []
    final_xgb_models = []
    final_cat_models = []

    for seed_idx, seed in enumerate(seeds):
        log(f'Final training seed {seed}...')

        # LightGBM
        fl = lgb.LGBMClassifier(
            objective='binary', metric='average_precision',
            device='gpu', gpu_platform_id=0, gpu_device_id=0,
            scale_pos_weight=scale_pos,
            n_estimators=int(lgbm_best_iter * 1.1), learning_rate=0.03,
            num_leaves=255, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0, max_bin=255,
            random_state=seed, n_jobs=4, verbose=-1,
        )
        fl.fit(X_full, y_full)
        path = str(MODELS_OUT / f'lgbm_final_s{seed}.txt')
        fl.booster_.save_model(path)
        final_lgbm_models.append(path)

        # XGBoost
        fx = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='aucpr',
            tree_method='hist', device='cuda',
            scale_pos_weight=scale_pos,
            n_estimators=int(xgb_best_iter * 1.1), learning_rate=0.03,
            max_depth=8, min_child_weight=30,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0, max_bin=255,
            random_state=seed, n_jobs=4, verbosity=0,
        )
        fx.fit(X_full, y_full)
        path = str(MODELS_OUT / f'xgb_final_s{seed}.json')
        fx.save_model(path)
        final_xgb_models.append(path)

        # CatBoost
        fc = CatBoostClassifier(
            iterations=int(cat_best_iter * 1.1), learning_rate=0.03, depth=8,
            task_type='GPU', devices='0',
            loss_function='Logloss', auto_class_weights='Balanced',
            l2_leaf_reg=3.0, random_seed=seed, verbose=0,
        )
        fc.fit(X_full, y_full)
        path = str(MODELS_OUT / f'cat_final_s{seed}.cbm')
        fc.save_model(path)
        final_cat_models.append(path)

    del X_full, y_full; gc.collect()

    # Save model paths
    model_info = {
        'weights': weights,
        'lgbm_models': final_lgbm_models,
        'xgb_models': final_xgb_models,
        'cat_models': final_cat_models,
        'features': available_feats,
    }
    with open(MODELS_OUT / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    # ═══════════════════════════════════════════
    # Predict test (WITH DEDUP)
    # ═══════════════════════════════════════════
    log('Loading test features...')
    df_test = pl.read_parquet(FEATURES_IN / 'test_features.parquet')

    # CRITICAL: deduplicate
    n_before = len(df_test)
    df_test = df_test.unique(subset=['event_id'], keep='first')
    n_after = len(df_test)
    log(f'Dedup: {n_before} -> {n_after} (-{n_before - n_after})')

    df_test = add_v2_features(df_test)
    X_test = df_test.select(available_feats).to_pandas()
    event_ids = df_test['event_id'].to_numpy()
    log(f'X_test: {X_test.shape}')

    # Average predictions across all seeds
    lgbm_preds_test = []
    for mp in final_lgbm_models:
        b = lgb.Booster(model_file=mp)
        lgbm_preds_test.append(b.predict(X_test))
    avg_lgbm_test = np.mean(lgbm_preds_test, axis=0)

    xgb_preds_test = []
    for mp in final_xgb_models:
        m = xgb.XGBClassifier()
        m.load_model(mp)
        xgb_preds_test.append(m.predict_proba(X_test)[:, 1])
    avg_xgb_test = np.mean(xgb_preds_test, axis=0)

    cat_preds_test = []
    for mp in final_cat_models:
        m = CatBoostClassifier()
        m.load_model(mp)
        cat_preds_test.append(m.predict_proba(X_test)[:, 1])
    avg_cat_test = np.mean(cat_preds_test, axis=0)

    ens_test = (avg_lgbm_test * weights['lgbm'] +
                avg_xgb_test  * weights['xgb'] +
                avg_cat_test  * weights['catboost'])
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
        log(f'Filled {n_null} missing with median={median_pred:.4f}')

    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = SUBMIT_OUT / f'submit_v3f_{ts}.csv'
    submit.write_csv(out_path)
    log(f'Saved: {out_path}')
    log(f'Rows: {len(submit):,}')

    log(f'\nTotal time: {(time.time()-t_start)/60:.1f} min')
    log('ALL DONE')


if __name__ == '__main__':
    main()
