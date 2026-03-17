"""
Resume pipeline_v2: обучаем final модели (val-модели уже есть) и делаем submit.
Берём val-результаты из первого запуска:
  LightGBM: 0.6082
  XGBoost:  0.6050 (best ~iter 600)
  CatBoost: 0.5984
  Ensemble: 0.6112
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

# Re-use v2 pipeline components
from pipeline_v2 import (
    ROOT, DATA, FEATURES_IN, MODELS_OUT, SUBMIT_OUT,
    log, add_v2_features, FEATURE_COLS_V2,
    optimize_ensemble_weights,
)

def main():
    t_start = time.time()

    # 1. Load data + v2 features
    log('Loading training data...')
    df_dataset = pl.read_parquet(FEATURES_IN / 'train_features.parquet')
    df_dataset = add_v2_features(df_dataset)

    available_feats = [c for c in FEATURE_COLS_V2 if c in df_dataset.columns]
    log(f'Features: {len(available_feats)}')

    # Validation split for scoring
    val_dt = datetime(2025, 4, 1)
    df_tr  = df_dataset.filter(pl.col('event_dttm') < val_dt)
    df_val = df_dataset.filter(pl.col('event_dttm') >= val_dt)

    def to_xy(d):
        X = d.select(available_feats).to_pandas()
        y = d['target'].to_numpy().astype(int)
        return X, y

    X_train, y_train = to_xy(df_tr)
    X_val, y_val     = to_xy(df_val)

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos = n_neg / n_pos

    # 2. Re-train models with early stopping, get val predictions
    # ── LightGBM ──
    log('Training LightGBM...')
    lgbm_model = lgb.LGBMClassifier(
        objective='binary', metric='average_precision',
        device='gpu', gpu_platform_id=0, gpu_device_id=0,
        scale_pos_weight=scale_pos,
        n_estimators=5000, learning_rate=0.03,
        num_leaves=255, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=2.0, max_bin=255,
        random_state=42, n_jobs=4, verbose=-1,
    )
    lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                   callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(100)])
    lgbm_preds = lgbm_model.predict_proba(X_val)[:, 1]
    lgbm_score = average_precision_score(y_val, lgbm_preds)
    log(f'>>> LightGBM PR-AUC: {lgbm_score:.4f} (best_iter={lgbm_model.best_iteration_})')

    # ── XGBoost with early stopping ──
    log('Training XGBoost...')
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='aucpr',
        tree_method='hist', device='cuda',
        scale_pos_weight=scale_pos,
        n_estimators=5000, learning_rate=0.03,
        max_depth=8, min_child_weight=30,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=2.0, max_bin=255,
        random_state=42, n_jobs=4, verbosity=0,
        early_stopping_rounds=200,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
    xgb_score = average_precision_score(y_val, xgb_preds)
    log(f'>>> XGBoost PR-AUC: {xgb_score:.4f} (best_iter={xgb_model.best_iteration})')

    # ── CatBoost ──
    log('Training CatBoost...')
    cat_model = CatBoostClassifier(
        iterations=5000, learning_rate=0.03, depth=8,
        task_type='GPU', devices='0',
        loss_function='Logloss', eval_metric='AUC',
        auto_class_weights='Balanced', l2_leaf_reg=3.0,
        random_seed=42, verbose=100, early_stopping_rounds=200,
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    cat_preds = cat_model.predict_proba(X_val)[:, 1]
    cat_score = average_precision_score(y_val, cat_preds)
    log(f'>>> CatBoost PR-AUC: {cat_score:.4f} (best_iter={cat_model.best_iteration_})')

    # 3. Optimize ensemble
    log('Optimizing ensemble...')
    weights = optimize_ensemble_weights(
        [lgbm_preds, xgb_preds, cat_preds], y_val,
        ['lgbm', 'xgb', 'catboost']
    )
    with open(MODELS_OUT / 'weights_v2.json', 'w') as f:
        json.dump(weights, f, indent=2)

    del X_train, y_train, X_val, y_val, df_tr, df_val; gc.collect()

    # 4. Final models on ALL data
    X_full, y_full = to_xy(df_dataset)
    del df_dataset; gc.collect()

    log('Training final LightGBM...')
    lgbm_best = lgbm_model.best_iteration_
    final_lgbm = lgb.LGBMClassifier(
        objective='binary', metric='average_precision',
        device='gpu', gpu_platform_id=0, gpu_device_id=0,
        scale_pos_weight=scale_pos,
        n_estimators=int(lgbm_best * 1.1), learning_rate=0.03,
        num_leaves=255, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=2.0, max_bin=255,
        random_state=42, n_jobs=4, verbose=-1,
    )
    final_lgbm.fit(X_full, y_full)
    final_lgbm.booster_.save_model(str(MODELS_OUT / 'lgbm_final_v2.txt'))
    log(f'Saved lgbm_final_v2.txt (iters={int(lgbm_best*1.1)})')

    log('Training final XGBoost...')
    xgb_best = xgb_model.best_iteration
    final_xgb = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='aucpr',
        tree_method='hist', device='cuda',
        scale_pos_weight=scale_pos,
        n_estimators=int(xgb_best * 1.1), learning_rate=0.03,
        max_depth=8, min_child_weight=30,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=2.0, max_bin=255,
        random_state=42, n_jobs=4, verbosity=0,
    )
    final_xgb.fit(X_full, y_full)
    final_xgb.save_model(str(MODELS_OUT / 'xgb_final_v2.json'))
    log(f'Saved xgb_final_v2.json (iters={int(xgb_best*1.1)})')

    log('Training final CatBoost...')
    cat_best = cat_model.best_iteration_
    final_cat = CatBoostClassifier(
        iterations=int(cat_best * 1.1), learning_rate=0.03, depth=8,
        task_type='GPU', devices='0',
        loss_function='Logloss', auto_class_weights='Balanced',
        l2_leaf_reg=3.0, random_seed=42, verbose=100,
    )
    final_cat.fit(X_full, y_full)
    final_cat.save_model(str(MODELS_OUT / 'catboost_final_v2.cbm'))
    log(f'Saved catboost_final_v2.cbm (iters={int(cat_best*1.1)})')

    del X_full, y_full; gc.collect()

    # 5. Predict test
    log('Loading test features...')
    df_test = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    df_test = add_v2_features(df_test)
    X_test  = df_test.select(available_feats).to_pandas()
    event_ids = df_test['event_id'].to_numpy()
    log(f'X_test: {X_test.shape}')

    lgbm_booster = lgb.Booster(model_file=str(MODELS_OUT / 'lgbm_final_v2.txt'))
    lgbm_test = lgbm_booster.predict(X_test)

    xgb_final = xgb.XGBClassifier()
    xgb_final.load_model(str(MODELS_OUT / 'xgb_final_v2.json'))
    xgb_test = xgb_final.predict_proba(X_test)[:, 1]

    cat_final = CatBoostClassifier()
    cat_final.load_model(str(MODELS_OUT / 'catboost_final_v2.cbm'))
    cat_test = cat_final.predict_proba(X_test)[:, 1]

    ens_test = (lgbm_test * weights['lgbm'] +
                xgb_test  * weights['xgb'] +
                cat_test  * weights['catboost'])
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

    # Summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'LightGBM:  {lgbm_score:.4f} (best_iter={lgbm_best})')
    print(f'XGBoost:   {xgb_score:.4f} (best_iter={xgb_best})')
    print(f'CatBoost:  {cat_score:.4f} (best_iter={cat_best})')
    print(f'Ensemble:  {-1*min(-average_precision_score(y_val, sum(weights[n]*p for n,p in zip(["lgbm","xgb","catboost"],[lgbm_preds,xgb_preds,cat_preds])))):.4f}' if False else '')
    print(f'Weights:   {weights}')
    print(f'Submit:    {out_path}')

    log(f'Total time: {(time.time()-t_start)/60:.1f} min')
    log('ALL DONE')


if __name__ == '__main__':
    main()
