"""
Adversarial Validation v2: проверить AUC после удаления/нормализации опасных фич.
Цель: найти минимальный набор фич для удаления, чтобы AUC < 0.6.
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path
from datetime import datetime

ROOT = Path('/home/vadim/PyPr/hak')
DATA = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'

def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

import importlib.util
spec = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
v9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9)

def adversarial_auc(X_train, X_test, feats):
    """Quick 3-fold adversarial AUC."""
    X = pd.concat([X_train[feats], X_test[feats]], ignore_index=True)
    y = np.array([0]*len(X_train) + [1]*len(X_test))

    params = {
        'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
        'n_jobs': -1, 'learning_rate': 0.05, 'num_leaves': 31,
        'min_child_samples': 100, 'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 42,
    }

    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in kf.split(X, y):
        dtrain = lgb.Dataset(X.iloc[train_idx], y[train_idx])
        dval = lgb.Dataset(X.iloc[val_idx], y[val_idx])
        model = lgb.train(params, dtrain, num_boost_round=300,
                          valid_sets=[dval], callbacks=[lgb.early_stopping(30, verbose=False)])
        preds = model.predict(X.iloc[val_idx])
        aucs.append(roc_auc_score(y[val_idx], preds))
    return np.mean(aucs)


if __name__ == '__main__':
    log('=== ADVERSARIAL V2: ПОИСК ОПАСНЫХ ФИЧ ===')

    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')

    train_df = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    train_df = v9.add_features(train_df)
    train_df = v9.add_customer_profiles(train_df, profiles)

    test_df = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    test_df = test_df.unique(subset=['event_id'], keep='first')
    test_df = v9.add_features(test_df)
    test_df = v9.add_customer_profiles(test_df, profiles)

    all_feats = [c for c in v9.FEATURE_COLS if c in train_df.columns and c in test_df.columns]

    n_test = len(test_df)
    train_sample = train_df.sample(n=min(len(train_df), n_test*2), seed=42)

    X_train = train_sample.select(all_feats).to_pandas().astype(np.float32)
    X_test = test_df.select(all_feats).to_pandas().astype(np.float32)

    log(f'Baseline: {len(all_feats)} фич')

    # === Тест 1: убрать dormancy_days ===
    feats_1 = [f for f in all_feats if f != 'dormancy_days']
    auc_1 = adversarial_auc(X_train, X_test, feats_1)
    log(f'Без dormancy_days ({len(feats_1)} фич): AUC={auc_1:.4f}')

    # === Тест 2: убрать dormancy + month ===
    feats_2 = [f for f in all_feats if f not in ('dormancy_days', 'month')]
    auc_2 = adversarial_auc(X_train, X_test, feats_2)
    log(f'Без dormancy+month ({len(feats_2)} фич): AUC={auc_2:.4f}')

    # === Тест 3: убрать топ-5 опасных ===
    drop_5 = {'dormancy_days', 'month', 'weekday', 'session_ops_before', 'cust_tenure_days'}
    feats_3 = [f for f in all_feats if f not in drop_5]
    auc_3 = adversarial_auc(X_train, X_test, feats_3)
    log(f'Без топ-5 опасных ({len(feats_3)} фич): AUC={auc_3:.4f}')

    # === Тест 4: убрать все temporal + session фичи ===
    temporal = {'dormancy_days', 'month', 'weekday', 'day_of_month', 'hour',
                'is_night', 'is_weekend', 'session_ops_before', 'session_amt_before',
                'cust_tenure_days', 'cust_n_tx'}
    feats_4 = [f for f in all_feats if f not in temporal]
    auc_4 = adversarial_auc(X_train, X_test, feats_4)
    log(f'Без temporal+session ({len(feats_4)} фич): AUC={auc_4:.4f}')

    # === Тест 5: нормализация вместо удаления ===
    # Заменить dormancy_days на percentile rank
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()
    from scipy.stats import rankdata
    combined_dorm = np.concatenate([X_train_norm['dormancy_days'].values,
                                     X_test_norm['dormancy_days'].values])
    ranks = rankdata(combined_dorm) / len(combined_dorm)
    X_train_norm['dormancy_days'] = ranks[:len(X_train_norm)]
    X_test_norm['dormancy_days'] = ranks[len(X_train_norm):]

    # Также нормализовать month
    combined_month = np.concatenate([X_train_norm['month'].values,
                                      X_test_norm['month'].values])
    ranks_m = rankdata(combined_month) / len(combined_month)
    X_train_norm['month'] = ranks_m[:len(X_train_norm)]
    X_test_norm['month'] = ranks_m[len(X_train_norm):]

    auc_5 = adversarial_auc(X_train_norm, X_test_norm, all_feats)
    log(f'С нормализацией dormancy+month ({len(all_feats)} фич): AUC={auc_5:.4f}')

    # === Тест 6: нормализация ВСЕХ опасных ===
    X_train_norm2 = X_train.copy()
    X_test_norm2 = X_test.copy()
    for feat in ['dormancy_days', 'session_ops_before', 'session_amt_before',
                 'cust_tenure_days', 'cust_n_tx', 'cum_unique_mcc_approx',
                 'cnt_30d', 'cnt_6h', 'cnt_24h', 'cnt_1h']:
        if feat in all_feats:
            combined = np.concatenate([X_train_norm2[feat].values, X_test_norm2[feat].values])
            r = rankdata(combined) / len(combined)
            X_train_norm2[feat] = r[:len(X_train_norm2)]
            X_test_norm2[feat] = r[len(X_train_norm2):]

    # Удалить month (категориальная, нормализация бессмысленна)
    feats_6 = [f for f in all_feats if f != 'month']
    auc_6 = adversarial_auc(X_train_norm2, X_test_norm2, feats_6)
    log(f'Нормализация 10 фич + без month ({len(feats_6)} фич): AUC={auc_6:.4f}')

    # === ИТОГИ ===
    log(f'\n=== ИТОГИ ===')
    log(f'Baseline (91 фич):                    AUC=1.0000')
    log(f'Без dormancy_days:                     AUC={auc_1:.4f}')
    log(f'Без dormancy+month:                    AUC={auc_2:.4f}')
    log(f'Без топ-5 опасных:                     AUC={auc_3:.4f}')
    log(f'Без всех temporal+session:             AUC={auc_4:.4f}')
    log(f'Нормализация dormancy+month:           AUC={auc_5:.4f}')
    log(f'Нормализация 10 фич + без month:       AUC={auc_6:.4f}')
    log(f'\nЦель: AUC < 0.60 (минимальный shift)')

    log('\nDONE')
