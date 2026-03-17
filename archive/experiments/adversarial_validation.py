"""
Adversarial Validation: обучить модель отличать train от test.
Если может — значит distribution shift, и мы узнаем КАКИЕ фичи сдвинуты.
Это объяснит почему val≠LB.
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

if __name__ == '__main__':
    log('=== ADVERSARIAL VALIDATION ===')
    log('Цель: отличить train от test → найти distribution shift')

    # Загрузить train features
    log('Загрузка train features...')
    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    train_df = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    train_df = v9.add_features(train_df)
    train_df = v9.add_customer_profiles(train_df, profiles)

    # Загрузить test features
    log('Загрузка test features...')
    test_df = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    test_df = test_df.unique(subset=['event_id'], keep='first')
    test_df = v9.add_features(test_df)
    test_df = v9.add_customer_profiles(test_df, profiles)

    feats = [c for c in v9.FEATURE_COLS if c in train_df.columns and c in test_df.columns]
    log(f'Фичи: {len(feats)}')

    # Сэмплировать train до размера test (для баланса)
    n_test = len(test_df)
    n_train_sample = min(len(train_df), n_test * 2)
    train_sample = train_df.sample(n=n_train_sample, seed=42)
    log(f'Train sample: {n_train_sample:,}, Test: {n_test:,}')

    # Создать adversarial dataset
    X_train = train_sample.select(feats).to_pandas().astype(np.float32)
    X_test = test_df.select(feats).to_pandas().astype(np.float32)

    X = pd.concat([X_train, X_test], ignore_index=True)
    y = np.array([0] * len(X_train) + [1] * len(X_test))

    log(f'Adversarial dataset: {len(X):,} rows, {len(feats)} features')
    log(f'  Class 0 (train): {(y==0).sum():,}')
    log(f'  Class 1 (test): {(y==1).sum():,}')

    # === 1. Сравнение распределений фич ===
    log('\n=== СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ ===')
    shifts = []
    for feat in feats:
        t_mean = X_train[feat].mean()
        s_mean = X_test[feat].mean()
        t_std = X_train[feat].std()
        s_std = X_test[feat].std()

        # KS-подобная метрика
        if t_std > 0:
            shift = abs(t_mean - s_mean) / t_std
        else:
            shift = 0.0

        t_null = X_train[feat].isna().mean()
        s_null = X_test[feat].isna().mean()

        shifts.append({
            'feature': feat,
            'train_mean': t_mean,
            'test_mean': s_mean,
            'shift_std': shift,
            'train_null%': t_null * 100,
            'test_null%': s_null * 100,
            'null_diff': abs(t_null - s_null) * 100,
        })

    shifts_df = pd.DataFrame(shifts).sort_values('shift_std', ascending=False)
    log('\nТоп-20 сдвинутых фич (по стандартным отклонениям):')
    for _, row in shifts_df.head(20).iterrows():
        log(f'  {row["feature"]:35s}: shift={row["shift_std"]:.3f}σ | '
            f'train={row["train_mean"]:.4f} test={row["test_mean"]:.4f} | '
            f'null: {row["train_null%"]:.1f}% → {row["test_null%"]:.1f}%')

    # Сохранить полную таблицу
    shifts_df.to_csv(ROOT / 'features' / 'adversarial_shifts.csv', index=False)
    log(f'\nПолная таблица: features/adversarial_shifts.csv')

    # === 2. Adversarial LGBM (5-fold CV) ===
    log('\n=== ADVERSARIAL LGBM ===')
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': -1,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    importances = np.zeros(len(feats))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        dtrain = lgb.Dataset(X.iloc[train_idx], y[train_idx])
        dval = lgb.Dataset(X.iloc[val_idx], y[val_idx])

        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        preds = model.predict(X.iloc[val_idx])
        auc = roc_auc_score(y[val_idx], preds)
        aucs.append(auc)
        importances += model.feature_importance(importance_type='gain')
        log(f'  Fold {fold+1}: AUC={auc:.4f} (iters={model.best_iteration})')

    mean_auc = np.mean(aucs)
    log(f'\n  Mean AUC: {mean_auc:.4f} ± {np.std(aucs):.4f}')

    if mean_auc > 0.55:
        log(f'  ⚠️  AUC={mean_auc:.4f} > 0.55 → ЕСТЬ distribution shift!')
    elif mean_auc > 0.52:
        log(f'  ⚠️  AUC={mean_auc:.4f} → небольшой shift')
    else:
        log(f'  ✓ AUC={mean_auc:.4f} → нет значимого shift')

    # Feature importance для adversarial модели
    importances /= 5
    feat_imp = pd.DataFrame({
        'feature': feats,
        'adversarial_importance': importances
    }).sort_values('adversarial_importance', ascending=False)

    log('\nТоп-20 фич, различающих train от test:')
    for i, (_, row) in enumerate(feat_imp.head(20).iterrows()):
        log(f'  {i+1:2d}. {row["feature"]:35s}: importance={row["adversarial_importance"]:.1f}')

    feat_imp.to_csv(ROOT / 'features' / 'adversarial_importance.csv', index=False)

    # === 3. Перекрытие клиентов ===
    log('\n=== ПЕРЕКРЫТИЕ КЛИЕНТОВ ===')
    train_custs = set(train_df['customer_id'].unique().to_list())
    test_custs = set(test_df['customer_id'].unique().to_list())
    overlap = train_custs & test_custs
    test_only = test_custs - train_custs
    train_only = train_custs - test_custs

    log(f'Train клиентов: {len(train_custs):,}')
    log(f'Test клиентов: {len(test_custs):,}')
    log(f'Пересечение: {len(overlap):,} ({100*len(overlap)/len(test_custs):.1f}% test)')
    log(f'Только в test: {len(test_only):,} ({100*len(test_only)/len(test_custs):.1f}% test)')
    log(f'Только в train: {len(train_only):,}')

    # === 4. Временной анализ ===
    log('\n=== ВРЕМЕННОЙ АНАЛИЗ ===')
    if 'hour' in train_df.columns:
        log(f'Train hour distribution: mean={train_df["hour"].mean():.2f}')
        log(f'Test hour distribution: mean={test_df["hour"].mean():.2f}')

    # === 5. Рекомендации ===
    log('\n=== РЕКОМЕНДАЦИИ ===')

    # Фичи с высоким shift И высокой adversarial importance = опасные
    merged = shifts_df.merge(feat_imp, on='feature')
    merged['danger_score'] = merged['shift_std'] * merged['adversarial_importance']
    merged = merged.sort_values('danger_score', ascending=False)

    log('\nТоп-10 ОПАСНЫХ фич (сдвиг × importance):')
    for i, (_, row) in enumerate(merged.head(10).iterrows()):
        log(f'  {i+1:2d}. {row["feature"]:35s}: danger={row["danger_score"]:.1f} '
            f'(shift={row["shift_std"]:.2f}σ, imp={row["adversarial_importance"]:.0f})')

    log('\nЕсли AUC > 0.6: рекомендуется УДАЛИТЬ топ опасных фич и пересоревноваться.')
    log('Если AUC < 0.55: shift минимальный, проблема в другом.')

    merged.to_csv(ROOT / 'features' / 'adversarial_danger.csv', index=False)
    log('\nDONE')
