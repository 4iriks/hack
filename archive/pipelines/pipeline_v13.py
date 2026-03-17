"""
Pipeline v13: Исправление distribution shift (adversarial validation).

Ключевая находка: adversarial AUC = 1.0 — train и test идеально различимы.
Главные утечки: dormancy_days (2.13σ shift), month, session_ops_before, cust_tenure_days.

Эксперименты:
A: Убрать dormancy_days + month (топ-2 leakers)
B: Убрать топ-5 leakers
C: v10 с 30K деревьев (seed 777 упирался в 10K cap)
D: Clip dormancy_days вместо удаления
E: Комбинация: remove leakers + больше деревьев + сильнее регуляризация
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from pathlib import Path
from datetime import datetime
import gc, json, time, os

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_OUT  = ROOT / 'models_v13'
SUBMIT_OUT  = ROOT / 'submissions'

for d in [MODELS_OUT, SUBMIT_OUT]:
    d.mkdir(exist_ok=True)

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

import importlib.util
spec = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
v9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9)

seeds = [42, 123, 777, 2024, 31337]

# v10 baseline params
LGBM_BASE = dict(
    objective='binary', metric='average_precision',
    device='gpu', gpu_platform_id=0, gpu_device_id=0,
    learning_rate=0.02,
    num_leaves=127, min_child_samples=200,
    subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
    n_jobs=4, verbose=-1,
)

# Фичи-утечки по adversarial validation
LEAK_TOP2 = {'dormancy_days', 'month'}
LEAK_TOP5 = {'dormancy_days', 'month', 'weekday', 'session_ops_before', 'cust_tenure_days'}


def load_data():
    """Загрузить train, val, профили."""
    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())

    # Val
    val_df = pl.read_parquet(FEATURES_IN / 'val_proper.parquet')
    val_df = val_df.with_columns(
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'))
    val_df = v9.add_features(val_df)
    val_df = v9.add_customer_profiles(val_df, profiles)

    # Train (minus val overlap)
    val_ids = set(pl.read_parquet(FEATURES_IN / 'val_event_ids.parquet')['event_id'].to_list())
    train_df = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    train_df = train_df.filter(~pl.col('event_id').is_in(val_ids))
    train_df = v9.add_features(train_df)
    train_df = v9.add_customer_profiles(train_df, profiles)

    all_feats = [c for c in v9.FEATURE_COLS if c in train_df.columns and c in val_df.columns]

    return train_df, val_df, all_feats, profiles


def train_experiment(name, X_train, y_train, X_val, y_val, params, n_trees=10000, patience=300):
    """Обучить 5-seed ensemble, вернуть val PR-AUC."""
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    spw = min(spw, 50.0)  # GPU cap

    log(f'\n{"="*60}')
    log(f'Experiment: {name}')
    log(f'Features: {X_train.shape[1]}, Trees: {n_trees}, spw: {spw:.1f}')
    log(f'{"="*60}')

    preds = np.zeros(len(X_val), dtype=np.float64)
    seed_results = []

    for i, seed in enumerate(seeds):
        m = lgb.LGBMClassifier(**params, random_state=seed,
            scale_pos_weight=spw, n_estimators=n_trees)
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)])
        best_iter = m.best_iteration_
        p = m.predict_proba(X_val)[:, 1]
        preds += p
        prauc = average_precision_score(y_val, p)
        seed_results.append(prauc)
        log(f'  Seed {seed}: iter={best_iter}, val={prauc:.6f}')

        # Сохранить модель
        model_path = MODELS_OUT / f'{name}_s{seed}.txt'
        m.booster_.save_model(str(model_path))
        del m; gc.collect()

    preds /= len(seeds)
    ensemble_prauc = average_precision_score(y_val, preds)
    log(f'  Ensemble ({name}): val={ensemble_prauc:.6f}')

    return ensemble_prauc, preds, seed_results


def generate_submission(name, feats, profiles, clip_dormancy=None):
    """Сгенерировать сабмит для эксперимента."""
    test_df = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    test_df = test_df.unique(subset=['event_id'], keep='first')
    test_df = v9.add_features(test_df)
    test_df = v9.add_customer_profiles(test_df, profiles)

    X_test = test_df.select(feats).to_pandas().astype(np.float32)
    if clip_dormancy and 'dormancy_days' in feats:
        X_test['dormancy_days'] = X_test['dormancy_days'].clip(upper=clip_dormancy)

    event_ids = test_df['event_id'].to_numpy()

    preds = np.mean([
        lgb.Booster(model_file=str(MODELS_OUT / f'{name}_s{s}.txt')).predict(X_test)
        for s in seeds], axis=0)

    sample = pl.read_csv(DATA / 'sample_submit.csv')
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    sub = pl.DataFrame({'event_id': event_ids, 'predict': preds.astype(np.float64)})
    sub = sample.select('event_id').join(sub, on='event_id', how='left')
    n_null = sub['predict'].is_null().sum()
    if n_null > 0:
        sub = sub.with_columns(pl.col('predict').fill_null(sub['predict'].drop_nulls().median()))

    path = SUBMIT_OUT / f'submit_v13_{name}_{ts}.csv'
    sub.write_csv(path)
    log(f'Saved: {path.name}')
    return path


if __name__ == '__main__':
    t_start = time.time()
    log('=== PIPELINE V13: FIX DISTRIBUTION SHIFT ===')

    train_df, val_df, all_feats, profiles = load_data()

    y_train = train_df['target'].to_numpy().astype(int)
    y_val = val_df['is_fraud'].to_numpy().astype(int)

    log(f'Train: {len(train_df):,}, Val: {len(val_df):,}')
    log(f'Val fraud: {y_val.sum()} ({100*y_val.mean():.4f}%)')
    log(f'All features: {len(all_feats)}')

    results = {}

    # ═══════════════════════════════════════
    # Exp A: Remove top-2 leakers (dormancy_days + month)
    # ═══════════════════════════════════════
    feats_a = [f for f in all_feats if f not in LEAK_TOP2]
    X_train_a = train_df.select(feats_a).to_pandas().astype(np.float32)
    X_val_a = val_df.select(feats_a).to_pandas().astype(np.float32)

    prauc_a, preds_a, seeds_a = train_experiment(
        'A_no_top2', X_train_a, y_train, X_val_a, y_val, LGBM_BASE)
    results['A_no_top2'] = prauc_a

    del X_train_a, X_val_a; gc.collect()

    # ═══════════════════════════════════════
    # Exp B: Remove top-5 leakers
    # ═══════════════════════════════════════
    feats_b = [f for f in all_feats if f not in LEAK_TOP5]
    X_train_b = train_df.select(feats_b).to_pandas().astype(np.float32)
    X_val_b = val_df.select(feats_b).to_pandas().astype(np.float32)

    prauc_b, preds_b, seeds_b = train_experiment(
        'B_no_top5', X_train_b, y_train, X_val_b, y_val, LGBM_BASE)
    results['B_no_top5'] = prauc_b

    del X_train_b, X_val_b; gc.collect()

    # ═══════════════════════════════════════
    # Exp C: All features but 30K trees (v10 had 10K cap)
    # ═══════════════════════════════════════
    X_train_c = train_df.select(all_feats).to_pandas().astype(np.float32)
    X_val_c = val_df.select(all_feats).to_pandas().astype(np.float32)

    prauc_c, preds_c, seeds_c = train_experiment(
        'C_30k_trees', X_train_c, y_train, X_val_c, y_val,
        LGBM_BASE, n_trees=30000, patience=500)
    results['C_30k_trees'] = prauc_c

    # ═══════════════════════════════════════
    # Exp D: Clip dormancy_days to 200 (reduce shift while keeping signal)
    # ═══════════════════════════════════════
    X_train_d = X_train_c.copy()
    X_val_d = X_val_c.copy()
    X_train_d['dormancy_days'] = X_train_d['dormancy_days'].clip(upper=200)
    X_val_d['dormancy_days'] = X_val_d['dormancy_days'].clip(upper=200)

    prauc_d, preds_d, seeds_d = train_experiment(
        'D_clip_dormancy', X_train_d, y_train, X_val_d, y_val, LGBM_BASE)
    results['D_clip_dormancy'] = prauc_d

    del X_train_d, X_val_d; gc.collect()

    # ═══════════════════════════════════════
    # Exp E: Remove top-2 + stronger reg + 30K trees
    # ═══════════════════════════════════════
    feats_e = [f for f in all_feats if f not in LEAK_TOP2]
    X_train_e = train_df.select(feats_e).to_pandas().astype(np.float32)
    X_val_e = val_df.select(feats_e).to_pandas().astype(np.float32)

    LGBM_REG = {**LGBM_BASE,
        'reg_alpha': 2.0, 'reg_lambda': 10.0,
        'num_leaves': 63, 'min_child_samples': 500,
        'colsample_bytree': 0.4}

    prauc_e, preds_e, seeds_e = train_experiment(
        'E_no_top2_reg', X_train_e, y_train, X_val_e, y_val,
        LGBM_REG, n_trees=30000, patience=500)
    results['E_no_top2_reg'] = prauc_e

    del X_train_e, X_val_e; gc.collect()

    # ═══════════════════════════════════════
    # Exp F: Remove top-5 + 30K trees
    # ═══════════════════════════════════════
    feats_f = [f for f in all_feats if f not in LEAK_TOP5]
    X_train_f = train_df.select(feats_f).to_pandas().astype(np.float32)
    X_val_f = val_df.select(feats_f).to_pandas().astype(np.float32)

    prauc_f, preds_f, seeds_f = train_experiment(
        'F_no_top5_30k', X_train_f, y_train, X_val_f, y_val,
        LGBM_BASE, n_trees=30000, patience=500)
    results['F_no_top5_30k'] = prauc_f

    del X_train_f, X_val_f, X_train_c, X_val_c; gc.collect()

    # ═══════════════════════════════════════
    # ИТОГИ
    # ═══════════════════════════════════════
    log(f'\n{"="*60}')
    log('ИТОГИ V13')
    log(f'{"="*60}')
    log(f'v10 baseline (reference):      val=0.039')

    sorted_res = sorted(results.items(), key=lambda x: -x[1])
    for name, prauc in sorted_res:
        delta = 100 * (prauc / 0.039 - 1)
        log(f'{name:30s}: val={prauc:.6f} ({delta:+.1f}% vs v10)')

    best_name = sorted_res[0][0]
    best_prauc = sorted_res[0][1]
    log(f'\nЛучший: {best_name} (val={best_prauc:.6f})')
    log(f'ВАЖНО: val может врать! Нужно проверить на LB.')

    # Сохранить результаты
    with open(MODELS_OUT / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Генерация сабмитов для лучших
    log('\n=== ГЕНЕРАЦИЯ САБМИТОВ ===')
    for name, prauc in sorted_res[:3]:
        feats_map = {
            'A_no_top2': [f for f in all_feats if f not in LEAK_TOP2],
            'B_no_top5': [f for f in all_feats if f not in LEAK_TOP5],
            'C_30k_trees': all_feats,
            'D_clip_dormancy': all_feats,
            'E_no_top2_reg': [f for f in all_feats if f not in LEAK_TOP2],
            'F_no_top5_30k': [f for f in all_feats if f not in LEAK_TOP5],
        }
        clip = 200 if name == 'D_clip_dormancy' else None
        generate_submission(name, feats_map[name], profiles, clip_dormancy=clip)

    total_min = (time.time() - t_start) / 60
    log(f'\nВсего: {total_min:.0f} мин')
    log('DONE')
