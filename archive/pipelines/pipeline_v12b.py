"""
Pipeline v12b: Hard Negatives на v10 train + Day-Level Stacking + Boost.

v12 (полные дни) ПРОВАЛИЛСЯ: val=0.020 vs baseline 0.039.
Причина: non-fraud транзакции в fraud-дне имеют те же фичи → модель путается.

Новая стратегия — НЕ менять формат train, а:
  A: Hard negative mining на v10 train (заменить часть green)
  B: Day-level стекинг (2я модель поверх v10 скоров)
  C: Customer-day boost (пост-процессинг)
  D: Battery фича + feature pruning
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from scipy.optimize import minimize_scalar, minimize
from pathlib import Path
from datetime import datetime
import gc, json, time, os

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
CHUNKS_DIR  = FEATURES_IN / '_tmp_train'
MODELS_V10  = ROOT / 'models_v10'
MODELS_OUT  = ROOT / 'models_v12b'
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

LGBM_PARAMS = dict(
    objective='binary', metric='average_precision',
    device='gpu', gpu_platform_id=0, gpu_device_id=0,
    learning_rate=0.02,
    num_leaves=127, min_child_samples=200,
    subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
    n_jobs=4, verbose=-1,
)


def load_v10_train_and_val():
    """Загрузить v10 train и val."""
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())
    val_id_set = set(pl.read_parquet(FEATURES_IN / 'val_event_ids.parquet')['event_id'].to_list())

    # Train (v10 assembled)
    train_df = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    train_df = train_df.filter(~pl.col('event_id').is_in(val_id_set))
    log(f'V10 train: {len(train_df):,} строк, fraud={train_df.filter(pl.col("target")==1).height:,}')

    # Val
    val_df = pl.read_parquet(FEATURES_IN / 'val_proper.parquet')
    val_df = val_df.with_columns(
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'))
    log(f'Val: {len(val_df):,} строк, fraud={val_df["is_fraud"].sum():,}')

    return train_df, val_df, labels


def prepare_features(df, profiles):
    """v9 feature engineering."""
    df = v9.add_features(df)
    df = v9.add_customer_profiles(df, profiles)
    return df


def get_v10_preds(X, model_dir=None):
    """Получить v10 предсказания."""
    if model_dir is None:
        model_dir = MODELS_V10
    return np.mean([
        lgb.Booster(model_file=str(model_dir / f'm1_lgbm_s{s}.txt')).predict(X)
        for s in seeds], axis=0)


# ═══════════════════════════════════════════════════════
# Эксперимент A: Hard Negative Mining на V10 TRAIN
# ═══════════════════════════════════════════════════════

def experiment_hard_negatives(train_df, val_df, profiles, top_pct=10):
    """
    Заменить часть random green в v10 train на hard negatives.
    Hard negatives = green строки с высоким v10 скором.
    """
    log(f'\n{"="*60}')
    log(f'ЭКСПЕРИМЕНТ A: Hard Negative Mining (top {top_pct}% green)')
    log(f'{"="*60}')

    train_feat = prepare_features(train_df.clone(), profiles)
    val_feat = prepare_features(val_df.clone(), profiles)

    available = [c for c in v9.FEATURE_COLS if c in train_feat.columns and c in val_feat.columns]
    log(f'Фичей: {len(available)}')

    # Скорим все green строки в train с помощью v10
    fraud_mask = train_feat['target'].to_numpy() == 1
    green_mask = ~fraud_mask

    X_green = train_feat.filter(pl.col('target') == 0).select(available).to_pandas().astype(np.float32)
    log(f'Скорим {len(X_green):,} green строк...')
    green_scores = get_v10_preds(X_green)

    # Top N% = hard negatives
    threshold = np.percentile(green_scores, 100 - top_pct)
    hard_mask = green_scores >= threshold
    n_hard = hard_mask.sum()
    log(f'Hard negatives: {n_hard:,} (порог={threshold:.4f})')

    # Новый train: все fraud + hard greens + sample easy greens
    fraud_df = train_feat.filter(pl.col('target') == 1)
    green_df = train_feat.filter(pl.col('target') == 0)

    hard_indices = np.where(hard_mask)[0]
    easy_indices = np.where(~hard_mask)[0]

    # Берём все hard + столько же easy (сохраняем баланс)
    n_easy = min(len(easy_indices), len(fraud_df) * 20)  # 20:1 easy
    np.random.seed(42)
    easy_sample = np.random.choice(easy_indices, size=n_easy, replace=False)

    all_green_idx = np.concatenate([hard_indices, easy_sample])
    all_green_idx.sort()

    green_selected = green_df[all_green_idx.tolist()]
    new_train = pl.concat([fraud_df, green_selected])

    X_tr = new_train.select(available).to_pandas().astype(np.float32)
    y_tr = new_train['target'].to_numpy().astype(int)

    X_val = val_feat.select(available).to_pandas().astype(np.float32)
    y_val = val_feat['is_fraud'].to_numpy().astype(int)

    spw = min((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 50.0)
    log(f'Новый train: {len(X_tr):,}, fraud={y_tr.sum():,}, spw={spw:.1f}')
    log(f'Hard: {n_hard:,}, Easy: {n_easy:,}')

    del train_feat, green_df, fraud_df, green_selected, new_train; gc.collect()

    # Обучение
    all_preds = np.zeros(len(X_val), dtype=np.float64)
    for seed in seeds:
        m = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed,
            scale_pos_weight=spw, n_estimators=50000)
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(0)])
        p = m.predict_proba(X_val)[:, 1]
        prauc = average_precision_score(y_val, p)
        log(f'  Seed {seed}: iter={m.best_iteration_}, val={prauc:.6f}')
        m.booster_.save_model(str(MODELS_OUT / f'hard_neg_s{seed}.txt'))
        all_preds += p
        del m; gc.collect()

    all_preds /= len(seeds)
    ensemble = average_precision_score(y_val, all_preds)
    log(f'Hard Neg Ensemble: val={ensemble:.6f}')

    del X_tr, X_green; gc.collect()
    return ensemble, all_preds, available, val_feat


# ═══════════════════════════════════════════════════════
# Эксперимент B: Day-Level стекинг поверх v10
# ═══════════════════════════════════════════════════════

def experiment_day_stacking(val_feat, base_preds, y_val):
    """
    Вторая модель на уровне (customer, day).
    Использует скоры базовой модели + day-level агрегаты.
    """
    log(f'\n{"="*60}')
    log(f'ЭКСПЕРИМЕНТ B: Day-Level Stacking')
    log(f'{"="*60}')

    customer_ids = val_feat['customer_id'].to_numpy()

    # Агрегируем скоры по (customer, day)
    val_with_scores = val_feat.with_columns([
        pl.Series('base_score', base_preds),
        pl.col('event_dttm').cast(pl.Date).alias('_day'),
    ])

    day_agg = val_with_scores.group_by(['customer_id', '_day']).agg([
        pl.col('base_score').max().alias('day_score_max'),
        pl.col('base_score').mean().alias('day_score_mean'),
        pl.col('base_score').std().alias('day_score_std'),
        pl.col('base_score').min().alias('day_score_min'),
        pl.col('base_score').median().alias('day_score_median'),
        (pl.col('base_score').max() - pl.col('base_score').min()).alias('day_score_range'),
        pl.len().alias('day_n_tx'),
    ])

    val_stacked = val_with_scores.join(day_agg, on=['customer_id', '_day'], how='left')

    results = {}

    # Метод 1: score × day_max^alpha
    log('\n--- Метод 1: score × day_max^alpha ---')
    day_max = val_stacked['day_score_max'].to_numpy()

    def neg_prauc_alpha(alpha):
        s = base_preds * (day_max ** alpha)
        return -average_precision_score(y_val, s)

    opt = minimize_scalar(neg_prauc_alpha, bounds=(0.05, 3.0), method='bounded')
    best_alpha = opt.x
    best_prauc = -opt.fun
    log(f'  score × day_max^{best_alpha:.3f}: val={best_prauc:.6f}')
    results['day_max_alpha'] = (best_prauc, best_alpha)

    # Метод 2: score × day_mean^alpha
    log('\n--- Метод 2: score × day_mean^alpha ---')
    day_mean = val_stacked['day_score_mean'].to_numpy()

    def neg_prauc_mean(alpha):
        s = base_preds * (day_mean ** alpha)
        return -average_precision_score(y_val, s)

    opt2 = minimize_scalar(neg_prauc_mean, bounds=(0.05, 3.0), method='bounded')
    log(f'  score × day_mean^{opt2.x:.3f}: val={-opt2.fun:.6f}')
    results['day_mean_alpha'] = (-opt2.fun, opt2.x)

    # Метод 3: multi-parameter: score^a × day_max^b × day_n_tx^c
    log('\n--- Метод 3: multi-parameter optimization ---')
    day_n = val_stacked['day_n_tx'].to_numpy().astype(float)

    def neg_prauc_multi(params):
        a, b, c = params
        s = (base_preds ** a) * (day_max ** b) * (day_n ** c)
        return -average_precision_score(y_val, s)

    from scipy.optimize import differential_evolution
    bounds_multi = [(0.5, 2.0), (0.0, 2.0), (-1.0, 1.0)]
    opt3 = differential_evolution(neg_prauc_multi, bounds_multi, seed=42, maxiter=100, tol=1e-6)
    a, b, c = opt3.x
    log(f'  score^{a:.2f} × day_max^{b:.2f} × n_tx^{c:.2f}: val={-opt3.fun:.6f}')
    results['multi_param'] = (-opt3.fun, opt3.x.tolist())

    return results


# ═══════════════════════════════════════════════════════
# Эксперимент C: Customer-Day Boost (оптимизированный)
# ═══════════════════════════════════════════════════════

def experiment_customer_boost(val_df, preds, y_val):
    """Оптимизированный customer-day boost."""
    log(f'\n{"="*60}')
    log(f'ЭКСПЕРИМЕНТ C: Customer-Day Boost')
    log(f'{"="*60}')

    customer_ids = val_df['customer_id'].to_numpy()
    unique_custs = np.unique(customer_ids)

    # Предвычислим customer max для скорости
    cust_max = np.zeros_like(preds)
    cust_mean = np.zeros_like(preds)
    for cid in unique_custs:
        mask = customer_ids == cid
        cust_preds = preds[mask]
        cust_max[mask] = cust_preds.max()
        cust_mean[mask] = cust_preds.mean()

    results = {}

    # 1. score × cust_max^alpha
    log('\n--- score × cust_max^alpha ---')
    def neg_prauc_max(alpha):
        return -average_precision_score(y_val, preds * (cust_max ** alpha))

    opt = minimize_scalar(neg_prauc_max, bounds=(0.05, 3.0), method='bounded')
    log(f'  cust_max^{opt.x:.3f}: val={-opt.fun:.6f}')
    results['cust_max'] = (-opt.fun, opt.x)

    # 2. score × cust_mean^alpha
    log('\n--- score × cust_mean^alpha ---')
    def neg_prauc_mean(alpha):
        return -average_precision_score(y_val, preds * (cust_mean ** alpha))

    opt2 = minimize_scalar(neg_prauc_mean, bounds=(0.05, 3.0), method='bounded')
    log(f'  cust_mean^{opt2.x:.3f}: val={-opt2.fun:.6f}')
    results['cust_mean'] = (-opt2.fun, opt2.x)

    # 3. Multi: score^a × cust_max^b × cust_mean^c
    log('\n--- multi-parameter ---')
    def neg_prauc_multi(params):
        a, b, c = params
        return -average_precision_score(y_val, (preds ** a) * (cust_max ** b) * (cust_mean ** c))

    from scipy.optimize import differential_evolution
    bounds = [(0.5, 2.0), (0.0, 2.0), (0.0, 2.0)]
    opt3 = differential_evolution(neg_prauc_multi, bounds, seed=42, maxiter=100, tol=1e-6)
    a, b, c = opt3.x
    log(f'  score^{a:.2f} × cust_max^{b:.2f} × cust_mean^{c:.2f}: val={-opt3.fun:.6f}')
    results['multi'] = (-opt3.fun, opt3.x.tolist())

    return results


# ═══════════════════════════════════════════════════════
# Эксперимент D: Battery + Feature Pruning
# ═══════════════════════════════════════════════════════

def experiment_battery_and_pruning(train_df, val_df, profiles):
    """Проверить battery фичу + убрать бесполезные фичи."""
    log(f'\n{"="*60}')
    log(f'ЭКСПЕРИМЕНТ D: Battery + Feature Pruning')
    log(f'{"="*60}')

    train_feat = prepare_features(train_df.clone(), profiles)
    val_feat = prepare_features(val_df.clone(), profiles)

    # Проверяем battery
    has_battery = 'battery' in train_feat.columns
    log(f'Battery в данных: {has_battery}')
    if has_battery:
        # Статистика battery
        bat_fraud = train_feat.filter(pl.col('target') == 1)['battery']
        bat_green = train_feat.filter(pl.col('target') == 0)['battery']
        log(f'  Battery fraud: null={bat_fraud.is_null().sum()}/{len(bat_fraud)}, '
            f'mean={bat_fraud.drop_nulls().mean():.1f}' if bat_fraud.drop_nulls().len() > 0 else '  all null')
        log(f'  Battery green: null={bat_green.is_null().sum()}/{len(bat_green)}, '
            f'mean={bat_green.drop_nulls().mean():.1f}' if bat_green.drop_nulls().len() > 0 else '  all null')

        # Добавляем battery фичи
        train_feat = train_feat.with_columns([
            pl.col('battery').fill_null(-1).alias('battery_val'),
            pl.col('battery').is_null().cast(pl.Int8).alias('battery_is_null'),
            (pl.col('battery') == 100).fill_null(False).cast(pl.Int8).alias('battery_is_100'),
        ])
        val_feat = val_feat.with_columns([
            pl.col('battery').fill_null(-1).alias('battery_val'),
            pl.col('battery').is_null().cast(pl.Int8).alias('battery_is_null'),
            (pl.col('battery') == 100).fill_null(False).cast(pl.Int8).alias('battery_is_100'),
        ])

    # Полный набор фичей (v10 + battery)
    all_feats = [c for c in v9.FEATURE_COLS if c in train_feat.columns and c in val_feat.columns]
    if has_battery:
        all_feats += ['battery_val', 'battery_is_null', 'battery_is_100']

    # Удалим bottom-20 бесполезных фичей (из feature importance v10)
    # Нижние 20 по gain: has_accept_lang, voip_x_logamt, has_browser_lang, etc.
    bottom_feats = {
        'has_accept_lang', 'voip_x_logamt', 'has_browser_lang', 'has_session',
        'has_device_ver', 'is_null_mcc', 'security_flags_sum', 'manual_entry_x_amt',
        'is_no_screen', 'is_no_timezone', 'has_pos_data', 'is_first_in_session',
        'is_manual_entry', 'is_fast_60s', 'log_cnt_30d', 'hourly_activity_ratio',
        'amt_pct_of_30d', 'is_high_risk_type', 'is_night', 'is_new_channel_indicator_type',
    }

    pruned_feats = [f for f in all_feats if f not in bottom_feats]
    log(f'Все фичи: {len(all_feats)}, после pruning: {len(pruned_feats)}')

    X_val = val_feat.select(all_feats).to_pandas().astype(np.float32)
    y_val = val_feat['is_fraud'].to_numpy().astype(int)

    results = {}

    # Тест 1: v10 train + battery (все фичи)
    if has_battery and len(all_feats) > len(v9.FEATURE_COLS):
        log('\n--- D1: + battery фичи ---')
        X_tr = train_feat.select(all_feats).to_pandas().astype(np.float32)
        y_tr = train_feat['target'].to_numpy().astype(int)
        spw = min((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 50.0)

        preds_d1 = np.zeros(len(X_val), dtype=np.float64)
        for seed in seeds:
            m = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed,
                scale_pos_weight=spw, n_estimators=50000)
            m.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(0)])
            p = m.predict_proba(X_val)[:, 1]
            prauc = average_precision_score(y_val, p)
            log(f'  Seed {seed}: iter={m.best_iteration_}, val={prauc:.6f}')
            preds_d1 += p
            del m; gc.collect()
        preds_d1 /= len(seeds)
        results['with_battery'] = average_precision_score(y_val, preds_d1)
        log(f'  D1 Ensemble: val={results["with_battery"]:.6f}')

    # Тест 2: pruned фичи (без bottom-20)
    log('\n--- D2: pruned фичи ---')
    X_tr_p = train_feat.select(pruned_feats).to_pandas().astype(np.float32)
    X_val_p = val_feat.select(pruned_feats).to_pandas().astype(np.float32)
    y_tr = train_feat['target'].to_numpy().astype(int)
    spw = min((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 50.0)

    preds_d2 = np.zeros(len(X_val_p), dtype=np.float64)
    for seed in seeds:
        m = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed,
            scale_pos_weight=spw, n_estimators=50000)
        m.fit(X_tr_p, y_tr,
              eval_set=[(X_val_p, y_val)],
              callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(0)])
        p = m.predict_proba(X_val_p)[:, 1]
        prauc = average_precision_score(y_val, p)
        log(f'  Seed {seed}: iter={m.best_iteration_}, val={prauc:.6f}')
        preds_d2 += p
        del m; gc.collect()
    preds_d2 /= len(seeds)
    results['pruned'] = average_precision_score(y_val, preds_d2)
    log(f'  D2 Ensemble: val={results["pruned"]:.6f}')

    return results


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    t_total = time.time()
    all_results = {}

    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    train_df, val_df, labels = load_v10_train_and_val()

    # ── Baseline v10 ──
    log('\n--- V10 Baseline ---')
    val_feat = prepare_features(val_df.clone(), profiles)
    v10_feats = [c for c in v9.FEATURE_COLS if c in val_feat.columns]
    X_val_base = val_feat.select(v10_feats).to_pandas().astype(np.float32)
    y_val = val_feat['is_fraud'].to_numpy().astype(int)
    v10_preds = get_v10_preds(X_val_base)
    v10_prauc = average_precision_score(y_val, v10_preds)
    log(f'V10 baseline: val={v10_prauc:.6f}')
    all_results['v10_baseline'] = v10_prauc
    del X_val_base; gc.collect()

    # ── Эксперимент A: Hard Negatives ──
    prauc_a, preds_a, feats_a, val_feat_a = experiment_hard_negatives(
        train_df, val_df, profiles, top_pct=10)
    all_results['A_hard_neg'] = prauc_a

    # Выбираем лучшую базовую модель
    if prauc_a > v10_prauc:
        best_preds = preds_a
        best_tag = 'hard_neg'
        log(f'\n>>> Hard negatives ЛУЧШЕ: {prauc_a:.6f} > {v10_prauc:.6f}')
    else:
        best_preds = v10_preds
        best_tag = 'v10'
        log(f'\n>>> V10 baseline лучше: {v10_prauc:.6f} >= {prauc_a:.6f}')

    best_prauc = max(prauc_a, v10_prauc)

    # ── Эксперимент B: Day-Level Stacking ──
    results_b = experiment_day_stacking(val_feat, best_preds, y_val)
    for name, (prauc, params) in results_b.items():
        all_results[f'B_day_{name}'] = prauc

    # ── Эксперимент C: Customer Boost ──
    results_c = experiment_customer_boost(val_df, best_preds, y_val)
    for name, (prauc, params) in results_c.items():
        all_results[f'C_boost_{name}'] = prauc

    # ── Эксперимент D: Battery + Pruning ──
    results_d = experiment_battery_and_pruning(train_df, val_df, profiles)
    for name, prauc in results_d.items():
        all_results[f'D_{name}'] = prauc

    # ═══════════════════════════════════════════════════════
    # ИТОГИ
    # ═══════════════════════════════════════════════════════
    log(f'\n{"="*60}')
    log(f'ИТОГИ ВСЕХ ЭКСПЕРИМЕНТОВ')
    log(f'{"="*60}')

    sorted_results = sorted(all_results.items(), key=lambda x: -x[1])
    for i, (name, prauc) in enumerate(sorted_results):
        delta = prauc - v10_prauc
        marker = ' ← ЛУЧШИЙ' if i == 0 else ''
        log(f'  {i+1:2d}. {name:30s}: val={prauc:.6f} ({delta:+.6f}, {100*delta/v10_prauc:+.1f}%){marker}')

    # Сохраняем
    with open(MODELS_OUT / 'results.json', 'w') as f:
        json.dump({k: float(v) for k, v in all_results.items()}, f, indent=2)

    log(f'\nОбщее время: {(time.time()-t_total)/60:.1f} мин')
    log('DONE')
