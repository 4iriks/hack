"""
Двухуровневая модель: Customer-level × Transaction-level.

Ключевая находка: customer-level PR-AUC = 0.214 (vs 0.039 transaction-level).
Модель УМЕЕТ находить fraud-клиентов, но тонет в их обычных транзакциях.

Стратегия:
1. Вычислить "подозрительность клиента" (из агрегатов скоров)
2. Умножить транзакционный скор на подозрительность клиента
3. Оптимизировать параметры на val
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize, differential_evolution
from pathlib import Path
from datetime import datetime
import gc

ROOT = Path('/home/vadim/PyPr/hak')
DATA = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_V10 = ROOT / 'models_v10'
MODELS_V12B = ROOT / 'models_v12b'

seeds = [42, 123, 777, 2024, 31337]

def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

import importlib.util
spec = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
v9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9)


def load_val_with_scores():
    """Загрузить val + v10 и hard_neg скоры."""
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())

    val_df = pl.read_parquet(FEATURES_IN / 'val_proper.parquet')
    val_df = val_df.with_columns(
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'))

    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    val_feat = v9.add_features(val_df)
    val_feat = v9.add_customer_profiles(val_feat, profiles)

    feats = [c for c in v9.FEATURE_COLS if c in val_feat.columns]
    X_val = val_feat.select(feats).to_pandas().astype(np.float32)

    # V10 скоры
    v10_preds = np.mean([
        lgb.Booster(model_file=str(MODELS_V10 / f'm1_lgbm_s{s}.txt')).predict(X_val)
        for s in seeds], axis=0)

    # Hard neg скоры (если есть)
    hard_neg_preds = None
    if (MODELS_V12B / 'hard_neg_s42.txt').exists():
        hard_neg_preds = np.mean([
            lgb.Booster(model_file=str(MODELS_V12B / f'hard_neg_s{s}.txt')).predict(X_val)
            for s in seeds], axis=0)

    y_val = val_feat['is_fraud'].to_numpy().astype(int)
    customer_ids = val_feat['customer_id'].to_numpy()

    return val_feat, X_val, v10_preds, hard_neg_preds, y_val, customer_ids


def compute_customer_features(preds, customer_ids, val_feat):
    """Вычислить customer-level фичи из скоров и данных."""
    unique_custs = np.unique(customer_ids)

    # Скоровые агрегаты
    cust_score_max = np.zeros_like(preds)
    cust_score_mean = np.zeros_like(preds)
    cust_score_std = np.zeros_like(preds)
    cust_score_min = np.zeros_like(preds)
    cust_score_p90 = np.zeros_like(preds)
    cust_n_high = np.zeros_like(preds)  # сколько транзакций с высоким скором
    cust_n_tx = np.zeros_like(preds)
    cust_score_range = np.zeros_like(preds)

    threshold_high = np.percentile(preds, 95)

    for cid in unique_custs:
        mask = customer_ids == cid
        s = preds[mask]
        cust_score_max[mask] = s.max()
        cust_score_mean[mask] = s.mean()
        cust_score_std[mask] = s.std() if len(s) > 1 else 0
        cust_score_min[mask] = s.min()
        cust_score_p90[mask] = np.percentile(s, 90) if len(s) >= 10 else s.max()
        cust_n_high[mask] = (s >= threshold_high).sum()
        cust_n_tx[mask] = len(s)
        cust_score_range[mask] = s.max() - s.min()

    # Позиция транзакции внутри клиента
    cust_rank = np.zeros_like(preds)
    cust_rank_pct = np.zeros_like(preds)
    for cid in unique_custs:
        mask = customer_ids == cid
        s = preds[mask]
        ranks = np.argsort(np.argsort(-s))  # 0 = highest
        cust_rank[mask] = ranks
        cust_rank_pct[mask] = ranks / max(len(s) - 1, 1)

    return {
        'cust_score_max': cust_score_max,
        'cust_score_mean': cust_score_mean,
        'cust_score_std': cust_score_std,
        'cust_score_min': cust_score_min,
        'cust_score_p90': cust_score_p90,
        'cust_n_high': cust_n_high,
        'cust_n_tx': cust_n_tx,
        'cust_score_range': cust_score_range,
        'cust_rank': cust_rank,
        'cust_rank_pct': cust_rank_pct,
    }


if __name__ == '__main__':
    log('=== ДВУХУРОВНЕВАЯ МОДЕЛЬ ===')

    val_feat, X_val, v10_preds, hard_neg_preds, y_val, customer_ids = load_val_with_scores()
    n_fraud = y_val.sum()
    log(f'Val: {len(y_val):,}, fraud={n_fraud}')

    # Базовые метрики
    v10_prauc = average_precision_score(y_val, v10_preds)
    log(f'V10 baseline: val={v10_prauc:.6f}')

    if hard_neg_preds is not None:
        hn_prauc = average_precision_score(y_val, hard_neg_preds)
        log(f'Hard neg baseline: val={hn_prauc:.6f}')

    # ═══════════════════════════════════════
    # Используем оба набора скоров: v10 и hard_neg
    # ═══════════════════════════════════════
    base_models = {'v10': v10_preds}
    if hard_neg_preds is not None:
        base_models['hard_neg'] = hard_neg_preds
        # Blend
        from scipy.stats import rankdata
        blend = 0.5 * rankdata(v10_preds) + 0.5 * rankdata(hard_neg_preds)
        base_models['blend_50_50'] = blend
        # Оптимальный blend
        def neg_blend(w):
            b = w * rankdata(v10_preds) + (1-w) * rankdata(hard_neg_preds)
            return -average_precision_score(y_val, b)
        from scipy.optimize import minimize_scalar
        opt = minimize_scalar(neg_blend, bounds=(0.0, 1.0), method='bounded')
        best_w = opt.x
        best_blend = best_w * rankdata(v10_preds) + (1-best_w) * rankdata(hard_neg_preds)
        base_models[f'blend_opt_{best_w:.2f}'] = best_blend
        log(f'Optimal blend: w_v10={best_w:.3f}, val={-opt.fun:.6f}')

    # ═══════════════════════════════════════
    # Для каждой базовой модели: customer-level boosting
    # ═══════════════════════════════════════
    results = {}

    for model_name, preds in base_models.items():
        base_prauc = average_precision_score(y_val, preds)
        log(f'\n{"="*60}')
        log(f'Модель: {model_name} (base val={base_prauc:.6f})')
        log(f'{"="*60}')

        cust_feats = compute_customer_features(preds, customer_ids, val_feat)

        # ── Метод 1: score × cust_max^a ──
        log('\n--- M1: score × cust_max^a ---')
        def neg1(a):
            return -average_precision_score(y_val, preds * (cust_feats['cust_score_max'] ** a))
        opt1 = minimize_scalar(neg1, bounds=(0.1, 5.0), method='bounded')
        prauc1 = -opt1.fun
        log(f'  cust_max^{opt1.x:.2f}: val={prauc1:.6f} ({prauc1/base_prauc-1:+.1%})')
        results[f'{model_name}_M1'] = prauc1

        # ── Метод 2: score^a × cust_max^b ──
        log('\n--- M2: score^a × cust_max^b ---')
        def neg2(params):
            a, b = params
            return -average_precision_score(y_val, (preds ** a) * (cust_feats['cust_score_max'] ** b))
        opt2 = differential_evolution(neg2, [(0.1, 3.0), (0.1, 5.0)], seed=42, maxiter=200)
        a2, b2 = opt2.x
        prauc2 = -opt2.fun
        log(f'  score^{a2:.2f} × cust_max^{b2:.2f}: val={prauc2:.6f} ({prauc2/base_prauc-1:+.1%})')
        results[f'{model_name}_M2'] = prauc2

        # ── Метод 3: score^a × cust_max^b × cust_mean^c ──
        log('\n--- M3: score^a × cust_max^b × cust_mean^c ---')
        def neg3(params):
            a, b, c = params
            s = (preds ** a) * (cust_feats['cust_score_max'] ** b) * (cust_feats['cust_score_mean'] ** c)
            return -average_precision_score(y_val, s)
        opt3 = differential_evolution(neg3, [(0.1, 3.0), (0.1, 5.0), (0.0, 3.0)],
                                       seed=42, maxiter=200)
        a3, b3, c3 = opt3.x
        prauc3 = -opt3.fun
        log(f'  score^{a3:.2f} × cust_max^{b3:.2f} × cust_mean^{c3:.2f}: val={prauc3:.6f} ({prauc3/base_prauc-1:+.1%})')
        results[f'{model_name}_M3'] = prauc3

        # ── Метод 4: score^a × cust_max^b × cust_n_high^c × cust_range^d ──
        log('\n--- M4: 4-param optimization ---')
        def neg4(params):
            a, b, c, d = params
            s = ((preds ** a) *
                 (cust_feats['cust_score_max'] ** b) *
                 ((cust_feats['cust_n_high'] + 1) ** c) *
                 ((cust_feats['cust_score_range'] + 0.001) ** d))
            return -average_precision_score(y_val, s)
        opt4 = differential_evolution(neg4,
            [(0.1, 3.0), (0.1, 5.0), (-1.0, 2.0), (-2.0, 2.0)],
            seed=42, maxiter=300)
        a4, b4, c4, d4 = opt4.x
        prauc4 = -opt4.fun
        log(f'  score^{a4:.2f} × max^{b4:.2f} × n_high^{c4:.2f} × range^{d4:.2f}: val={prauc4:.6f} ({prauc4/base_prauc-1:+.1%})')
        results[f'{model_name}_M4'] = prauc4

        # ── Метод 5: Внутриклиентская нормализация + customer boost ──
        log('\n--- M5: cust_rank_weighted ---')
        # Идея: скор = позиция внутри клиента × подозрительность клиента
        # Транзакция #1 клиента (самая подозрительная) получает максимальный буст
        rank_score = (1 - cust_feats['cust_rank_pct'])  # 1 = top, 0 = bottom
        def neg5(params):
            a, b, c = params
            s = (preds ** a) * (cust_feats['cust_score_max'] ** b) * (rank_score ** c)
            return -average_precision_score(y_val, s)
        opt5 = differential_evolution(neg5, [(0.1, 3.0), (0.1, 5.0), (0.0, 3.0)],
                                       seed=42, maxiter=200)
        a5, b5, c5 = opt5.x
        prauc5 = -opt5.fun
        log(f'  score^{a5:.2f} × max^{b5:.2f} × rank^{c5:.2f}: val={prauc5:.6f} ({prauc5/base_prauc-1:+.1%})')
        results[f'{model_name}_M5'] = prauc5

        # ── Метод 6: Top-K customer filtering ──
        log('\n--- M6: customer top-K filter ---')
        unique_custs = np.unique(customer_ids)
        cust_max_dict = {}
        for cid in unique_custs:
            mask = customer_ids == cid
            cust_max_dict[cid] = preds[mask].max()

        # Для разных K: обнулить скор клиентов ВНЕ top-K
        for top_k in [100, 200, 500, 1000, 2000, 5000]:
            sorted_custs = sorted(cust_max_dict.items(), key=lambda x: -x[1])
            top_cust_set = set(cid for cid, _ in sorted_custs[:top_k])

            filtered = preds.copy()
            for i in range(len(filtered)):
                if customer_ids[i] not in top_cust_set:
                    filtered[i] = 0

            prauc_f = average_precision_score(y_val, filtered)
            # Сколько fraud-клиентов в top-K?
            fraud_custs = set(customer_ids[y_val == 1])
            n_fraud_in_topk = len(fraud_custs & top_cust_set)
            log(f'  Top-{top_k:5d} customers: val={prauc_f:.6f}, fraud_custs={n_fraud_in_topk}/{len(fraud_custs)}')
            results[f'{model_name}_topK_{top_k}'] = prauc_f

    # ═══════════════════════════════════════
    # BLEND v10 + hard_neg + customer boost (всё вместе)
    # ═══════════════════════════════════════
    if hard_neg_preds is not None:
        log(f'\n{"="*60}')
        log('ФИНАЛЬНАЯ ОПТИМИЗАЦИЯ: blend + customer boost')
        log(f'{"="*60}')

        cust_max_v10 = np.zeros_like(v10_preds)
        cust_max_hn = np.zeros_like(hard_neg_preds)
        cust_mean_v10 = np.zeros_like(v10_preds)
        unique_custs = np.unique(customer_ids)
        for cid in unique_custs:
            mask = customer_ids == cid
            cust_max_v10[mask] = v10_preds[mask].max()
            cust_max_hn[mask] = hard_neg_preds[mask].max()
            cust_mean_v10[mask] = v10_preds[mask].mean()

        # 6 параметров: w_v10, a_score, b_max_v10, c_max_hn, d_mean
        def neg_final(params):
            w, a, b, c, d = params
            blended = w * rankdata(v10_preds) / len(v10_preds) + (1-w) * rankdata(hard_neg_preds) / len(hard_neg_preds)
            s = (blended ** a) * (cust_max_v10 ** b) * (cust_max_hn ** c) * (cust_mean_v10 ** d)
            return -average_precision_score(y_val, s)

        opt_final = differential_evolution(neg_final,
            [(0.0, 1.0), (0.1, 3.0), (0.0, 5.0), (0.0, 5.0), (0.0, 3.0)],
            seed=42, maxiter=500, tol=1e-7)
        w, a, b, c, d = opt_final.x
        prauc_final = -opt_final.fun
        log(f'\nФинальный: w_v10={w:.3f}, score^{a:.2f} × v10_max^{b:.2f} × hn_max^{c:.2f} × v10_mean^{d:.2f}')
        log(f'Val PR-AUC = {prauc_final:.6f} ({prauc_final/v10_prauc-1:+.1%} vs v10 baseline)')
        results['FINAL_blend_boost'] = prauc_final

    # ═══════════════════════════════════════
    # ИТОГИ
    # ═══════════════════════════════════════
    log(f'\n{"="*60}')
    log('ИТОГИ (TOP 15)')
    log(f'{"="*60}')

    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    for i, (name, prauc) in enumerate(sorted_results[:15]):
        delta_pct = 100 * (prauc / v10_prauc - 1)
        marker = ' ← ЛУЧШИЙ' if i == 0 else ''
        log(f'  {i+1:2d}. {name:40s}: val={prauc:.6f} ({delta_pct:+.1f}% vs v10){marker}')

    # Калибровка
    best_name, best_prauc = sorted_results[0]
    log(f'\nЛучший: {best_name}')
    log(f'Val PR-AUC: {best_prauc:.6f}')
    log(f'Ожидаемый LB (×2.3): {best_prauc * 2.3:.4f}')
    log(f'Ожидаемый LB (×2.5): {best_prauc * 2.5:.4f}')

    import json
    with open(ROOT / 'models_v12b' / 'twolevel_results.json', 'w') as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)

    log('\nDONE')
