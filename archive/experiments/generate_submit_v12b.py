"""
Генерация сабмитов v12b: hard_neg модель + customer boost (M1 и M4).

Два варианта:
- M1 (консервативный): score × cust_max^3.13 — 1 параметр, устойчив
- M4 (агрессивный): score^1.21 × cust_max^4.80 × n_high^0.05 × range^-0.11 — лучший val
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
from pathlib import Path
from datetime import datetime
import gc

ROOT = Path('/home/vadim/PyPr/hak')
DATA = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_V10 = ROOT / 'models_v10'
MODELS_V12B = ROOT / 'models_v12b'
SUBMIT_OUT = ROOT / 'submissions'

seeds = [42, 123, 777, 2024, 31337]

def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

# Import feature engineering from v9
import importlib.util
spec = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
v9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9)


def compute_customer_features(preds, customer_ids):
    """Вычислить customer-level фичи из скоров."""
    unique_custs = np.unique(customer_ids)
    cust_score_max = np.zeros_like(preds)
    cust_n_high = np.zeros_like(preds)
    cust_score_range = np.zeros_like(preds)

    threshold_high = np.percentile(preds, 95)

    for cid in unique_custs:
        mask = customer_ids == cid
        s = preds[mask]
        cust_score_max[mask] = s.max()
        cust_n_high[mask] = (s >= threshold_high).sum()
        cust_score_range[mask] = s.max() - s.min()

    return cust_score_max, cust_n_high, cust_score_range


if __name__ == '__main__':
    log('=== ГЕНЕРАЦИЯ САБМИТОВ v12b ===')

    # Загрузить тестовые данные
    log('Загрузка test features...')
    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    df_test = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    df_test = df_test.unique(subset=['event_id'], keep='first')
    df_test = v9.add_features(df_test)
    df_test = v9.add_customer_profiles(df_test, profiles)

    feats = [c for c in v9.FEATURE_COLS if c in df_test.columns]
    X_test = df_test.select(feats).to_pandas().astype(np.float32)
    event_ids = df_test['event_id'].to_numpy()
    customer_ids = df_test['customer_id'].to_numpy()
    log(f'Test: {X_test.shape[0]:,} rows, {X_test.shape[1]} features')

    # Hard neg скоры (5 сидов)
    log('Скоринг hard_neg моделями...')
    hard_neg_preds = np.mean([
        lgb.Booster(model_file=str(MODELS_V12B / f'hard_neg_s{s}.txt')).predict(X_test)
        for s in seeds], axis=0)
    log(f'Hard neg scores: mean={hard_neg_preds.mean():.6f}, max={hard_neg_preds.max():.6f}')

    # Customer-level фичи
    log('Вычисление customer-level features...')
    cust_max, cust_n_high, cust_range = compute_customer_features(hard_neg_preds, customer_ids)

    # ── Метод M1: score × cust_max^3.13 ──
    # Оптимальный параметр с val (1 степень свободы — безопасный)
    a_m1 = 3.13
    preds_m1 = hard_neg_preds * (cust_max ** a_m1)
    log(f'M1: score × cust_max^{a_m1:.2f}')

    # ── Метод M4: score^1.21 × cust_max^4.80 × n_high^0.05 × range^-0.11 ──
    a4, b4, c4, d4 = 1.21, 4.80, 0.05, -0.11
    preds_m4 = ((hard_neg_preds ** a4) *
                (cust_max ** b4) *
                ((cust_n_high + 1) ** c4) *
                ((cust_range + 0.001) ** d4))
    log(f'M4: score^{a4} × max^{b4} × n_high^{c4} × range^{d4}')

    # ── Метод M1_v10: для сравнения, v10 + customer boost ──
    log('Скоринг v10 моделями...')
    v10_preds = np.mean([
        lgb.Booster(model_file=str(MODELS_V10 / f'm1_lgbm_s{s}.txt')).predict(X_test)
        for s in seeds], axis=0)
    cust_max_v10, _, _ = compute_customer_features(v10_preds, customer_ids)
    preds_v10_boost = v10_preds * (cust_max_v10 ** a_m1)
    log(f'V10+M1: score × cust_max^{a_m1:.2f}')

    # Сохранить сабмиты
    sample = pl.read_csv(DATA / 'sample_submit.csv')
    ts = datetime.now().strftime('%Y%m%d_%H%M')

    methods = {
        'v12b_M1_safe': preds_m1,
        'v12b_M4_aggressive': preds_m4,
        'v10_M1_boost': preds_v10_boost,
    }

    SUBMIT_OUT.mkdir(exist_ok=True)

    for name, score in methods.items():
        sub = pl.DataFrame({'event_id': event_ids, 'predict': score.astype(np.float64)})
        sub = sample.select('event_id').join(sub, on='event_id', how='left')
        n_null = sub['predict'].is_null().sum()
        if n_null > 0:
            median_val = sub['predict'].drop_nulls().median()
            sub = sub.with_columns(pl.col('predict').fill_null(median_val))
            log(f'  WARN: {n_null} nulls filled with median={median_val:.6f}')

        path = SUBMIT_OUT / f'submit_{name}_{ts}.csv'
        sub.write_csv(path)

        # Статистика
        scores = sub['predict'].to_numpy()
        log(f'  Saved: {path.name} | mean={np.mean(scores):.6f}, p50={np.median(scores):.6f}, p99={np.percentile(scores, 99):.6f}, max={np.max(scores):.6f}')

    log('\n=== РЕКОМЕНДАЦИЯ ===')
    log('1. Сначала сабмитить v12b_M1_safe (консервативный, 1 параметр)')
    log('2. Если M1 лучше v10, сабмитить v12b_M4_aggressive')
    log('3. v10_M1_boost — для проверки, даёт ли customer boost на v10 модели')
    log('\nDONE')
