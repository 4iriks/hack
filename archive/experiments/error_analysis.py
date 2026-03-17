"""
Анализ ошибок: где модель теряет PR-AUC?
Смотрим false negatives (пропущенный fraud) и false positives.
Ищем паттерны для целенаправленного улучшения.
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, precision_recall_curve
from pathlib import Path
from datetime import datetime
import gc, json

ROOT = Path('/home/vadim/PyPr/hak')
DATA = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_V10 = ROOT / 'models_v10'

seeds = [42, 123, 777, 2024, 31337]

def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

import importlib.util
spec = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
v9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9)


if __name__ == '__main__':
    log('=== АНАЛИЗ ОШИБОК V10 ===')

    # Загружаем val
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())
    confirmed_ids = set(labels.filter(pl.col('target') == 0)['event_id'].to_list())

    val_df = pl.read_parquet(FEATURES_IN / 'val_proper.parquet')
    val_df = val_df.with_columns([
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'),
        pl.col('event_id').is_in(confirmed_ids).cast(pl.Int8).alias('is_confirmed'),
    ])

    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    val_feat = v9.add_features(val_df)
    val_feat = v9.add_customer_profiles(val_feat, profiles)

    v10_feats = [c for c in v9.FEATURE_COLS if c in val_feat.columns]
    X_val = val_feat.select(v10_feats).to_pandas().astype(np.float32)

    # V10 предсказания
    v10_preds = np.mean([
        lgb.Booster(model_file=str(MODELS_V10 / f'm1_lgbm_s{s}.txt')).predict(X_val)
        for s in seeds], axis=0)

    y_val = val_feat['is_fraud'].to_numpy().astype(int)
    y_confirmed = val_feat['is_confirmed'].to_numpy().astype(int)
    customer_ids = val_feat['customer_id'].to_numpy()

    n_fraud = y_val.sum()
    n_confirmed = y_confirmed.sum()
    n_green = len(y_val) - n_fraud - n_confirmed
    log(f'Val: {len(y_val):,} всего, {n_fraud} fraud, {n_confirmed} confirmed, {n_green:,} green')
    log(f'V10 PR-AUC: {average_precision_score(y_val, v10_preds):.6f}')

    # ═══════════════════════════════════════
    # 1. Распределение скоров по классам
    # ═══════════════════════════════════════
    log('\n=== 1. РАСПРЕДЕЛЕНИЕ СКОРОВ ===')

    fraud_scores = v10_preds[y_val == 1]
    confirmed_scores = v10_preds[y_confirmed == 1]
    green_scores = v10_preds[(y_val == 0) & (y_confirmed == 0)]

    for name, scores in [('Fraud', fraud_scores), ('Confirmed', confirmed_scores), ('Green', green_scores)]:
        log(f'\n  {name} ({len(scores):,} строк):')
        log(f'    mean={scores.mean():.6f}, median={np.median(scores):.6f}')
        log(f'    min={scores.min():.6f}, max={scores.max():.6f}')
        log(f'    p25={np.percentile(scores, 25):.6f}, p75={np.percentile(scores, 75):.6f}')
        log(f'    p90={np.percentile(scores, 90):.6f}, p95={np.percentile(scores, 95):.6f}')
        log(f'    p99={np.percentile(scores, 99):.6f}')

    # ═══════════════════════════════════════
    # 2. PR-кривая: при каких порогах теряем?
    # ═══════════════════════════════════════
    log('\n=== 2. PRECISION-RECALL КРИВАЯ ===')

    precision, recall, thresholds = precision_recall_curve(y_val, v10_preds)

    # Ключевые точки
    for target_recall in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        idx = np.searchsorted(-recall, -target_recall)
        if idx < len(precision):
            p = precision[idx]
            r = recall[idx]
            t = thresholds[min(idx, len(thresholds)-1)]
            n_predicted = (v10_preds >= t).sum()
            log(f'  Recall={r:.2f}: precision={p:.4f}, threshold={t:.4f}, predicted={n_predicted:,}')

    # ═══════════════════════════════════════
    # 3. Анализ FALSE NEGATIVES (пропущенный fraud)
    # ═══════════════════════════════════════
    log('\n=== 3. FALSE NEGATIVES (пропущенный fraud) ===')

    # Ранжируем fraud по скору (низкий скор = пропущенный)
    fraud_idx = np.where(y_val == 1)[0]
    fraud_ranked = sorted(zip(fraud_idx, fraud_scores), key=lambda x: x[1])

    # Нижние 50% fraud (те, кого модель плохо видит)
    n_missed = len(fraud_ranked) // 2
    missed_idx = [idx for idx, _ in fraud_ranked[:n_missed]]
    caught_idx = [idx for idx, _ in fraud_ranked[n_missed:]]

    log(f'\nПропущенные (нижние 50% скора): {n_missed} fraud')
    log(f'Пойманные (верхние 50% скора): {len(caught_idx)} fraud')

    # Сравниваем характеристики пропущенных vs пойманных
    missed_df = val_feat[missed_idx]
    caught_df = val_feat[caught_idx]

    compare_cols = [
        'operaton_amt', 'log_amount', 'hour', 'weekday',
        'phone_voip_call_state', 'web_rdp_connection', 'compromised', 'developer_tools',
        'cnt_24h', 'cnt_30d', 'amt_sum_24h',
        'pos_cd', 'is_manual_entry', 'mcc_code',
        'cust_avg_amt', 'amt_zscore', 'dormancy_days',
        'cust_pct_high_risk', 'cust_hour_std',
        'event_type_nm', 'channel_indicator_type',
        'secs_since_last', 'is_night',
    ]

    log('\n  Фича                    | Пропущенные   | Пойманные     | Разница')
    log('  ' + '-'*75)

    for col in compare_cols:
        if col in missed_df.columns:
            m_vals = missed_df[col].drop_nulls().to_numpy().astype(float)
            c_vals = caught_df[col].drop_nulls().to_numpy().astype(float)
            if len(m_vals) > 0 and len(c_vals) > 0:
                m_mean = m_vals.mean()
                c_mean = c_vals.mean()
                diff = c_mean - m_mean
                pct = 100 * diff / (abs(m_mean) + 0.001)
                log(f'  {col:25s} | {m_mean:13.2f} | {c_mean:13.2f} | {pct:+.1f}%')

    # ═══════════════════════════════════════
    # 4. Анализ FALSE POSITIVES (green с высоким скором)
    # ═══════════════════════════════════════
    log('\n=== 4. FALSE POSITIVES (green/confirmed с высоким скором) ===')

    # Top-430 по скору (столько же сколько fraud)
    top_k = n_fraud
    top_idx = np.argsort(-v10_preds)[:top_k]
    top_labels = y_val[top_idx]
    top_confirmed = y_confirmed[top_idx]

    n_tp = top_labels.sum()
    n_fp_confirmed = top_confirmed.sum()
    n_fp_green = top_k - n_tp - n_fp_confirmed

    log(f'Top-{top_k} предсказаний:')
    log(f'  True Positives (fraud): {n_tp} ({100*n_tp/top_k:.1f}%)')
    log(f'  False Positives (confirmed): {n_fp_confirmed} ({100*n_fp_confirmed/top_k:.1f}%)')
    log(f'  False Positives (green): {n_fp_green} ({100*n_fp_green/top_k:.1f}%)')

    # Precision@K для разных K
    log('\nPrecision@K:')
    for k in [50, 100, 200, 430, 1000, 5000]:
        top_k_idx = np.argsort(-v10_preds)[:k]
        tp = y_val[top_k_idx].sum()
        log(f'  P@{k:5d}: {tp}/{k} = {tp/k:.4f} (вспомнили {tp}/{n_fraud} = {tp/n_fraud:.3f} fraud)')

    # ═══════════════════════════════════════
    # 5. Сегментный анализ: где модель сильна/слаба?
    # ═══════════════════════════════════════
    log('\n=== 5. СЕГМЕНТНЫЙ АНАЛИЗ ===')

    # По сумме транзакции
    log('\n--- По сумме (operaton_amt) ---')
    amt = val_feat['operaton_amt'].to_numpy()
    for lo, hi, name in [(0, 1000, '<1K'), (1000, 10000, '1K-10K'),
                          (10000, 50000, '10K-50K'), (50000, float('inf'), '>50K')]:
        mask = (amt >= lo) & (amt < hi)
        if mask.sum() > 0 and y_val[mask].sum() > 0:
            prauc = average_precision_score(y_val[mask], v10_preds[mask])
            log(f'  {name:8s}: {mask.sum():7,} строк, {y_val[mask].sum():3d} fraud, PR-AUC={prauc:.4f}')

    # По часу
    log('\n--- По часу ---')
    hour = val_feat['hour'].to_numpy()
    for h_lo, h_hi, name in [(0, 6, 'Ночь 0-6'), (6, 12, 'Утро 6-12'),
                              (12, 18, 'День 12-18'), (18, 24, 'Вечер 18-24')]:
        mask = (hour >= h_lo) & (hour < h_hi)
        if mask.sum() > 0 and y_val[mask].sum() > 0:
            prauc = average_precision_score(y_val[mask], v10_preds[mask])
            log(f'  {name:12s}: {mask.sum():7,} строк, {y_val[mask].sum():3d} fraud, PR-AUC={prauc:.4f}')

    # По security flags
    log('\n--- По security flags ---')
    voip = val_feat['phone_voip_call_state'].to_numpy()
    rdp = val_feat['web_rdp_connection'].to_numpy()
    comp = val_feat['compromised'].to_numpy()

    for flag_name, flag_vals in [('VoIP', voip), ('RDP', rdp), ('Compromised', comp)]:
        for v in [0, 1]:
            mask = flag_vals == v
            if mask.sum() > 0 and y_val[mask].sum() > 0:
                prauc = average_precision_score(y_val[mask], v10_preds[mask])
                log(f'  {flag_name}={v}: {mask.sum():7,} строк, {y_val[mask].sum():3d} fraud, PR-AUC={prauc:.4f}')

    # По pos_cd (manual entry)
    log('\n--- По pos_cd ---')
    pos = val_feat['pos_cd'].to_numpy()
    for p_val in [0, 1, 2, 5, 7, 9]:
        mask = pos == p_val
        if mask.sum() > 100 and y_val[mask].sum() > 0:
            prauc = average_precision_score(y_val[mask], v10_preds[mask])
            log(f'  pos_cd={p_val}: {mask.sum():7,} строк, {y_val[mask].sum():3d} fraud, PR-AUC={prauc:.4f}')

    # По типу клиента (много транзакций vs мало)
    log('\n--- По активности клиента (cnt_30d) ---')
    cnt30 = val_feat['cnt_30d'].to_numpy()
    for lo, hi, name in [(0, 5, 'Тихий <5'), (5, 20, 'Средний 5-20'),
                          (20, 100, 'Активный 20-100'), (100, float('inf'), 'Гипер >100')]:
        mask = (cnt30 >= lo) & (cnt30 < hi)
        if mask.sum() > 0 and y_val[mask].sum() > 0:
            prauc = average_precision_score(y_val[mask], v10_preds[mask])
            log(f'  {name:18s}: {mask.sum():7,} строк, {y_val[mask].sum():3d} fraud, PR-AUC={prauc:.4f}')

    # ═══════════════════════════════════════
    # 6. Overlap fraud vs confirmed (главная проблема)
    # ═══════════════════════════════════════
    log('\n=== 6. OVERLAP FRAUD vs CONFIRMED ===')

    confirmed_idx_arr = np.where(y_confirmed == 1)[0]
    if len(confirmed_idx_arr) > 0:
        confirmed_scores_val = v10_preds[confirmed_idx_arr]
        log(f'Fraud скоры:     mean={fraud_scores.mean():.4f}, median={np.median(fraud_scores):.4f}')
        log(f'Confirmed скоры: mean={confirmed_scores_val.mean():.4f}, median={np.median(confirmed_scores_val):.4f}')

        # Какая доля confirmed имеет скор выше медианы fraud?
        fraud_median = np.median(fraud_scores)
        n_conf_above = (confirmed_scores_val >= fraud_median).sum()
        log(f'Confirmed со скором >= медианы fraud: {n_conf_above}/{len(confirmed_scores_val)} = {100*n_conf_above/len(confirmed_scores_val):.1f}%')

        # Какая доля fraud имеет скор ниже медианы confirmed?
        conf_median = np.median(confirmed_scores_val)
        n_fraud_below = (fraud_scores <= conf_median).sum()
        log(f'Fraud со скором <= медианы confirmed: {n_fraud_below}/{len(fraud_scores)} = {100*n_fraud_below/len(fraud_scores):.1f}%')

    # ═══════════════════════════════════════
    # 7. Клиентский анализ
    # ═══════════════════════════════════════
    log('\n=== 7. КЛИЕНТСКИЙ АНАЛИЗ ===')

    # Клиенты с fraud
    fraud_custs = set(customer_ids[y_val == 1])
    log(f'Клиентов с fraud в val: {len(fraud_custs)}')

    # Сколько транзакций у клиентов с fraud?
    fraud_cust_sizes = []
    fraud_cust_max_scores = []
    for cid in fraud_custs:
        mask = customer_ids == cid
        fraud_cust_sizes.append(mask.sum())
        fraud_cust_max_scores.append(v10_preds[mask].max())

    log(f'Транзакций/день у fraud-клиентов: mean={np.mean(fraud_cust_sizes):.1f}, '
        f'median={np.median(fraud_cust_sizes):.1f}, max={max(fraud_cust_sizes)}')

    # Customer-level: если берём max скор на клиента, какой precision?
    cust_scores = {}
    cust_labels = {}
    for i in range(len(y_val)):
        cid = customer_ids[i]
        if cid not in cust_scores or v10_preds[i] > cust_scores[cid]:
            cust_scores[cid] = v10_preds[i]
        if y_val[i] == 1:
            cust_labels[cid] = 1
        elif cid not in cust_labels:
            cust_labels[cid] = 0

    cust_ids_list = sorted(cust_scores.keys())
    cust_s = np.array([cust_scores[c] for c in cust_ids_list])
    cust_y = np.array([cust_labels.get(c, 0) for c in cust_ids_list])

    cust_prauc = average_precision_score(cust_y, cust_s)
    log(f'\nCustomer-level PR-AUC (max score per customer): {cust_prauc:.6f}')
    log(f'(vs transaction-level: {average_precision_score(y_val, v10_preds):.6f})')

    n_fraud_custs = cust_y.sum()
    for k in [100, 200, 400, n_fraud_custs]:
        top_k_idx = np.argsort(-cust_s)[:k]
        tp = cust_y[top_k_idx].sum()
        log(f'  Customer P@{k}: {tp}/{k} = {tp/k:.4f} (recall={tp}/{n_fraud_custs}={tp/n_fraud_custs:.3f})')

    log('\nDONE')
