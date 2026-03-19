"""
02_honest_val.py — Честная валидация: профили без утечки будущего.

ПРОБЛЕМА:
  deep_customer_profiles считались по ВСЕМ 177M ops (pretrain Oct'23 + train Oct'24-May'25).
  Val-транзакция в ноябре 2024 "видит" профиль с данными до мая 2025 = УТЕЧКА.

РЕШЕНИЕ:
  Для каждого клиента, считать профиль ТОЛЬКО по данным ДО его val-даты:
    - pretrain (Oct'23-Sep'24): включаем ВСЕГДА (до val периода)
    - train (Oct'24-May'25): включаем только строки ДО val_date клиента

РЕЗУЛЬТАТ:
  Сравниваем val PR-AUC с "утёкшими" vs "честными" профилями.
  Это покажет, насколько текущая val-метрика раздута утечкой.
"""

import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from pathlib import Path
from datetime import datetime
import gc, time, math, importlib.util

ROOT        = Path('/home/vadim/PyPr/hak')
RAW         = ROOT / 'Pre-train_Train'
FEATURES    = ROOT / 'features'
DATA        = ROOT / 'main_data'
MODELS_DIR  = ROOT / 'models_v14'

PRETRAIN_FILES = sorted(RAW.glob('pretrain_*.parquet'))
TRAIN_FILES    = sorted(RAW.glob('train_*.parquet'))

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

def _entropy_from_counts(counts_list):
    if counts_list is None or len(counts_list) == 0:
        return 0.0
    total = sum(counts_list)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts_list:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


# ═══════════════════════════════════════════════════════════
# Загрузка файла с per-customer фильтрацией
# ═══════════════════════════════════════════════════════════

def load_filtered(filepath, val_cutoffs, columns=None):
    """
    Загрузить raw parquet файл. Для train файлов — отфильтровать
    строки, оставив только данные ДО val_date каждого клиента.
    Pretrain файлы загружаются целиком (они все до Oct'24).
    """
    is_pretrain = 'pretrain' in filepath.name

    df = pl.read_parquet(filepath, columns=columns)

    if not is_pretrain:
        n_before = len(df)
        df = df.join(val_cutoffs, on='customer_id', how='left')
        # Оставляем строки:
        #   - если клиент не в val (val_cutoff IS NULL) → все строки
        #   - если event_dttm < val_cutoff → строго ДО val-дня
        df = df.filter(
            pl.col('val_cutoff').is_null() |
            (pl.col('event_dttm') < pl.col('val_cutoff'))
        )
        df = df.drop('val_cutoff')
        n_after = len(df)
        log(f'    {filepath.name}: {n_before:,} → {n_after:,} (-{n_before - n_after:,} отфильтровано)')
    else:
        log(f'    {filepath.name}: {len(df):,} (pretrain, без фильтрации)')

    return df


# ═══════════════════════════════════════════════════════════
# Пересчёт профилей (аналог build_all_profiles из v14)
# ═══════════════════════════════════════════════════════════

def build_honest_profiles(val_cutoffs):
    """
    Пересчёт deep_customer_profiles с per-customer фильтрацией.
    Для каждого клиента используются только данные ДО его val-даты.
    """
    log('═══ ПЕРЕСЧЁТ ЧЕСТНЫХ ПРОФИЛЕЙ ═══')
    all_files = PRETRAIN_FILES + TRAIN_FILES

    # ── Step 1: MCC profiles ──
    log('Step 1/5: MCC-Amount профили...')
    all_mcc = []
    for f in all_files:
        df = load_filtered(f, val_cutoffs,
                           columns=['customer_id', 'mcc_code', 'operaton_amt', 'event_dttm'])
        df = df.drop('event_dttm')
        mcc = df.group_by(['customer_id', 'mcc_code']).agg([
            pl.col('operaton_amt').mean().alias('mcc_amt_mean'),
            pl.len().alias('mcc_n_tx'),
        ])
        all_mcc.append(mcc)
        del df; gc.collect()

    mcc_combined = pl.concat(all_mcc)
    del all_mcc; gc.collect()
    mcc_profiles = mcc_combined.group_by(['customer_id', 'mcc_code']).agg([
        (pl.col('mcc_amt_mean') * pl.col('mcc_n_tx')).sum().alias('_weighted_sum'),
        pl.col('mcc_n_tx').sum().alias('mcc_n_tx_total'),
    ]).with_columns(
        (pl.col('_weighted_sum') / pl.col('mcc_n_tx_total')).alias('mcc_amt_mean')
    ).drop('_weighted_sum')

    if mcc_profiles['mcc_code'].dtype != pl.Int32:
        mcc_profiles = mcc_profiles.with_columns(pl.col('mcc_code').cast(pl.Int32))

    log(f'  MCC profiles: {mcc_profiles.height:,} пар (customer×MCC)')
    del mcc_combined; gc.collect()

    # ── Step 2: Daily patterns ──
    log('Step 2/5: Daily patterns...')
    all_daily = []
    for f in all_files:
        df = load_filtered(f, val_cutoffs,
                           columns=['customer_id', 'event_dttm'])
        df = df.with_columns(
            pl.col('event_dttm').str.slice(0, 10).alias('date')
        )
        daily = df.group_by(['customer_id', 'date']).agg(pl.len().alias('n_ops'))
        daily_stats = daily.group_by('customer_id').agg([
            pl.col('n_ops').mean().alias('_daily_mean'),
            pl.col('n_ops').max().alias('_daily_max'),
            pl.col('n_ops').quantile(0.95).alias('_daily_p95'),
            pl.len().alias('_n_active_days'),
        ])
        all_daily.append(daily_stats)
        del df, daily; gc.collect()

    daily_combined = pl.concat(all_daily)
    del all_daily; gc.collect()
    daily_profiles = daily_combined.group_by('customer_id').agg([
        (pl.col('_daily_mean') * pl.col('_n_active_days')).sum().alias('_wsum_daily'),
        pl.col('_n_active_days').sum().alias('n_active_days'),
        pl.col('_daily_max').max().alias('daily_max'),
        pl.col('_daily_p95').max().alias('daily_p95'),
    ]).with_columns(
        (pl.col('_wsum_daily') / pl.col('n_active_days')).alias('daily_mean')
    ).drop('_wsum_daily')
    log(f'  Daily profiles: {daily_profiles.height:,} клиентов')

    # ── Step 3: Main aggregates (lazy scan + streaming) ──
    log('Step 3/5: Основные агрегаты (streaming)...')

    # Строим lazy frames с per-customer фильтрацией
    # Выбираем только нужные колонки (типы session_id различаются между файлами)
    NEEDED_COLS = ['customer_id', 'event_dttm', 'operaton_amt', 'mcc_code',
                   'channel_indicator_type', 'operating_system_type',
                   'phone_voip_call_state', 'web_rdp_connection']

    lf_parts = []
    for f in PRETRAIN_FILES:
        lf_parts.append(pl.scan_parquet(f).select(NEEDED_COLS))

    val_cutoffs_lf = val_cutoffs.lazy()
    for f in TRAIN_FILES:
        lf = (
            pl.scan_parquet(f)
            .select(NEEDED_COLS)
            .join(val_cutoffs_lf, on='customer_id', how='left')
            .filter(
                pl.col('val_cutoff').is_null() |
                (pl.col('event_dttm') < pl.col('val_cutoff'))
            )
            .drop('val_cutoff')
        )
        lf_parts.append(lf)

    all_lf = pl.concat(lf_parts)

    main_agg = (
        all_lf
        .with_columns([
            pl.col('event_dttm').str.slice(11, 2).cast(pl.Int32).alias('hour'),
            pl.col('event_dttm').str.slice(0, 10).str.to_date('%Y-%m-%d').dt.weekday().alias('weekday'),
        ])
        .group_by('customer_id')
        .agg([
            pl.col('operaton_amt').mean().alias('prof_amt_mean'),
            pl.col('operaton_amt').std().alias('prof_amt_std'),
            pl.col('operaton_amt').median().alias('prof_amt_median'),
            pl.col('operaton_amt').quantile(0.10).alias('prof_amt_p10'),
            pl.col('operaton_amt').quantile(0.25).alias('prof_amt_p25'),
            pl.col('operaton_amt').quantile(0.75).alias('prof_amt_p75'),
            pl.col('operaton_amt').quantile(0.90).alias('prof_amt_p90'),
            pl.col('operaton_amt').quantile(0.95).alias('prof_amt_p95'),
            pl.col('operaton_amt').quantile(0.99).alias('prof_amt_p99'),
            pl.col('operaton_amt').max().alias('prof_amt_max'),
            pl.col('operaton_amt').min().alias('prof_amt_min'),
            pl.len().alias('prof_n_tx'),
            pl.col('hour').mean().alias('prof_hour_mean'),
            pl.col('hour').std().alias('prof_hour_std'),
            ((pl.col('hour') >= 22) | (pl.col('hour') < 6)).mean().alias('prof_pct_night'),
            (pl.col('weekday') >= 6).mean().alias('prof_pct_weekend'),
            pl.col('mcc_code').n_unique().alias('prof_n_unique_mcc'),
            pl.col('channel_indicator_type').n_unique().alias('prof_n_unique_channel'),
            pl.col('operating_system_type').n_unique().alias('prof_n_unique_os'),
            (pl.col('phone_voip_call_state') == 1).sum().alias('prof_voip_sum'),
            (pl.col('web_rdp_connection') == 1).sum().alias('prof_rdp_sum'),
            (pl.col('operaton_amt').quantile(0.75) - pl.col('operaton_amt').quantile(0.25)).alias('prof_amt_iqr'),
        ])
        .collect(engine="streaming")
    )
    log(f'  Main aggregates: {main_agg.height:,} клиентов')

    # ── Step 4: Hour/MCC entropy ──
    log('Step 4/5: Энтропия час/MCC...')
    hour_counts_all = []
    mcc_counts_all = []
    for f in all_files:
        df = load_filtered(f, val_cutoffs,
                           columns=['customer_id', 'event_dttm', 'mcc_code'])
        df = df.with_columns(
            pl.col('event_dttm').str.slice(11, 2).cast(pl.Int32).alias('hour')
        )
        hc = df.group_by(['customer_id', 'hour']).agg(pl.len().alias('cnt'))
        hour_counts_all.append(hc)
        mc = df.group_by(['customer_id', 'mcc_code']).agg(pl.len().alias('cnt'))
        mcc_counts_all.append(mc)
        del df; gc.collect()

    hour_counts = pl.concat(hour_counts_all).group_by(['customer_id', 'hour']).agg(
        pl.col('cnt').sum()
    )
    del hour_counts_all; gc.collect()

    hour_entropy = (
        hour_counts
        .group_by('customer_id')
        .agg(pl.col('cnt').alias('counts'))
        .with_columns(
            pl.col('counts').map_elements(
                lambda lst: _entropy_from_counts(lst),
                return_dtype=pl.Float64
            ).alias('prof_hour_entropy')
        )
        .select(['customer_id', 'prof_hour_entropy'])
    )
    del hour_counts; gc.collect()

    mcc_counts = pl.concat(mcc_counts_all).group_by(['customer_id', 'mcc_code']).agg(
        pl.col('cnt').sum()
    )
    del mcc_counts_all; gc.collect()

    mcc_entropy = (
        mcc_counts
        .group_by('customer_id')
        .agg(pl.col('cnt').alias('counts'))
        .with_columns(
            pl.col('counts').map_elements(
                lambda lst: _entropy_from_counts(lst),
                return_dtype=pl.Float64
            ).alias('prof_mcc_entropy')
        )
        .select(['customer_id', 'prof_mcc_entropy'])
    )
    del mcc_counts; gc.collect()
    log(f'  Entropy: hour={hour_entropy.height:,}, mcc={mcc_entropy.height:,} клиентов')

    # ── Step 5: Gap statistics ──
    log('Step 5/5: Gap stats...')
    gap_stats_all = []
    for f in all_files:
        df = load_filtered(f, val_cutoffs,
                           columns=['customer_id', 'event_dttm'])
        df = df.with_columns(
            pl.col('event_dttm').str.to_datetime('%Y-%m-%d %H:%M:%S').alias('dt')
        ).sort(['customer_id', 'dt'])

        df = df.with_columns(
            (pl.col('dt') - pl.col('dt').shift(1).over('customer_id'))
            .dt.total_seconds().alias('gap_sec')
        )

        gaps = df.filter(
            pl.col('gap_sec').is_not_null() & (pl.col('gap_sec') >= 0)
        ).group_by('customer_id').agg([
            pl.col('gap_sec').median().alias('_gap_median'),
            pl.col('gap_sec').mean().alias('_gap_mean'),
            pl.col('gap_sec').std().alias('_gap_std'),
            pl.col('gap_sec').quantile(0.95).alias('_gap_p95'),
            pl.len().alias('_gap_n'),
        ])
        gap_stats_all.append(gaps)
        del df; gc.collect()

    gap_combined = pl.concat(gap_stats_all)
    del gap_stats_all; gc.collect()
    gap_profiles = gap_combined.group_by('customer_id').agg([
        (pl.col('_gap_mean') * pl.col('_gap_n')).sum().alias('_wsum'),
        (pl.col('_gap_std') * pl.col('_gap_n')).sum().alias('_wsum_std'),
        pl.col('_gap_n').sum().alias('_total_n'),
        pl.col('_gap_p95').max().alias('prof_gap_p95'),
        pl.col('_gap_median').median().alias('prof_gap_median'),
    ]).with_columns([
        (pl.col('_wsum') / pl.col('_total_n')).alias('prof_gap_mean'),
        (pl.col('_wsum_std') / pl.col('_total_n')).alias('prof_gap_std'),
    ]).select(['customer_id', 'prof_gap_mean', 'prof_gap_std', 'prof_gap_median', 'prof_gap_p95'])
    del gap_combined; gc.collect()
    log(f'  Gap profiles: {gap_profiles.height:,} клиентов')

    # ── Merge all ──
    log('Мержим все профили...')
    profiles = main_agg
    profiles = profiles.join(daily_profiles, on='customer_id', how='left')
    profiles = profiles.join(hour_entropy, on='customer_id', how='left')
    profiles = profiles.join(mcc_entropy, on='customer_id', how='left')
    profiles = profiles.join(gap_profiles, on='customer_id', how='left')

    profiles = profiles.with_columns([
        (pl.col('prof_voip_sum') / pl.col('prof_n_tx')).alias('prof_voip_rate'),
        (pl.col('prof_rdp_sum') / pl.col('prof_n_tx')).alias('prof_rdp_rate'),
        (pl.col('prof_n_tx') / pl.col('n_active_days').clip(lower_bound=1)).alias('prof_tx_per_day'),
        (pl.col('prof_amt_std') / pl.col('prof_amt_mean').clip(lower_bound=0.01)).alias('prof_amt_cv'),
    ])

    log(f'Честные профили: {profiles.height:,} клиентов, {len(profiles.columns)} колонок')
    return profiles, mcc_profiles


# ═══════════════════════════════════════════════════════════
# Оценка: сравнение leaked vs honest val PR-AUC
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_start = time.time()
    log('═══════════════════════════════════════════════════════')
    log('ЧЕСТНАЯ ВАЛИДАЦИЯ: ПРОФИЛИ БЕЗ УТЕЧКИ')
    log('═══════════════════════════════════════════════════════')

    # ── 1. Загрузка val и вычисление cutoff-дат ──
    log('1. Загрузка val_proper и вычисление cutoff дат...')
    val_df = pl.read_parquet(FEATURES / 'val_proper.parquet')
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())

    # Val cutoff = самая ранняя дата val-транзакций клиента (строка ISO)
    # event_dttm в val_proper — datetime. Конвертируем в строку для сравнения с raw данными.
    val_cutoffs = val_df.group_by('customer_id').agg(
        pl.col('event_dttm').min().dt.strftime('%Y-%m-%d %H:%M:%S').alias('val_cutoff')
    )
    log(f'  Val: {len(val_df):,} строк, {val_cutoffs.height:,} клиентов')
    log(f'  Cutoff диапазон: {val_cutoffs["val_cutoff"].min()} → {val_cutoffs["val_cutoff"].max()}')

    # Распределение val по месяцам
    val_months = val_df.with_columns(
        pl.col('event_dttm').dt.strftime('%Y-%m').alias('month')
    ).group_by('month').len().sort('month')
    for row in val_months.iter_rows():
        log(f'    {row[0]}: {row[1]:,} val ops')

    # ── 2. Пересчёт честных профилей ──
    honest_profiles, honest_mcc = build_honest_profiles(val_cutoffs)

    # ── 3. Загрузка утёкших профилей (для сравнения) ──
    log('\n3. Загрузка утёкших (текущих) профилей...')
    leaked_profiles = pl.read_parquet(FEATURES / 'deep_customer_profiles.parquet')
    leaked_mcc = pl.read_parquet(FEATURES / 'customer_mcc_profiles.parquet')
    if leaked_mcc['mcc_code'].dtype != pl.Int32:
        leaked_mcc = leaked_mcc.with_columns(pl.col('mcc_code').cast(pl.Int32))
    old_profiles = pl.read_parquet(FEATURES / 'customer_profiles.parquet')

    # ── 4. Подготовка val фичей ──
    log('\n4. Подготовка val фичей (оба варианта)...')

    # Загрузка v9 и v14 для add_features / add_anomaly_features
    spec9 = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
    v9 = importlib.util.module_from_spec(spec9)
    spec9.loader.exec_module(v9)

    spec14 = importlib.util.spec_from_file_location("v14", ROOT / "pipeline_v14.py")
    v14 = importlib.util.module_from_spec(spec14)
    spec14.loader.exec_module(v14)

    # Базовые фичи (одинаковые для обоих вариантов — нет утечки)
    val_base = v9.add_features(val_df)
    val_base = v9.add_customer_profiles(val_base, old_profiles)

    # Вариант A: УТЁКШИЕ профили (текущие)
    log('  Вариант A: утёкшие профили...')
    val_leaked = v14.add_anomaly_features(val_base.clone(), leaked_profiles, leaked_mcc)

    # Вариант B: ЧЕСТНЫЕ профили
    log('  Вариант B: честные профили...')
    val_honest = v14.add_anomaly_features(val_base.clone(), honest_profiles, honest_mcc)

    del val_base; gc.collect()

    # ── 5. Загрузка модели и предсказание ──
    log('\n5. Предсказания v14-C (5 seeds)...')
    seeds = [42, 123, 777, 2024, 31337]

    # Получаем имена фичей из модели
    bst_tmp = lgb.Booster(model_file=str(MODELS_DIR / 'C_base_plus_anomaly_s42.txt'))
    model_feats = bst_tmp.feature_name()
    del bst_tmp

    y_val = val_df.with_columns(
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud')
    )['is_fraud'].to_numpy()

    def prepare_X(df, feat_names):
        """Подготовить матрицу фичей, заполнив отсутствующие нулями."""
        for c in feat_names:
            if c not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(c))
        return df.select(feat_names).to_pandas().values.astype(np.float32)

    X_leaked = prepare_X(val_leaked, model_feats)
    X_honest = prepare_X(val_honest, model_feats)

    del val_leaked, val_honest; gc.collect()

    # Предсказания ансамбля
    preds_leaked = np.zeros(len(y_val), dtype=np.float64)
    preds_honest = np.zeros(len(y_val), dtype=np.float64)

    for seed in seeds:
        bst = lgb.Booster(model_file=str(MODELS_DIR / f'C_base_plus_anomaly_s{seed}.txt'))
        preds_leaked += bst.predict(X_leaked)
        preds_honest += bst.predict(X_honest)
        del bst; gc.collect()

    preds_leaked /= len(seeds)
    preds_honest /= len(seeds)

    # ── 6. Результаты ──
    prauc_leaked = average_precision_score(y_val, preds_leaked)
    prauc_honest = average_precision_score(y_val, preds_honest)

    log('\n' + '=' * 60)
    log('РЕЗУЛЬТАТЫ: ЧЕСТНАЯ vs УТЁКШАЯ ВАЛИДАЦИЯ')
    log('=' * 60)
    log(f'  Val PR-AUC (утёкшие профили):  {prauc_leaked:.6f}')
    log(f'  Val PR-AUC (честные профили):  {prauc_honest:.6f}')
    log(f'  Разница:                       {prauc_honest - prauc_leaked:+.6f} ({100*(prauc_honest/prauc_leaked - 1):+.1f}%)')
    log(f'')
    log(f'  LB PR-AUC (v14-C):             0.1010')
    log(f'  Ratio LB/Leaked val:           {0.1010/prauc_leaked:.2f}')
    log(f'  Ratio LB/Honest val:           {0.1010/prauc_honest:.2f}')
    log(f'')

    if prauc_honest < prauc_leaked:
        log('  ВЫВОД: Утечка РАЗДУВАЕТ val PR-AUC.')
        log(f'         Настоящий val = {prauc_honest:.6f}, было {prauc_leaked:.6f}')
        log(f'         Ratio LB/honest_val = {0.1010/prauc_honest:.2f} (стабильнее, чем {0.1010/prauc_leaked:.2f})')
    elif prauc_honest > prauc_leaked:
        log('  ВЫВОД: Утечка МЕШАЕТ модели (шум от будущих данных?).')
        log('         Честные профили ЛУЧШЕ → модель не умеет использовать будущее.')
    else:
        log('  ВЫВОД: Разницы нет — профили устойчивы к temporal leakage.')

    # ── 7. Детальный анализ по месяцам val ──
    log('\n' + '=' * 60)
    log('ДЕТАЛЬНЫЙ АНАЛИЗ ПО МЕСЯЦАМ VAL')
    log('=' * 60)

    val_months_col = val_df['event_dttm'].dt.strftime('%Y-%m').to_numpy()

    for month in sorted(set(val_months_col)):
        mask = val_months_col == month
        n = mask.sum()
        n_fraud = y_val[mask].sum()
        if n_fraud == 0:
            log(f'  {month}: {n:,} ops, 0 fraud → skip')
            continue

        p_leaked = average_precision_score(y_val[mask], preds_leaked[mask])
        p_honest = average_precision_score(y_val[mask], preds_honest[mask])
        diff_pct = 100 * (p_honest / p_leaked - 1) if p_leaked > 0 else 0

        log(f'  {month}: {n:>7,} ops, {n_fraud:>3} fraud | '
            f'leaked={p_leaked:.4f}  honest={p_honest:.4f}  diff={diff_pct:+.1f}%')

    log(f'\nВремя: {(time.time()-t_start)/60:.1f} мин')
    log('ГОТОВО')
