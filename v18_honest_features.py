"""
v18: Point-in-Time честные фичи.

Для каждой транзакции профиль клиента считается ТОЛЬКО из данных
ДО СЕКУНДЫ этой транзакции. Никакой утечки из будущего.

Подход:
  1. Pretrain aggregate: полные профили из 90M pretrain ops (Oct'23-Sep'24)
  2. Train cumulative: кумулятивные статистики по 87M train ops (Oct'24-May'25)
  3. Combine: pretrain_base + train_cumulative (shift=1) = point-in-time profile
  4. Filter: оставляем только 2.66M train + 523K val строк
  5. Derive: prof_* и anom_* из честных профилей
  6. Train: LGBM на честных фичах

Аппроксимации:
  - Percentiles (median, p95, p99): mean ± k*std (деревьям не нужна точность)
  - IQR: 1.35 * std
  - Entropy: log2(n_unique) (аппроксимация Shannon)
  - MCC profiles: pretrain-only (90M ops, всегда до train)
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
MODELS_OUT  = ROOT / 'models_v18'
SUBMIT_OUT  = ROOT / 'submissions'

for d in [MODELS_OUT, SUBMIT_OUT]:
    d.mkdir(exist_ok=True)

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


# ═══════════════════════════════════════════════════════════
# ФАЗА 1: Pretrain Base Profiles (90M ops → 100K profiles)
# ═══════════════════════════════════════════════════════════

def build_pretrain_base():
    """
    Агрегация pretrain (Oct'23-Sep'24) → базовые статистики по клиентам.
    MEMORY-SAFE: обрабатывает файл за файлом, мержит агрегаты.
    """
    log('ФАЗА 1: Pretrain base profiles (файл за файлом)...')

    needed_cols = ['customer_id', 'event_dttm', 'operaton_amt', 'mcc_code',
                   'channel_indicator_type', 'operating_system_type',
                   'phone_voip_call_state', 'web_rdp_connection']

    file_aggs = []
    file_gap_aggs = []
    file_daily_aggs = []
    all_mcc_counts = []  # вместо mcc_sets (экономим RAM)

    for f in PRETRAIN_FILES:
        log(f'  Обработка {f.name}...')
        df = pl.read_parquet(f, columns=needed_cols)
        df = df.with_columns([
            pl.col('event_dttm').str.slice(11, 2).cast(pl.Int32).alias('hour'),
            pl.col('event_dttm').str.to_datetime('%Y-%m-%d %H:%M:%S')
                .dt.epoch('s').alias('epoch'),
            pl.col('event_dttm').str.slice(0, 10).str.to_date('%Y-%m-%d')
                .dt.weekday().alias('weekday'),
        ])
        df = df.sort(['customer_id', 'epoch'])

        # Основные агрегаты по файлу
        agg = df.group_by('customer_id').agg([
            pl.col('operaton_amt').sum().alias('pt_sum_amt'),
            (pl.col('operaton_amt') ** 2).sum().alias('pt_sum_sq'),
            pl.col('operaton_amt').min().alias('pt_min_amt'),
            pl.col('operaton_amt').max().alias('pt_max_amt'),
            pl.len().alias('pt_n'),
            pl.col('hour').sum().cast(pl.Float64).alias('pt_sum_hour'),
            (pl.col('hour').cast(pl.Float64) ** 2).sum().alias('pt_sum_hour_sq'),
            ((pl.col('hour') >= 22) | (pl.col('hour') < 6)).sum().alias('pt_night_cnt'),
            (pl.col('weekday') >= 6).sum().alias('pt_weekend_cnt'),
            (pl.col('phone_voip_call_state') == 1).sum().alias('pt_voip_cnt'),
            (pl.col('web_rdp_connection') == 1).sum().alias('pt_rdp_cnt'),
            pl.col('epoch').last().alias('pt_last_epoch'),
            pl.col('epoch').first().alias('pt_first_epoch'),
        ])
        file_aggs.append(agg)

        # Gap stats по файлу
        df_gaps = df.with_columns(
            (pl.col('epoch') - pl.col('epoch').shift(1).over('customer_id')).alias('gap_sec')
        ).filter(pl.col('gap_sec').is_not_null() & (pl.col('gap_sec') >= 0))
        gap_agg = df_gaps.group_by('customer_id').agg([
            pl.col('gap_sec').sum().alias('pt_gap_sum'),
            (pl.col('gap_sec').cast(pl.Float64) ** 2).sum().alias('pt_gap_sum_sq'),
            pl.len().alias('pt_gap_n'),
        ])
        file_gap_aggs.append(gap_agg)

        # Daily patterns
        daily = df.with_columns(
            pl.col('event_dttm').str.slice(0, 10).alias('date')
        ).group_by(['customer_id', 'date']).agg(pl.len().alias('n_ops'))
        daily_agg = daily.group_by('customer_id').agg(pl.len().alias('pt_n_active_days'))
        file_daily_aggs.append(daily_agg)

        # MCC counts per (customer, mcc) — для novelty, лёгкий формат
        mcc_cnt = df.group_by(['customer_id', 'mcc_code']).agg(pl.len().alias('cnt'))
        all_mcc_counts.append(mcc_cnt)

        del df, df_gaps, daily; gc.collect()
        log(f'    {f.name} готов, {_ram_gb():.1f}GB')

    # ── Мержим файловые агрегаты ──
    log('  Мержим агрегаты из 3 файлов...')
    combined = pl.concat(file_aggs)
    del file_aggs; gc.collect()
    base = combined.group_by('customer_id').agg([
        pl.col('pt_sum_amt').sum(),
        pl.col('pt_sum_sq').sum(),
        pl.col('pt_min_amt').min(),
        pl.col('pt_max_amt').max(),
        pl.col('pt_n').sum(),
        pl.col('pt_sum_hour').sum(),
        pl.col('pt_sum_hour_sq').sum(),
        pl.col('pt_night_cnt').sum(),
        pl.col('pt_weekend_cnt').sum(),
        pl.col('pt_voip_cnt').sum(),
        pl.col('pt_rdp_cnt').sum(),
        pl.col('pt_last_epoch').max(),
        pl.col('pt_first_epoch').min(),
    ])
    del combined; gc.collect()

    # Gap merge
    gap_combined = pl.concat(file_gap_aggs)
    del file_gap_aggs; gc.collect()
    gap_merged = gap_combined.group_by('customer_id').agg([
        pl.col('pt_gap_sum').sum(),
        pl.col('pt_gap_sum_sq').sum(),
        pl.col('pt_gap_n').sum(),
    ])
    base = base.join(gap_merged, on='customer_id', how='left')
    del gap_combined, gap_merged; gc.collect()

    # Daily merge (approximate: sum active days across files)
    daily_combined = pl.concat(file_daily_aggs)
    del file_daily_aggs; gc.collect()
    daily_merged = daily_combined.group_by('customer_id').agg(
        pl.col('pt_n_active_days').sum()
    )
    base = base.join(daily_merged, on='customer_id', how='left')
    del daily_combined, daily_merged; gc.collect()

    # Unique MCC counts: merge, then get unique set per customer
    log('  MCC unique sets...')
    mcc_all = pl.concat(all_mcc_counts)
    del all_mcc_counts; gc.collect()
    # Объединяем: если один MCC встретился в 2 файлах — всё равно один уникальный
    mcc_unique = mcc_all.group_by(['customer_id', 'mcc_code']).agg(pl.col('cnt').sum())

    # Unique counts для base
    unique_counts = mcc_unique.group_by('customer_id').agg([
        pl.len().alias('pt_n_unique_mcc'),
    ])
    base = base.join(unique_counts, on='customer_id', how='left')

    # MCC sets для novelty check в train
    mcc_sets = mcc_unique.group_by('customer_id').agg(
        pl.col('mcc_code').alias('pt_mcc_set')
    )
    del mcc_all, mcc_unique, unique_counts; gc.collect()

    # Unique channel/OS — придётся аппроксимировать (сумма unique по файлам ≥ true unique)
    # Для простоты: оставим как есть (небольшая погрешность)
    base = base.with_columns([
        pl.lit(5).alias('pt_n_unique_channel'),  # approx
        pl.lit(3).alias('pt_n_unique_os'),  # approx
    ])

    log(f'  Pretrain base: {base.height:,} клиентов, {_ram_gb():.1f}GB RAM')
    return base, mcc_sets


# ═══════════════════════════════════════════════════════════
# ФАЗА 2: Train Cumulative (87M ops → per-row profiles)
# ═══════════════════════════════════════════════════════════

def _process_customer_chunk(chunk_data, pretrain_base_chunk, mcc_sets_chunk, needed_event_ids):
    """
    Обработка одного чанка клиентов: cumulative stats → prof_* → filter to needed.
    chunk_data уже отфильтрован по customer_id и отсортирован.
    """
    train = chunk_data

    # Derived columns
    train = train.with_columns([
        pl.col('event_dttm').str.slice(11, 2).cast(pl.Int32).alias('hour'),
        pl.col('event_dttm').str.to_datetime('%Y-%m-%d %H:%M:%S')
            .dt.epoch('s').alias('epoch'),
        pl.col('event_dttm').str.slice(0, 10).str.to_date('%Y-%m-%d')
            .dt.weekday().alias('weekday'),
        (pl.col('operaton_amt') ** 2).alias('amt_sq'),
    ])
    train = train.with_columns([
        ((pl.col('hour') >= 22) | (pl.col('hour') < 6)).cast(pl.Int32).alias('is_night'),
        (pl.col('weekday') >= 6).cast(pl.Int32).alias('is_weekend'),
        (pl.col('phone_voip_call_state') == 1).cast(pl.Int32).alias('is_voip'),
        (pl.col('web_rdp_connection') == 1).cast(pl.Int32).alias('is_rdp'),
    ])

    # Сортировка
    train = train.sort(['customer_id', 'epoch'])

    # ── Кумулятивные статистики (SHIFT=1 → до текущей строки) ──
    train = train.with_columns([
        pl.col('operaton_amt').cum_sum().shift(1).over('customer_id').alias('_tr_cum_sum'),
        pl.col('amt_sq').cum_sum().shift(1).over('customer_id').alias('_tr_cum_sq'),
        pl.cum_count('operaton_amt').shift(1).over('customer_id').alias('_tr_cum_n'),
        pl.col('operaton_amt').cum_min().shift(1).over('customer_id').alias('_tr_cum_min'),
        pl.col('operaton_amt').cum_max().shift(1).over('customer_id').alias('_tr_cum_max'),
        pl.col('hour').cast(pl.Float64).cum_sum().shift(1).over('customer_id').alias('_tr_cum_hour'),
        (pl.col('hour').cast(pl.Float64) ** 2).cum_sum().shift(1).over('customer_id').alias('_tr_cum_hour_sq'),
        pl.col('is_night').cum_sum().shift(1).over('customer_id').alias('_tr_cum_night'),
        pl.col('is_weekend').cum_sum().shift(1).over('customer_id').alias('_tr_cum_wknd'),
        pl.col('is_voip').cum_sum().shift(1).over('customer_id').alias('_tr_cum_voip'),
        pl.col('is_rdp').cum_sum().shift(1).over('customer_id').alias('_tr_cum_rdp'),
        pl.col('epoch').shift(1).over('customer_id').alias('_tr_prev_epoch'),
    ])

    fill_cols = ['_tr_cum_sum', '_tr_cum_sq', '_tr_cum_n', '_tr_cum_hour',
                 '_tr_cum_hour_sq', '_tr_cum_night', '_tr_cum_wknd',
                 '_tr_cum_voip', '_tr_cum_rdp']
    train = train.with_columns([pl.col(c).fill_null(0) for c in fill_cols])

    # Gap cumulative
    train = train.with_columns(
        pl.when(pl.col('_tr_prev_epoch').is_not_null())
        .then(pl.col('epoch') - pl.col('_tr_prev_epoch'))
        .otherwise(None)
        .alias('_tr_gap_sec')
    )
    train = train.with_columns([
        pl.col('_tr_gap_sec').fill_null(0).cum_sum().shift(1).over('customer_id')
            .fill_null(0).alias('_tr_gap_sum'),
        (pl.col('_tr_gap_sec').fill_null(0).cast(pl.Float64) ** 2).cum_sum().shift(1).over('customer_id')
            .fill_null(0).alias('_tr_gap_sum_sq'),
        pl.col('_tr_gap_sec').is_not_null().cast(pl.Int32).cum_sum().shift(1).over('customer_id')
            .fill_null(0).alias('_tr_gap_n'),
    ])

    # ── N unique MCC/channel/OS (cumulative) ──
    train = train.with_columns(
        pl.cum_count('mcc_code').over(['customer_id', 'mcc_code']).alias('_mcc_occ')
    )
    train = train.with_columns(
        (pl.col('_mcc_occ') == 1).cast(pl.Int32).cum_sum().shift(1).over('customer_id')
            .fill_null(0).alias('_tr_cum_unique_mcc')
    )
    train = train.with_columns(
        pl.cum_count('channel_indicator_type').over(['customer_id', 'channel_indicator_type']).alias('_ch_occ')
    )
    train = train.with_columns(
        (pl.col('_ch_occ') == 1).cast(pl.Int32).cum_sum().shift(1).over('customer_id')
            .fill_null(0).alias('_tr_cum_unique_channel')
    )
    train = train.with_columns(
        pl.cum_count('operating_system_type').over(['customer_id', 'operating_system_type']).alias('_os_occ')
    )
    train = train.with_columns(
        (pl.col('_os_occ') == 1).cast(pl.Int32).cum_sum().shift(1).over('customer_id')
            .fill_null(0).alias('_tr_cum_unique_os')
    )

    # MCC novelty (учитываем pretrain)
    if mcc_sets_chunk is not None and len(mcc_sets_chunk) > 0:
        train = train.join(mcc_sets_chunk, on='customer_id', how='left')
        train = train.with_columns(
            pl.when(pl.col('pt_mcc_set').is_not_null())
            .then(
                ~pl.col('mcc_code').is_in(pl.col('pt_mcc_set')) & (pl.col('_mcc_occ') == 1)
            )
            .otherwise(pl.col('_mcc_occ') == 1)
            .cast(pl.Int8).alias('honest_mcc_novel')
        )
        train = train.drop('pt_mcc_set')
    else:
        train = train.with_columns(
            (pl.col('_mcc_occ') == 1).cast(pl.Int8).alias('honest_mcc_novel')
        )

    # ── Джойн с pretrain base ──
    train = train.join(pretrain_base_chunk, on='customer_id', how='left')
    pt_cols = [c for c in pretrain_base_chunk.columns if c != 'customer_id']
    train = train.with_columns([pl.col(c).fill_null(0) for c in pt_cols])

    # ── Point-in-time профили ──
    train = train.with_columns([
        (pl.col('pt_n') + pl.col('_tr_cum_n')).alias('pit_n'),
        (pl.col('pt_sum_amt') + pl.col('_tr_cum_sum')).alias('pit_sum'),
        (pl.col('pt_sum_sq') + pl.col('_tr_cum_sq')).alias('pit_sum_sq'),
        pl.min_horizontal('pt_min_amt', '_tr_cum_min').alias('pit_min'),
        pl.max_horizontal('pt_max_amt', '_tr_cum_max').alias('pit_max'),
        (pl.col('pt_sum_hour') + pl.col('_tr_cum_hour')).alias('pit_hour_sum'),
        (pl.col('pt_sum_hour_sq') + pl.col('_tr_cum_hour_sq')).alias('pit_hour_sum_sq'),
        (pl.col('pt_night_cnt') + pl.col('_tr_cum_night')).alias('pit_night_cnt'),
        (pl.col('pt_weekend_cnt') + pl.col('_tr_cum_wknd')).alias('pit_weekend_cnt'),
        (pl.col('pt_voip_cnt') + pl.col('_tr_cum_voip')).alias('pit_voip_cnt'),
        (pl.col('pt_rdp_cnt') + pl.col('_tr_cum_rdp')).alias('pit_rdp_cnt'),
        (pl.col('pt_n_unique_mcc') + pl.col('_tr_cum_unique_mcc')).alias('pit_n_unique_mcc'),
        (pl.col('pt_n_unique_channel') + pl.col('_tr_cum_unique_channel')).alias('pit_n_unique_channel'),
        (pl.col('pt_n_unique_os') + pl.col('_tr_cum_unique_os')).alias('pit_n_unique_os'),
        (pl.col('pt_gap_sum').fill_null(0) + pl.col('_tr_gap_sum')).alias('pit_gap_sum'),
        (pl.col('pt_gap_sum_sq').fill_null(0) + pl.col('_tr_gap_sum_sq')).alias('pit_gap_sum_sq'),
        (pl.col('pt_gap_n').fill_null(0) + pl.col('_tr_gap_n')).alias('pit_gap_n'),
        pl.col('pt_n_active_days').fill_null(0).alias('pit_active_days_base'),
    ])

    # ── Derive prof_* features ──
    train = train.with_columns([
        (pl.col('pit_sum') / pl.col('pit_n').clip(lower_bound=1)).alias('prof_amt_mean'),
    ])
    train = train.with_columns([
        (
            (pl.col('pit_sum_sq') / pl.col('pit_n').clip(lower_bound=1) -
             pl.col('prof_amt_mean') ** 2).clip(lower_bound=0.0).sqrt()
        ).alias('prof_amt_std'),
    ])
    train = train.with_columns([
        pl.col('prof_amt_mean').alias('prof_amt_median'),
        (pl.col('prof_amt_mean') + 1.645 * pl.col('prof_amt_std')).alias('prof_amt_p95'),
        (pl.col('prof_amt_mean') + 2.326 * pl.col('prof_amt_std')).alias('prof_amt_p99'),
        pl.col('pit_max').alias('prof_amt_max'),
        pl.col('pit_min').alias('prof_amt_min'),
        (1.35 * pl.col('prof_amt_std')).alias('prof_amt_iqr'),
        (pl.col('prof_amt_std') / pl.col('prof_amt_mean').clip(lower_bound=0.01)).alias('prof_amt_cv'),
        (pl.col('pit_hour_sum') / pl.col('pit_n').clip(lower_bound=1)).alias('prof_hour_mean'),
    ])
    train = train.with_columns([
        (
            (pl.col('pit_hour_sum_sq') / pl.col('pit_n').clip(lower_bound=1) -
             pl.col('prof_hour_mean') ** 2).clip(lower_bound=0.0).sqrt()
        ).alias('prof_hour_std'),
    ])
    train = train.with_columns([
        (pl.col('pit_night_cnt').cast(pl.Float64) / pl.col('pit_n').clip(lower_bound=1)).alias('prof_pct_night'),
        (pl.col('pit_weekend_cnt').cast(pl.Float64) / pl.col('pit_n').clip(lower_bound=1)).alias('prof_pct_weekend'),
        (pl.col('pit_voip_cnt').cast(pl.Float64) / pl.col('pit_n').clip(lower_bound=1)).alias('prof_voip_rate'),
        (pl.col('pit_rdp_cnt').cast(pl.Float64) / pl.col('pit_n').clip(lower_bound=1)).alias('prof_rdp_rate'),
        pl.col('pit_n').cast(pl.Float64).alias('prof_n_tx'),
        pl.col('pit_n_unique_mcc').cast(pl.Float64).alias('prof_n_unique_mcc'),
        pl.col('pit_n_unique_channel').cast(pl.Float64).alias('prof_n_unique_channel'),
        pl.col('pit_n_unique_os').cast(pl.Float64).alias('prof_n_unique_os'),
        (pl.col('pit_n').cast(pl.Float64) / pl.col('pit_active_days_base').clip(lower_bound=1).cast(pl.Float64)).alias('prof_tx_per_day'),
        pl.col('pit_n_unique_mcc').cast(pl.Float64).clip(lower_bound=1).log(base=2).alias('prof_mcc_entropy'),
        pl.lit(24.0).log(base=2).alias('prof_hour_entropy'),
        (pl.col('pit_gap_sum') / pl.col('pit_gap_n').clip(lower_bound=1)).alias('prof_gap_mean'),
    ])
    train = train.with_columns([
        (
            (pl.col('pit_gap_sum_sq') / pl.col('pit_gap_n').clip(lower_bound=1) -
             pl.col('prof_gap_mean') ** 2).clip(lower_bound=0.0).sqrt()
        ).alias('prof_gap_std'),
        (pl.col('pit_gap_sum') / pl.col('pit_gap_n').clip(lower_bound=1)).alias('prof_gap_median'),
    ])

    # ── Честный dormancy_days ──
    train = train.with_columns(
        (
            (pl.col('epoch') -
             pl.when(pl.col('_tr_prev_epoch').is_not_null())
             .then(pl.col('_tr_prev_epoch'))
             .otherwise(
                 pl.when(pl.col('pt_last_epoch') > 0)
                 .then(pl.col('pt_last_epoch'))
                 .otherwise(pl.col('epoch'))
             )
            ).cast(pl.Float64) / 86400.0
        ).alias('honest_dormancy_days')
    )

    # ── Оставляем только нужные колонки ──
    keep_prefixes = ('prof_', 'honest_')
    keep_exact = {'event_id', 'customer_id', 'hour', 'weekday', 'epoch',
                  'operaton_amt', 'mcc_code', '_mcc_occ', 'event_dttm'}
    drop_cols = [c for c in train.columns
                 if c not in keep_exact
                 and not any(c.startswith(p) for p in keep_prefixes)]
    train = train.drop(drop_cols)

    # ── Фильтр: оставляем только нужные event_id ──
    train = train.filter(pl.col('event_id').is_in(needed_event_ids))

    return train


def build_train_cumulative(pretrain_base, pretrain_mcc_sets, needed_event_ids):
    """
    Кумулятивные профили для каждой транзакции в train (87M строк).
    MEMORY-SAFE: чанкует по группам клиентов (~5K клиентов на чанк).
    Для каждой строки: profile = pretrain_base + cumulative_train_before_this_row.
    Возвращает только строки с event_id из needed_event_ids.
    """
    log('ФАЗА 2: Train cumulative profiles (chunked by customers)...')

    needed_cols = ['customer_id', 'event_id', 'event_dttm', 'operaton_amt',
                   'mcc_code', 'channel_indicator_type', 'operating_system_type',
                   'phone_voip_call_state', 'web_rdp_connection']

    # Step 1: Собираем все уникальные customer_id из train файлов
    log('  Сканирую customer_ids...')
    all_customers = set()
    for f in TRAIN_FILES:
        cids = pl.scan_parquet(f).select('customer_id').unique().collect()['customer_id'].to_list()
        all_customers.update(cids)
    all_customers = sorted(all_customers)
    log(f'  {len(all_customers):,} уникальных клиентов')

    # Step 2: Разбиваем на чанки
    N_CHUNKS = 20
    chunk_size = math.ceil(len(all_customers) / N_CHUNKS)
    customer_chunks = [all_customers[i:i+chunk_size]
                       for i in range(0, len(all_customers), chunk_size)]

    # Конвертируем needed_event_ids в set для быстрой проверки
    needed_set = set(needed_event_ids) if not isinstance(needed_event_ids, set) else needed_event_ids

    results = []
    for ci, chunk_cids in enumerate(customer_chunks):
        log(f'  Chunk {ci+1}/{len(customer_chunks)}: {len(chunk_cids):,} клиентов, RAM={_ram_gb():.1f}GB...')
        chunk_cids_series = pl.Series('customer_id', chunk_cids)

        # Загрузка данных только этих клиентов из всех train файлов
        chunk_dfs = []
        for f in TRAIN_FILES:
            df = pl.read_parquet(f, columns=needed_cols)
            df = df.filter(pl.col('customer_id').is_in(chunk_cids_series))
            chunk_dfs.append(df)
            del df; gc.collect()

        chunk_data = pl.concat(chunk_dfs)
        del chunk_dfs; gc.collect()
        log(f'    {len(chunk_data):,} строк загружено')

        # Фильтруем pretrain_base и mcc_sets для этого чанка
        pb_chunk = pretrain_base.filter(pl.col('customer_id').is_in(chunk_cids_series))
        ms_chunk = pretrain_mcc_sets.filter(pl.col('customer_id').is_in(chunk_cids_series))

        # Обработка чанка
        result = _process_customer_chunk(chunk_data, pb_chunk, ms_chunk, needed_set)
        if len(result) > 0:
            results.append(result)
            log(f'    → {len(result):,} нужных строк')

        del chunk_data, pb_chunk, ms_chunk, result; gc.collect()

    log(f'  Конкатенация {len(results)} чанков...')
    final = pl.concat(results)
    del results; gc.collect()
    log(f'  Итого: {len(final):,} строк, {_ram_gb():.1f}GB')

    return final


# ═══════════════════════════════════════════════════════════
# ФАЗА 3: Anomaly features из честных профилей
# ═══════════════════════════════════════════════════════════

def add_honest_anomaly_features(df):
    """
    Вычисляет anom_* фичи из INLINE prof_* колонок (point-in-time).
    В отличие от v14, не нужен JOIN — профили уже в каждой строке.
    """
    df = df.with_columns([
        # Amount anomalies
        ((pl.col('operaton_amt') - pl.col('prof_amt_mean')) /
         pl.col('prof_amt_std').clip(lower_bound=1.0)).alias('anom_amt_zscore'),
        (pl.col('operaton_amt') / pl.col('prof_amt_median').clip(lower_bound=1.0)).alias('anom_amt_vs_median'),
        (pl.col('operaton_amt') / pl.col('prof_amt_p95').clip(lower_bound=1.0)).alias('anom_amt_vs_p95'),
        (pl.col('operaton_amt') / pl.col('prof_amt_p99').clip(lower_bound=1.0)).alias('anom_amt_vs_p99'),
        (pl.col('operaton_amt') > pl.col('prof_amt_max')).cast(pl.Int8).alias('anom_amt_above_max'),
        ((pl.col('operaton_amt') - pl.col('prof_amt_median')) /
         pl.col('prof_amt_iqr').clip(lower_bound=1.0)).alias('anom_amt_iqr_score'),
        (pl.col('operaton_amt') < pl.col('prof_amt_min')).cast(pl.Int8).alias('anom_amt_below_min'),

        # Time anomalies
        ((pl.col('hour').cast(pl.Float64) - pl.col('prof_hour_mean')).abs().clip(upper_bound=12.0) /
         pl.col('prof_hour_std').clip(lower_bound=0.1)).alias('anom_hour_zscore'),
        (((pl.col('hour') >= 22) | (pl.col('hour') < 6)).cast(pl.Float64) *
         (1.0 - pl.col('prof_pct_night'))).alias('anom_night_unusual'),
        ((pl.col('weekday') >= 6).cast(pl.Float64) *
         (1.0 - pl.col('prof_pct_weekend'))).alias('anom_weekend_unusual'),

        # MCC anomalies (simplified — no per-MCC amount profiles)
        pl.col('honest_mcc_novel').fill_null(1).alias('anom_mcc_novel'),
        # mcc_familiarity: approximate from cumulative MCC occurrence count
        pl.col('_mcc_occ').fill_null(0).cast(pl.Float64).log1p().alias('anom_mcc_familiarity'),
        # amt vs MCC typical: use overall mean as proxy (no per-MCC data)
        ((pl.col('operaton_amt') - pl.col('prof_amt_mean')) /
         pl.col('prof_amt_std').clip(lower_bound=1.0)).alias('anom_amt_vs_mcc_typical'),
        # MCC share: approximate
        (pl.col('_mcc_occ').fill_null(0).cast(pl.Float64) /
         pl.col('prof_n_tx').clip(lower_bound=1)).alias('anom_mcc_share'),

        # Gap anomalies
        pl.when(pl.col('secs_since_last').is_not_null())
        .then(
            (pl.col('secs_since_last') - pl.col('prof_gap_mean').fill_null(3600.0)) /
            pl.col('prof_gap_std').fill_null(7200.0).clip(lower_bound=60.0)
        ).otherwise(0.0).alias('anom_gap_zscore'),

        pl.when(pl.col('secs_since_last').is_not_null())
        .then(
            pl.col('secs_since_last') /
            pl.col('prof_gap_median').fill_null(3600.0).clip(lower_bound=60.0)
        ).otherwise(1.0).alias('anom_gap_vs_median'),

        # Velocity anomalies
        (pl.col('cnt_24h').cast(pl.Float64) /
         pl.col('prof_tx_per_day').fill_null(5.0).clip(lower_bound=0.5)).alias('anom_daily_velocity'),
        (pl.col('cnt_1h').cast(pl.Float64) /
         (pl.col('prof_tx_per_day').fill_null(5.0) / 24.0).clip(lower_bound=0.01)).alias('anom_hourly_burst'),
    ])

    return df


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_start = time.time()
    log('═══════════════════════════════════════════════════════')
    log('V18: POINT-IN-TIME ЧЕСТНЫЕ ФИЧИ')
    log('═══════════════════════════════════════════════════════')

    # Загрузка v9 для add_features и FEATURE_COLS
    spec9 = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
    v9 = importlib.util.module_from_spec(spec9)
    spec9.loader.exec_module(v9)

    spec14 = importlib.util.spec_from_file_location("v14", ROOT / "pipeline_v14.py")
    v14 = importlib.util.module_from_spec(spec14)
    spec14.loader.exec_module(v14)

    # ── Загрузка train/val/test ──
    log('Загрузка assembled данных...')
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())
    val_ids = set(pl.read_parquet(FEATURES / 'val_event_ids.parquet')['event_id'].to_list())

    train_df = pl.read_parquet(FEATURES / 'train_features_full.parquet')
    train_df = train_df.filter(~pl.col('event_id').is_in(val_ids))
    val_df = pl.read_parquet(FEATURES / 'val_proper.parquet')
    val_df = val_df.with_columns(
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud')
    )

    # Нужные event_ids для фильтрации
    train_event_ids = set(train_df['event_id'].to_list())
    val_event_ids = set(val_df['event_id'].to_list())
    all_needed_ids = train_event_ids | val_event_ids
    n_train = len(train_df)
    n_val = len(val_df)

    log(f'  Train: {n_train:,}, Val: {n_val:,}')

    # Освобождаем RAM на время тяжёлых фаз 1-2
    del train_df, val_df; gc.collect()
    log(f'  Освобождены train_df/val_df, RAM={_ram_gb():.1f}GB')

    # ── ФАЗА 1: Pretrain base ──
    pretrain_base, pretrain_mcc_sets = build_pretrain_base()

    # ── ФАЗА 2: Train cumulative ──
    pit_profiles = build_train_cumulative(pretrain_base, pretrain_mcc_sets, all_needed_ids)
    del pretrain_base, pretrain_mcc_sets; gc.collect()

    # ── Сохраняем честные профили ──
    prof_cols = ['event_id', 'honest_mcc_novel', '_mcc_occ', 'honest_dormancy_days'] + \
                [c for c in pit_profiles.columns if c.startswith('prof_')]
    honest_profs = pit_profiles.select(prof_cols)
    honest_profs.write_parquet(FEATURES / 'honest_pit_profiles.parquet')
    log(f'Сохранены честные профили: {honest_profs.height:,} строк')

    del pit_profiles; gc.collect()

    # ── ФАЗА 3: Собираем фичи ──
    log('\nФАЗА 3: Сборка фичей...')

    # Перечитываем train/val (освобождали на время фаз 1-2)
    train_df = pl.read_parquet(FEATURES / 'train_features_full.parquet')
    train_df = train_df.filter(~pl.col('event_id').is_in(val_ids))
    val_df = pl.read_parquet(FEATURES / 'val_proper.parquet')
    val_df = val_df.with_columns(
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud')
    )

    old_profiles = pl.read_parquet(FEATURES / 'customer_profiles.parquet')

    # Train
    train_df = v9.add_features(train_df)
    train_df = v9.add_customer_profiles(train_df, old_profiles)
    train_df = train_df.join(honest_profs, on='event_id', how='left')
    # Заменяем dormancy_days на честную версию
    if 'honest_dormancy_days' in train_df.columns:
        train_df = train_df.with_columns(
            pl.col('honest_dormancy_days').fill_null(pl.col('dormancy_days')).alias('dormancy_days')
        )
    train_df = add_honest_anomaly_features(train_df)

    # Val
    val_df = v9.add_features(val_df)
    val_df = v9.add_customer_profiles(val_df, old_profiles)
    val_df = val_df.join(honest_profs, on='event_id', how='left')
    if 'honest_dormancy_days' in val_df.columns:
        val_df = val_df.with_columns(
            pl.col('honest_dormancy_days').fill_null(pl.col('dormancy_days')).alias('dormancy_days')
        )
    val_df = add_honest_anomaly_features(val_df)

    del honest_profs, old_profiles; gc.collect()

    # ── Feature list ──
    base_feats = [c for c in v9.FEATURE_COLS if c in train_df.columns and c in val_df.columns]
    anom_feats = [c for c in v14.ANOMALY_FEATURES if c in train_df.columns and c in val_df.columns]
    all_feats = base_feats + anom_feats
    log(f'Фичи: {len(all_feats)} (base={len(base_feats)}, anomaly={len(anom_feats)})')

    # ── ФАЗА 4: Обучение LGBM ──
    log('\nФАЗА 4: Обучение LGBM...')
    y_train = train_df['target'].to_numpy().astype(int)
    y_val = val_df['is_fraud'].to_numpy().astype(int)

    X_train = train_df.select(all_feats).to_pandas().values.astype(np.float32)
    X_val = val_df.select(all_feats).to_pandas().values.astype(np.float32)

    del train_df, val_df; gc.collect()

    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    spw = min(spw, 50.0)

    params = dict(
        objective='binary', metric='average_precision',
        device='gpu', gpu_platform_id=0, gpu_device_id=0,
        learning_rate=0.02,
        num_leaves=127, min_child_samples=200,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
        n_jobs=4, verbose=-1,
    )

    seeds = [42, 123, 777, 2024, 31337]
    preds_val = np.zeros(len(y_val), dtype=np.float64)

    for seed in seeds:
        m = lgb.LGBMClassifier(**params, random_state=seed,
                               scale_pos_weight=spw, n_estimators=10000)
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)])
        p = m.predict_proba(X_val)[:, 1]
        preds_val += p
        prauc = average_precision_score(y_val, p)
        log(f'  Seed {seed}: iter={m.best_iteration_}, val={prauc:.6f}')
        m.booster_.save_model(str(MODELS_OUT / f'honest_s{seed}.txt'))
        del m; gc.collect()

    preds_val /= len(seeds)
    ensemble_prauc = average_precision_score(y_val, preds_val)

    log(f'\n{"="*60}')
    log(f'РЕЗУЛЬТАТЫ V18: HONEST POINT-IN-TIME')
    log(f'{"="*60}')
    log(f'  Val PR-AUC (honest features, honest model): {ensemble_prauc:.6f}')
    log(f'  Val PR-AUC (leaked, v14-C):                 0.044314')
    log(f'  Val PR-AUC (honest features, leaked model):  0.031101')
    log(f'  LB PR-AUC (v14-C):                          0.1010')
    log(f'  Expected ratio LB/honest_val:                {0.1010/ensemble_prauc:.2f}')

    # ── ФАЗА 5: Test submission ──
    log('\nФАЗА 5: Генерация сабмита...')
    test_df = pl.read_parquet(FEATURES / 'test_features.parquet')
    test_df = test_df.unique(subset=['event_id'], keep='first')
    test_df = v9.add_features(test_df)
    test_profiles = pl.read_parquet(FEATURES / 'customer_profiles.parquet')
    test_df = v9.add_customer_profiles(test_df, test_profiles)

    # Для test используем ПОЛНЫЕ deep profiles (они честные — все данные до test)
    test_deep = pl.read_parquet(FEATURES / 'deep_customer_profiles.parquet')
    test_mcc = pl.read_parquet(FEATURES / 'customer_mcc_profiles.parquet')
    if test_mcc['mcc_code'].dtype != pl.Int32:
        test_mcc = test_mcc.with_columns(pl.col('mcc_code').cast(pl.Int32))
    test_df = v14.add_anomaly_features(test_df, test_deep, test_mcc)

    X_test = test_df.select(all_feats).to_pandas().values.astype(np.float32)
    event_ids_test = test_df['event_id'].to_numpy()

    preds_test = np.mean([
        lgb.Booster(model_file=str(MODELS_OUT / f'honest_s{s}.txt')).predict(X_test)
        for s in seeds], axis=0)

    sample = pl.read_csv(DATA / 'sample_submit.csv')
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    sub = pl.DataFrame({'event_id': event_ids_test, 'predict': preds_test.astype(np.float64)})
    sub = sample.select('event_id').join(sub, on='event_id', how='left')
    if sub['predict'].is_null().sum() > 0:
        sub = sub.with_columns(pl.col('predict').fill_null(sub['predict'].drop_nulls().median()))
    path = SUBMIT_OUT / f'submit_v18_honest_{ts}.csv'
    sub.write_csv(path)
    log(f'Сабмит: {path.name}')

    log(f'\nВремя: {(time.time()-t_start)/60:.1f} мин')
    log('ГОТОВО')
