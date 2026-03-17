"""
Pipeline v14: Per-Customer Anomaly Detection.

Идея: фрод = аномалия для конкретного клиента.
Обучаем "норму" каждого из 100K клиентов на ВСЕХ 177M транзакциях (pretrain+train).
Потом для каждой транзакции считаем: насколько она отличается от нормы этого клиента.

Фазы:
1. Глубокие per-customer профили из 177M строк
2. Anomaly features для train/val/test
3. LGBM с anomaly features (+ blend с v10)
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from pathlib import Path
from datetime import datetime
import gc, json, time, os, math

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_OUT  = ROOT / 'models_v14'
SUBMIT_OUT  = ROOT / 'submissions'
RAW_TRAIN   = ROOT / 'Pre-train_Train'
RAW_TEST    = ROOT / 'Pre-test_Test'

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


# ═══════════════════════════════════════════════════════════
# ФАЗА 1: Per-Customer Profiles из 177M строк
# ═══════════════════════════════════════════════════════════

def parse_datetime(df):
    """Parse event_dttm string to extract temporal features."""
    return df.with_columns([
        pl.col('event_dttm').str.slice(11, 2).cast(pl.Int32).alias('hour'),
        pl.col('event_dttm').str.slice(0, 10).str.to_date('%Y-%m-%d').dt.weekday().alias('weekday'),
        pl.col('event_dttm').str.to_datetime('%Y-%m-%d %H:%M:%S').alias('dt'),
    ])


def build_profiles_from_file(filepath):
    """Compute per-customer aggregates from one parquet file."""
    log(f'  Reading {filepath.name}...')
    df = pl.read_parquet(filepath)
    df = parse_datetime(df)

    # Amount stats per customer
    amt_stats = df.group_by('customer_id').agg([
        # Amount distribution
        pl.col('operaton_amt').mean().alias('_amt_mean'),
        pl.col('operaton_amt').std().alias('_amt_std'),
        pl.col('operaton_amt').median().alias('_amt_median'),
        pl.col('operaton_amt').quantile(0.10).alias('_amt_p10'),
        pl.col('operaton_amt').quantile(0.25).alias('_amt_p25'),
        pl.col('operaton_amt').quantile(0.75).alias('_amt_p75'),
        pl.col('operaton_amt').quantile(0.90).alias('_amt_p90'),
        pl.col('operaton_amt').quantile(0.95).alias('_amt_p95'),
        pl.col('operaton_amt').quantile(0.99).alias('_amt_p99'),
        pl.col('operaton_amt').max().alias('_amt_max'),
        pl.col('operaton_amt').min().alias('_amt_min'),

        # Count
        pl.len().alias('_n_tx'),

        # Hour distribution (mean + std)
        pl.col('hour').mean().alias('_hour_mean'),
        pl.col('hour').std().alias('_hour_std'),

        # Weekday distribution
        pl.col('weekday').mean().alias('_weekday_mean'),

        # Night ratio (22-6)
        ((pl.col('hour') >= 22) | (pl.col('hour') < 6)).mean().alias('_pct_night'),

        # Weekend ratio
        (pl.col('weekday') >= 6).mean().alias('_pct_weekend'),

        # Unique counts
        pl.col('mcc_code').n_unique().alias('_n_unique_mcc'),
        pl.col('channel_indicator_type').n_unique().alias('_n_unique_channel'),
        pl.col('operating_system_type').n_unique().alias('_n_unique_os'),

        # Security flags
        (pl.col('phone_voip_call_state') == 1).sum().alias('_voip_sum'),
        (pl.col('web_rdp_connection') == 1).sum().alias('_rdp_sum'),
        (pl.col('developer_tools') == '1').sum().alias('_devtools_sum'),

        # MCC entropy prep: list of MCC codes
        pl.col('mcc_code').alias('_mcc_list'),

        # Sorted timestamps for gap calculation
        pl.col('dt').sort().alias('_dt_sorted'),

        # Hour histogram (counts per hour bucket)
        pl.col('hour').alias('_hour_list'),
    ])

    return amt_stats


def compute_mcc_entropy(mcc_list):
    """Compute Shannon entropy of MCC distribution."""
    if mcc_list is None or len(mcc_list) == 0:
        return 0.0
    from collections import Counter
    counts = Counter(mcc_list)
    total = len(mcc_list)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_hour_entropy(hour_list):
    """Compute Shannon entropy of hour distribution."""
    if hour_list is None or len(hour_list) == 0:
        return 0.0
    from collections import Counter
    counts = Counter(hour_list)
    total = len(hour_list)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_gap_stats(dt_sorted):
    """Compute gap statistics from sorted timestamps."""
    if dt_sorted is None or len(dt_sorted) < 2:
        return (0.0, 0.0, 0.0)
    # Convert to seconds
    gaps = []
    for i in range(1, len(dt_sorted)):
        if dt_sorted[i] is not None and dt_sorted[i-1] is not None:
            diff = (dt_sorted[i] - dt_sorted[i-1]).total_seconds()
            if diff >= 0:
                gaps.append(diff)
    if not gaps:
        return (0.0, 0.0, 0.0)
    gaps = np.array(gaps)
    return (float(np.median(gaps)), float(np.percentile(gaps, 95)), float(np.std(gaps)))


def compute_mcc_amt_profiles(filepath):
    """Compute per-customer-per-MCC amount stats."""
    df = pl.read_parquet(filepath, columns=['customer_id', 'mcc_code', 'operaton_amt'])
    mcc_stats = df.group_by(['customer_id', 'mcc_code']).agg([
        pl.col('operaton_amt').mean().alias('mcc_amt_mean'),
        pl.col('operaton_amt').std().alias('mcc_amt_std'),
        pl.col('operaton_amt').median().alias('mcc_amt_median'),
        pl.len().alias('mcc_n_tx'),
    ])
    return mcc_stats


def compute_daily_patterns(filepath):
    """Compute per-customer daily transaction count stats."""
    df = pl.read_parquet(filepath, columns=['customer_id', 'event_dttm'])
    df = df.with_columns(
        pl.col('event_dttm').str.slice(0, 10).alias('date')
    )
    daily = df.group_by(['customer_id', 'date']).agg(pl.len().alias('day_n'))
    daily_stats = daily.group_by('customer_id').agg([
        pl.col('day_n').mean().alias('_daily_mean'),
        pl.col('day_n').std().alias('_daily_std'),
        pl.col('day_n').median().alias('_daily_median'),
        pl.col('day_n').max().alias('_daily_max'),
        pl.col('day_n').quantile(0.95).alias('_daily_p95'),
        pl.len().alias('_n_active_days'),
    ])
    return daily_stats


def build_all_profiles():
    """Build comprehensive per-customer profiles from ALL data."""
    log('=== ФАЗА 1: СТРОИМ ПРОФИЛИ ИЗ 177M СТРОК ===')

    profile_path = FEATURES_IN / 'deep_customer_profiles.parquet'
    mcc_profile_path = FEATURES_IN / 'customer_mcc_profiles.parquet'

    if profile_path.exists() and mcc_profile_path.exists():
        log(f'Профили уже есть, загружаем из кэша')
        profs = pl.read_parquet(profile_path)
        mcc_profs = pl.read_parquet(mcc_profile_path)
        if mcc_profs['mcc_code'].dtype != pl.Int32:
            mcc_profs = mcc_profs.with_columns(pl.col('mcc_code').cast(pl.Int32))
        return profs, mcc_profs

    files = sorted([
        RAW_TRAIN / 'pretrain_part_1.parquet',
        RAW_TRAIN / 'pretrain_part_2.parquet',
        RAW_TRAIN / 'pretrain_part_3.parquet',
        RAW_TRAIN / 'train_part_1.parquet',
        RAW_TRAIN / 'train_part_2.parquet',
        RAW_TRAIN / 'train_part_3.parquet',
    ])

    # ── Step 1: MCC-Amount profiles (lightweight, separate pass) ──
    log('Считаем MCC-Amount профили...')
    all_mcc = []
    for f in files:
        log(f'  MCC profiles from {f.name}...')
        mcc = compute_mcc_amt_profiles(f)
        all_mcc.append(mcc)
        gc.collect()

    mcc_combined = pl.concat(all_mcc)
    del all_mcc; gc.collect()

    # Aggregate across files
    mcc_profiles = mcc_combined.group_by(['customer_id', 'mcc_code']).agg([
        # Weighted mean by count
        (pl.col('mcc_amt_mean') * pl.col('mcc_n_tx')).sum().alias('_weighted_sum'),
        pl.col('mcc_n_tx').sum().alias('mcc_n_tx_total'),
    ]).with_columns(
        (pl.col('_weighted_sum') / pl.col('mcc_n_tx_total')).alias('mcc_amt_mean')
    ).drop('_weighted_sum')

    # Also compute per-customer-MCC std properly: need raw data again or approximate
    # For now, just use mean - std will be computed from deviations

    # Ensure mcc_code is i32 to match main data
    if mcc_profiles['mcc_code'].dtype != pl.Int32:
        mcc_profiles = mcc_profiles.with_columns(pl.col('mcc_code').cast(pl.Int32))

    mcc_profiles.write_parquet(mcc_profile_path)
    log(f'MCC profiles: {mcc_profiles.height:,} rows (customer×MCC pairs)')
    del mcc_combined; gc.collect()

    # ── Step 2: Daily pattern stats ──
    log('Считаем daily patterns...')
    all_daily = []
    for f in files:
        log(f'  Daily from {f.name}...')
        d = compute_daily_patterns(f)
        all_daily.append(d)
        gc.collect()

    # Merge daily stats across files (approximate: take weighted average)
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

    log(f'Daily profiles: {daily_profiles.height:,} customers')

    # ── Step 3: Main profiles (amounts, time, etc.) ──
    log('Считаем основные профили...')

    # We need to merge aggregates from multiple files carefully
    # For percentiles/means: accumulate sums and counts, compute final stats
    # Simpler approach: read all data in polars lazy scan

    # Actually, let's use Polars lazy scan for the heavy aggregation
    log('  Lazy scan all 6 files for main aggregates...')

    all_files = [str(f) for f in files]

    # Main numeric aggregates (Polars can handle this lazily)
    main_agg = (
        pl.scan_parquet(all_files)
        .with_columns([
            pl.col('event_dttm').str.slice(11, 2).cast(pl.Int32).alias('hour'),
            pl.col('event_dttm').str.slice(0, 10).str.to_date('%Y-%m-%d').dt.weekday().alias('weekday'),
        ])
        .group_by('customer_id')
        .agg([
            # Amount distribution
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

            # Count
            pl.len().alias('prof_n_tx'),

            # Hour stats
            pl.col('hour').mean().alias('prof_hour_mean'),
            pl.col('hour').std().alias('prof_hour_std'),

            # Night/weekend
            ((pl.col('hour') >= 22) | (pl.col('hour') < 6)).mean().alias('prof_pct_night'),
            (pl.col('weekday') >= 6).mean().alias('prof_pct_weekend'),

            # Unique counts
            pl.col('mcc_code').n_unique().alias('prof_n_unique_mcc'),
            pl.col('channel_indicator_type').n_unique().alias('prof_n_unique_channel'),
            pl.col('operating_system_type').n_unique().alias('prof_n_unique_os'),

            # Security flags
            (pl.col('phone_voip_call_state') == 1).sum().alias('prof_voip_sum'),
            (pl.col('web_rdp_connection') == 1).sum().alias('prof_rdp_sum'),

            # Amount IQR
            (pl.col('operaton_amt').quantile(0.75) - pl.col('operaton_amt').quantile(0.25)).alias('prof_amt_iqr'),
        ])
        .collect()
    )

    log(f'  Main aggregates: {main_agg.height:,} customers')

    # ── Step 4: Hour distribution entropy (need per-customer hour lists) ──
    # Do this per-file and merge
    log('  Computing hour/MCC entropy...')

    hour_counts_all = []
    mcc_counts_all = []
    for f in files:
        log(f'    Entropy from {f.name}...')
        df = pl.read_parquet(f, columns=['customer_id', 'event_dttm', 'mcc_code'])
        df = df.with_columns(
            pl.col('event_dttm').str.slice(11, 2).cast(pl.Int32).alias('hour')
        )
        # Hour counts per customer per hour-bucket
        hc = df.group_by(['customer_id', 'hour']).agg(pl.len().alias('cnt'))
        hour_counts_all.append(hc)
        # MCC counts per customer per MCC
        mc = df.group_by(['customer_id', 'mcc_code']).agg(pl.len().alias('cnt'))
        mcc_counts_all.append(mc)
        del df; gc.collect()

    # Merge hour counts
    hour_counts = pl.concat(hour_counts_all).group_by(['customer_id', 'hour']).agg(
        pl.col('cnt').sum()
    )
    del hour_counts_all; gc.collect()

    # Compute entropy per customer
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

    # Merge MCC counts
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

    # Also compute top MCC per customer (most frequent)
    top_mcc = (
        mcc_counts
        .sort(['customer_id', 'cnt'], descending=[False, True])
        .group_by('customer_id').first()
        .select(['customer_id', pl.col('mcc_code').alias('prof_top_mcc'),
                 pl.col('cnt').alias('prof_top_mcc_cnt')])
    )

    del hour_counts, mcc_counts; gc.collect()

    # ── Step 5: Gap statistics (time between transactions) ──
    log('  Computing transaction gap stats...')
    gap_stats_all = []
    for f in files:
        log(f'    Gaps from {f.name}...')
        df = pl.read_parquet(f, columns=['customer_id', 'event_dttm'])
        df = df.with_columns(
            pl.col('event_dttm').str.to_datetime('%Y-%m-%d %H:%M:%S').alias('dt')
        ).sort(['customer_id', 'dt'])

        # Compute gaps within customer
        df = df.with_columns(
            (pl.col('dt') - pl.col('dt').shift(1).over('customer_id'))
            .dt.total_seconds().alias('gap_sec')
        )

        gaps = df.filter(pl.col('gap_sec').is_not_null() & (pl.col('gap_sec') >= 0)).group_by('customer_id').agg([
            pl.col('gap_sec').median().alias('_gap_median'),
            pl.col('gap_sec').mean().alias('_gap_mean'),
            pl.col('gap_sec').std().alias('_gap_std'),
            pl.col('gap_sec').quantile(0.95).alias('_gap_p95'),
            pl.len().alias('_gap_n'),
        ])
        gap_stats_all.append(gaps)
        del df; gc.collect()

    # Merge gap stats (weighted average)
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

    # ── Step 6: Merge everything ──
    log('  Merging all profiles...')
    profiles = main_agg
    profiles = profiles.join(daily_profiles, on='customer_id', how='left')
    profiles = profiles.join(hour_entropy, on='customer_id', how='left')
    profiles = profiles.join(mcc_entropy, on='customer_id', how='left')
    profiles = profiles.join(top_mcc, on='customer_id', how='left')
    profiles = profiles.join(gap_profiles, on='customer_id', how='left')

    # Compute derived features
    profiles = profiles.with_columns([
        # Voip/RDP rate
        (pl.col('prof_voip_sum') / pl.col('prof_n_tx')).alias('prof_voip_rate'),
        (pl.col('prof_rdp_sum') / pl.col('prof_n_tx')).alias('prof_rdp_rate'),
        # Daily tx rate
        (pl.col('prof_n_tx') / pl.col('n_active_days').clip(lower_bound=1)).alias('prof_tx_per_day'),
        # Amount coefficient of variation
        (pl.col('prof_amt_std') / pl.col('prof_amt_mean').clip(lower_bound=0.01)).alias('prof_amt_cv'),
    ])

    profiles.write_parquet(profile_path)
    log(f'Профили сохранены: {profile_path.name} ({profiles.height:,} клиентов, {len(profiles.columns)} колонок)')
    log(f'  Колонки: {profiles.columns}')

    return profiles, mcc_profiles


def _entropy_from_counts(counts_list):
    """Compute Shannon entropy from a list of counts."""
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
# ФАЗА 2: Anomaly Features
# ═══════════════════════════════════════════════════════════

def add_anomaly_features(df, profiles, mcc_profiles):
    """Add per-customer anomaly features to a transaction dataframe."""

    # Join deep profiles
    df = df.join(profiles, on='customer_id', how='left')

    # ── Amount anomalies ──
    df = df.with_columns([
        # Z-score vs deep profile
        ((pl.col('operaton_amt') - pl.col('prof_amt_mean')) /
         pl.col('prof_amt_std').clip(lower_bound=1.0)).alias('anom_amt_zscore'),

        # Ratio to median
        (pl.col('operaton_amt') / pl.col('prof_amt_median').clip(lower_bound=1.0)).alias('anom_amt_vs_median'),

        # Ratio to p95
        (pl.col('operaton_amt') / pl.col('prof_amt_p95').clip(lower_bound=1.0)).alias('anom_amt_vs_p95'),

        # Ratio to p99
        (pl.col('operaton_amt') / pl.col('prof_amt_p99').clip(lower_bound=1.0)).alias('anom_amt_vs_p99'),

        # Above max ever?
        (pl.col('operaton_amt') > pl.col('prof_amt_max')).cast(pl.Int8).alias('anom_amt_above_max'),

        # IQR-based outlier score
        ((pl.col('operaton_amt') - pl.col('prof_amt_median')) /
         pl.col('prof_amt_iqr').clip(lower_bound=1.0)).alias('anom_amt_iqr_score'),

        # Below min?
        (pl.col('operaton_amt') < pl.col('prof_amt_min')).cast(pl.Int8).alias('anom_amt_below_min'),
    ])

    # ── Time anomalies ──
    # Need hour from the transaction
    if 'hour' not in df.columns:
        df = df.with_columns(
            pl.col('event_dttm').str.slice(11, 2).cast(pl.Int32).alias('hour')
        )

    df = df.with_columns([
        # Hour deviation from customer's mean hour
        (((pl.col('hour') - pl.col('prof_hour_mean')).abs())
         .clip(upper_bound=12.0)  # circular distance
         / pl.col('prof_hour_std').clip(lower_bound=0.1)).alias('anom_hour_zscore'),

        # Is this hour unusual? (night for day-person, etc.)
        ((pl.col('hour') >= 22) | (pl.col('hour') < 6)).cast(pl.Float32).alias('_is_night_tx'),
    ])

    df = df.with_columns([
        # Night anomaly: night tx for person who rarely transacts at night
        (pl.col('_is_night_tx') * (1.0 - pl.col('prof_pct_night'))).alias('anom_night_unusual'),
    ]).drop('_is_night_tx')

    # ── Weekday anomalies ──
    if 'weekday' not in df.columns:
        if 'event_dttm' in df.columns:
            df = df.with_columns(
                pl.col('event_dttm').str.slice(0, 10).str.to_date('%Y-%m-%d').dt.weekday().alias('weekday')
            )

    if 'weekday' in df.columns:
        df = df.with_columns([
            # Weekend tx for weekday person
            ((pl.col('weekday') >= 6).cast(pl.Float32) *
             (1.0 - pl.col('prof_pct_weekend'))).alias('anom_weekend_unusual'),
        ])

    # ── MCC anomaly ──
    if 'mcc_code' in df.columns:
        # Join MCC profiles
        df = df.join(
            mcc_profiles.select(['customer_id', 'mcc_code', 'mcc_amt_mean', 'mcc_n_tx_total']),
            on=['customer_id', 'mcc_code'],
            how='left'
        )

        df = df.with_columns([
            # New MCC for this customer (never seen in 177M history)
            pl.col('mcc_n_tx_total').is_null().cast(pl.Int8).alias('anom_mcc_novel'),

            # How many times customer used this MCC (log)
            pl.col('mcc_n_tx_total').fill_null(0).log1p().alias('anom_mcc_familiarity'),

            # Amount vs customer's typical amount at this MCC
            ((pl.col('operaton_amt') - pl.col('mcc_amt_mean').fill_null(pl.col('prof_amt_mean'))) /
             pl.col('prof_amt_std').clip(lower_bound=1.0)).alias('anom_amt_vs_mcc_typical'),
        ])

        # MCC rarity for this customer (how diverse their MCC usage is)
        df = df.with_columns([
            (pl.col('mcc_n_tx_total').fill_null(0) /
             pl.col('prof_n_tx').clip(lower_bound=1)).alias('anom_mcc_share'),
        ])

    # ── Frequency/Gap anomalies ──
    if 'secs_since_last' in df.columns:
        df = df.with_columns([
            # Gap z-score vs customer profile
            ((pl.col('secs_since_last') - pl.col('prof_gap_mean').fill_null(3600.0)) /
             pl.col('prof_gap_std').fill_null(7200.0).clip(lower_bound=60.0)).alias('anom_gap_zscore'),

            # Ratio to customer's typical gap
            (pl.col('secs_since_last') /
             pl.col('prof_gap_median').fill_null(3600.0).clip(lower_bound=60.0)).alias('anom_gap_vs_median'),
        ])

    # ── Velocity anomalies ──
    if 'cnt_24h' in df.columns:
        df = df.with_columns([
            # Daily count vs customer's typical daily count
            (pl.col('cnt_24h') /
             pl.col('prof_tx_per_day').fill_null(5.0).clip(lower_bound=0.5)).alias('anom_daily_velocity'),
        ])

    if 'cnt_1h' in df.columns and 'prof_tx_per_day' in df.columns:
        df = df.with_columns([
            # Hourly burst: cnt_1h vs customer's average hourly rate
            (pl.col('cnt_1h') /
             (pl.col('prof_tx_per_day').fill_null(5.0) / 24.0).clip(lower_bound=0.01)
            ).alias('anom_hourly_burst'),
        ])

    # ── Composite anomaly score ──
    anom_cols = [c for c in df.columns if c.startswith('anom_')]
    log(f'  Added {len(anom_cols)} anomaly features: {anom_cols}')

    return df


# ═══════════════════════════════════════════════════════════
# ФАЗА 3: Training
# ═══════════════════════════════════════════════════════════

# Import v9 for existing features
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

ANOMALY_FEATURES = [
    'anom_amt_zscore', 'anom_amt_vs_median', 'anom_amt_vs_p95', 'anom_amt_vs_p99',
    'anom_amt_above_max', 'anom_amt_iqr_score', 'anom_amt_below_min',
    'anom_hour_zscore', 'anom_night_unusual', 'anom_weekend_unusual',
    'anom_mcc_novel', 'anom_mcc_familiarity', 'anom_amt_vs_mcc_typical', 'anom_mcc_share',
    'anom_gap_zscore', 'anom_gap_vs_median',
    'anom_daily_velocity', 'anom_hourly_burst',
    # Profile features (also useful directly)
    'prof_amt_cv', 'prof_hour_entropy', 'prof_mcc_entropy',
    'prof_voip_rate', 'prof_rdp_rate', 'prof_tx_per_day',
    'prof_n_unique_mcc', 'prof_n_unique_channel', 'prof_n_unique_os',
    'prof_amt_iqr', 'prof_gap_mean', 'prof_gap_std',
]


def train_model(name, X_train, y_train, X_val, y_val, n_trees=10000, patience=300):
    """Train 5-seed LGBM ensemble."""
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    spw = min(spw, 50.0)

    log(f'\n{"="*60}')
    log(f'Training: {name}')
    log(f'Features: {X_train.shape[1]}, Trees: {n_trees}, spw: {spw:.1f}')
    log(f'{"="*60}')

    preds = np.zeros(len(X_val), dtype=np.float64)
    seed_results = []

    for i, seed in enumerate(seeds):
        m = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed,
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

        model_path = MODELS_OUT / f'{name}_s{seed}.txt'
        m.booster_.save_model(str(model_path))
        del m; gc.collect()

    preds /= len(seeds)
    ensemble_prauc = average_precision_score(y_val, preds)
    log(f'  Ensemble ({name}): val={ensemble_prauc:.6f}')

    return ensemble_prauc, preds, seed_results


def generate_submission(name, feats, profiles, mcc_profiles):
    """Generate submission for an experiment."""
    test_df = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    test_df = test_df.unique(subset=['event_id'], keep='first')
    test_df = v9.add_features(test_df)
    test_df = v9.add_customer_profiles(test_df, profiles)
    test_df = add_anomaly_features(test_df, deep_profiles, mcc_profiles)

    available_feats = [f for f in feats if f in test_df.columns]
    X_test = test_df.select(available_feats).to_pandas().astype(np.float32)
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

    path = SUBMIT_OUT / f'submit_v14_{name}_{ts}.csv'
    sub.write_csv(path)
    log(f'Saved: {path.name}')
    return path


if __name__ == '__main__':
    t_start = time.time()
    log('=== PIPELINE V14: PER-CUSTOMER ANOMALY DETECTION ===')

    # ── Фаза 1: Profiles ──
    deep_profiles, mcc_profiles = build_all_profiles()
    gc.collect()
    log(f'RAM после профилей: {_ram_gb():.1f}GB')

    # ── Load existing features ──
    log('\nЗагрузка данных...')
    old_profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())

    # Val
    val_df = pl.read_parquet(FEATURES_IN / 'val_proper.parquet')
    val_df = val_df.with_columns(
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'))
    val_df = v9.add_features(val_df)
    val_df = v9.add_customer_profiles(val_df, old_profiles)

    # Train
    val_ids = set(pl.read_parquet(FEATURES_IN / 'val_event_ids.parquet')['event_id'].to_list())
    train_df = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    train_df = train_df.filter(~pl.col('event_id').is_in(val_ids))
    train_df = v9.add_features(train_df)
    train_df = v9.add_customer_profiles(train_df, old_profiles)

    # ── Фаза 2: Add anomaly features ──
    log('\n=== ФАЗА 2: ANOMALY FEATURES ===')
    train_df = add_anomaly_features(train_df, deep_profiles, mcc_profiles)
    val_df = add_anomaly_features(val_df, deep_profiles, mcc_profiles)

    # Get feature lists
    base_feats = [c for c in v9.FEATURE_COLS if c in train_df.columns and c in val_df.columns]
    anom_feats = [c for c in ANOMALY_FEATURES if c in train_df.columns and c in val_df.columns]
    all_feats = base_feats + anom_feats

    y_train = train_df['target'].to_numpy().astype(int)
    y_val = val_df['is_fraud'].to_numpy().astype(int)

    log(f'Train: {len(train_df):,}, Val: {len(val_df):,}')
    log(f'Val fraud: {y_val.sum()} ({100*y_val.mean():.4f}%)')
    log(f'Base features: {len(base_feats)}, Anomaly features: {len(anom_feats)}, Total: {len(all_feats)}')

    results = {}

    # ── Exp A: v10 baseline (для сравнения) ──
    X_train_base = train_df.select(base_feats).to_pandas().astype(np.float32)
    X_val_base = val_df.select(base_feats).to_pandas().astype(np.float32)

    prauc_a, preds_a, seeds_a = train_model(
        'A_baseline', X_train_base, y_train, X_val_base, y_val)
    results['A_baseline'] = prauc_a

    del X_train_base, X_val_base; gc.collect()

    # ── Exp B: Anomaly features ONLY ──
    X_train_anom = train_df.select(anom_feats).to_pandas().astype(np.float32)
    X_val_anom = val_df.select(anom_feats).to_pandas().astype(np.float32)

    prauc_b, preds_b, seeds_b = train_model(
        'B_anomaly_only', X_train_anom, y_train, X_val_anom, y_val)
    results['B_anomaly_only'] = prauc_b

    del X_train_anom, X_val_anom; gc.collect()

    # ── Exp C: Base + Anomaly features ──
    X_train_all = train_df.select(all_feats).to_pandas().astype(np.float32)
    X_val_all = val_df.select(all_feats).to_pandas().astype(np.float32)

    prauc_c, preds_c, seeds_c = train_model(
        'C_base_plus_anomaly', X_train_all, y_train, X_val_all, y_val)
    results['C_base_plus_anomaly'] = prauc_c

    # ── Exp D: Base + Anomaly + 20K trees ──
    prauc_d, preds_d, seeds_d = train_model(
        'D_all_20k', X_train_all, y_train, X_val_all, y_val, n_trees=20000, patience=500)
    results['D_all_20k'] = prauc_d

    del X_train_all, X_val_all; gc.collect()

    # ── ИТОГИ ──
    log(f'\n{"="*60}')
    log('ИТОГИ V14')
    log(f'{"="*60}')

    sorted_res = sorted(results.items(), key=lambda x: -x[1])
    for name, prauc in sorted_res:
        delta = 100 * (prauc / 0.039 - 1)
        log(f'{name:30s}: val={prauc:.6f} ({delta:+.1f}% vs v10)')

    best_name = sorted_res[0][0]
    log(f'\nЛучший: {best_name} (val={sorted_res[0][1]:.6f})')

    # Save results
    with open(MODELS_OUT / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate submissions for top-2
    log('\n=== ГЕНЕРАЦИЯ САБМИТОВ ===')
    for name, prauc in sorted_res[:2]:
        feat_map = {
            'A_baseline': base_feats,
            'B_anomaly_only': anom_feats,
            'C_base_plus_anomaly': all_feats,
            'D_all_20k': all_feats,
        }
        generate_submission(name, feat_map[name], old_profiles, mcc_profiles)

    total_min = (time.time() - t_start) / 60
    log(f'\nВсего: {total_min:.0f} мин')
    log('DONE')
