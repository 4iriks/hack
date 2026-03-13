"""
Feature engineering pipeline — GPU-accelerated.
Тяжёлые операции (rolling windows, groupby) выполняются на GPU через cuDF + numba CUDA.
Данные загружаются чанками по customer_id через polars lazy scan.
"""
import polars as pl
import cudf
import cupy as cp
import numpy as np
from numba import cuda
from pathlib import Path
import gc, time

ROOT = Path('/home/vadim/PyPr/hak')
PRETRAIN_TRAIN = ROOT / 'Pre-train_Train'
PRETEST_TEST = ROOT / 'Pre-test_Test'
DATA = ROOT / 'main_data'
FEATURES_OUT = ROOT / 'features'
FEATURES_OUT.mkdir(exist_ok=True)

HIGH_RISK_TYPES = [12, 15, 10, 6, 3]
WINDOW_SECONDS = {
    '1h': 3600, '6h': 21600, '24h': 86400,
    '7d': 604800, '30d': 2592000,
}

NEEDED_COLS = [
    'customer_id', 'event_id', 'event_dttm',
    'event_type_nm', 'channel_indicator_type', 'channel_indicator_sub_type',
    'operaton_amt', 'currency_iso_cd', 'mcc_code', 'pos_cd',
    'timezone', 'session_id', 'operating_system_type', 'battery',
    'developer_tools', 'phone_voip_call_state', 'web_rdp_connection', 'compromised',
]

FEATURE_COLS = [
    'hour', 'weekday', 'month', 'is_night',
    'log_amount', 'operaton_amt', 'is_null_amount',
    'phone_voip_call_state', 'web_rdp_connection', 'compromised',
    'developer_tools', 'security_flags_sum',
    'event_type_nm', 'is_high_risk_type',
    'mcc_code', 'is_null_mcc',
    'channel_indicator_type', 'channel_indicator_sub_type', 'currency_iso_cd',
    'battery', 'operating_system_type', 'pos_cd',
    'cnt_1h', 'cnt_6h', 'cnt_24h', 'cnt_7d', 'cnt_30d',
    'amt_sum_1h', 'amt_sum_6h', 'amt_sum_24h', 'amt_sum_7d', 'amt_sum_30d',
    'secs_since_last', 'voip_cnt_24h',
    'is_new_mcc_code', 'is_new_channel_indicator_type', 'is_new_currency_iso_cd',
    'cum_unique_mcc_approx',
    'session_ops_before', 'session_amt_before',
    'timezone',
]


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_lazy(paths):
    """Lazy scan parquet, select only needed columns, unify schema."""
    frames = []
    for p in paths:
        schema = pl.scan_parquet(p).collect_schema()
        avail = [c for c in NEEDED_COLS if c in schema.names()]
        lf = pl.scan_parquet(p).select(avail)
        for c in avail:
            if schema[c] == pl.Int32 and c in ('session_id', 'event_id', 'customer_id'):
                lf = lf.with_columns(pl.col(c).cast(pl.Int64))
        frames.append(lf)
    return pl.concat(frames)


def clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Парсим типы — lazy."""
    return lf.with_columns(
        pl.col('event_dttm').str.to_datetime('%Y-%m-%d %H:%M:%S'),
        pl.col('compromised').cast(pl.Int8, strict=False).fill_null(0),
        pl.col('developer_tools').cast(pl.Int8, strict=False).fill_null(0),
        pl.col('mcc_code').cast(pl.Int32, strict=False),
        pl.col('battery').cast(pl.Float32, strict=False),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CUDA kernel для binary search границы rolling window
# ═══════════════════════════════════════════════════════════════════════════════

@cuda.jit
def _rolling_boundary_kernel(epoch, group_start, group_id, window_sec, out):
    """
    Для каждой строки i находит индекс первой строки в той же группе,
    где epoch >= epoch[i] - window_sec (левая граница окна).
    """
    i = cuda.grid(1)
    if i >= epoch.shape[0]:
        return
    gs = group_start[group_id[i]]
    target = epoch[i] - window_sec
    # Binary search in epoch[gs:i] for first value >= target
    lo, hi = gs, i
    while lo < hi:
        mid = (lo + hi) // 2
        if epoch[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    out[i] = lo


def compute_rolling_boundaries(epoch_np, group_starts_np, group_ids_np, window_sec):
    """Запускает CUDA kernel для поиска границ rolling window."""
    n = len(epoch_np)
    epoch_d = cuda.to_device(epoch_np)
    gs_d = cuda.to_device(group_starts_np)
    gi_d = cuda.to_device(group_ids_np)
    out_d = cuda.device_array(n, dtype=np.int64)

    threads = 256
    blocks = (n + threads - 1) // threads
    _rolling_boundary_kernel[blocks, threads](epoch_d, gs_d, gi_d, window_sec, out_d)
    return out_d.copy_to_host()


# ═══════════════════════════════════════════════════════════════════════════════
# Feature functions (GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def add_base_features(gdf: cudf.DataFrame) -> cudf.DataFrame:
    """Базовые фичи — всё на GPU."""
    dt = gdf['event_dttm'].dt
    gdf['hour'] = dt.hour.astype('int8')
    gdf['weekday'] = dt.weekday.astype('int8')
    gdf['month'] = dt.month.astype('int8')
    gdf['is_night'] = (dt.hour < 6).astype('int8')

    gdf['is_null_amount'] = gdf['operaton_amt'].isna().astype('int8')
    gdf['operaton_amt'] = gdf['operaton_amt'].fillna(0.0)
    gdf['log_amount'] = cp.log1p(cp.asarray(gdf['operaton_amt'].values))

    gdf['is_null_mcc'] = gdf['mcc_code'].isna().astype('int8')
    gdf['mcc_code'] = gdf['mcc_code'].fillna(-1)

    gdf['is_high_risk_type'] = gdf['event_type_nm'].isin(HIGH_RISK_TYPES).astype('int8')

    gdf['compromised'] = gdf['compromised'].fillna(0)
    gdf['web_rdp_connection'] = gdf['web_rdp_connection'].fillna(0)
    gdf['phone_voip_call_state'] = gdf['phone_voip_call_state'].fillna(0)
    gdf['developer_tools'] = gdf['developer_tools'].fillna(0)

    gdf['security_flags_sum'] = (
        gdf['compromised'] + gdf['web_rdp_connection'] +
        gdf['phone_voip_call_state'] + gdf['developer_tools']
    ).astype('int8')

    gdf['battery'] = gdf['battery'].fillna(-1.0)
    return gdf


def add_velocity_features(gdf: cudf.DataFrame) -> cudf.DataFrame:
    """
    Rolling windows на GPU: numba CUDA kernel для binary search +
    cumsum trick для O(1) rolling sum/count.
    """
    n = len(gdf)

    # Epoch seconds
    epoch = gdf['event_dttm'].astype('int64').values
    epoch_np = cp.asnumpy(cp.asarray(epoch)) // 10**9

    # Group boundaries (data already sorted by customer_id + event_dttm)
    cid = cp.asarray(gdf['customer_id'].values)
    group_change = cp.concatenate([cp.array([True]), cid[1:] != cid[:-1]])
    group_starts = cp.where(group_change)[0]
    group_ids = cp.cumsum(group_change) - 1
    group_starts_np = cp.asnumpy(group_starts).astype(np.int64)
    group_ids_np = cp.asnumpy(group_ids).astype(np.int64)

    # Exclusive prefix sum of amt per group:
    # prefix[i] = sum of amt in [group_start, i) (excluding current row)
    amt = cp.asarray(gdf['operaton_amt'].values, dtype=cp.float64)
    cumsum_amt = cp.zeros(n, dtype=cp.float64)
    # Compute per-group cumsum then shift by 1 within group
    raw_cumsum = gdf.groupby('customer_id')['operaton_amt'].cumsum()
    cumsum_amt = cp.asarray(raw_cumsum.values, dtype=cp.float64)
    prefix_amt = cumsum_amt - amt  # exclusive of current row

    # Prefix count: just position within group
    pos_in_group = cp.arange(n, dtype=cp.int64)
    # Subtract group start to get 0-based position
    pos_in_group = pos_in_group - cp.asarray(group_starts_np)[group_ids]

    # Rolling windows
    for wname, wsec in WINDOW_SECONDS.items():
        boundary = compute_rolling_boundaries(epoch_np, group_starts_np, group_ids_np, wsec)
        boundary_cp = cp.asarray(boundary)

        # Rolling count (closed='left', excluding current row)
        rolling_cnt = pos_in_group - (boundary_cp - cp.asarray(group_starts_np)[group_ids])
        gdf[f'cnt_{wname}'] = cudf.Series(rolling_cnt.astype(cp.int32))

        # Rolling sum (closed='left', excluding current row)
        prefix_at_boundary = prefix_amt[boundary_cp]
        rolling_sum = prefix_amt - prefix_at_boundary
        gdf[f'amt_sum_{wname}'] = cudf.Series(rolling_sum)

    # secs_since_last (shift within group)
    epoch_series = gdf['event_dttm'].astype('int64') // 10**9
    prev_epoch = gdf.groupby('customer_id')[epoch_series.name].shift(1) if epoch_series.name else None
    # Workaround: add as column then shift
    gdf['_epoch_s'] = epoch_series
    gdf['_prev_epoch'] = gdf.groupby('customer_id')['_epoch_s'].shift(1)
    gdf['secs_since_last'] = (gdf['_epoch_s'] - gdf['_prev_epoch']).astype('float64')
    gdf.drop(columns=['_epoch_s', '_prev_epoch'], inplace=True)

    # voip_cnt_24h — reuse the 24h boundary
    boundary_24h = compute_rolling_boundaries(epoch_np, group_starts_np, group_ids_np, 86400)
    voip = cp.asarray(gdf['phone_voip_call_state'].values, dtype=cp.float64)
    cumsum_voip = cp.asarray(
        gdf.groupby('customer_id')['phone_voip_call_state'].cumsum().values,
        dtype=cp.float64,
    )
    prefix_voip = cumsum_voip - voip
    boundary_24h_cp = cp.asarray(boundary_24h)
    gdf['voip_cnt_24h'] = cudf.Series(
        (prefix_voip - prefix_voip[boundary_24h_cp]).astype(cp.int32)
    )

    return gdf


def add_novelty_features(gdf: cudf.DataFrame) -> cudf.DataFrame:
    """Novelty фичи на GPU."""
    for col in ['mcc_code', 'channel_indicator_type', 'currency_iso_cd']:
        gdf[f'is_new_{col}'] = (
            gdf.groupby(['customer_id', col]).cumcount() == 0
        ).astype('int8')

    gdf['cum_unique_mcc_approx'] = gdf.groupby('customer_id')['mcc_code'].cumcount()
    return gdf


def add_session_features(gdf: cudf.DataFrame) -> cudf.DataFrame:
    """Session фичи на GPU."""
    gdf['session_ops_before'] = gdf.groupby('session_id').cumcount().astype('int32')

    gdf['_session_cumsum'] = gdf.groupby('session_id')['operaton_amt'].cumsum()
    gdf['session_amt_before'] = gdf['_session_cumsum'] - gdf['operaton_amt']
    gdf.drop(columns=['_session_cumsum'], inplace=True)
    return gdf


def process_chunk_gpu(df_pl: pl.DataFrame) -> cudf.DataFrame:
    """Polars DataFrame → cuDF → фичи на GPU."""
    # polars → arrow → cudf
    gdf = cudf.DataFrame.from_arrow(df_pl.to_arrow())
    gdf = gdf.sort_values(['customer_id', 'event_dttm']).reset_index(drop=True)

    gdf = add_base_features(gdf)
    gdf = add_velocity_features(gdf)
    gdf = add_novelty_features(gdf)
    gdf = add_session_features(gdf)
    return gdf


# ═══════════════════════════════════════════════════════════════════════════════
# Chunked processing
# ═══════════════════════════════════════════════════════════════════════════════

def build_features_chunked(lf_clean: pl.LazyFrame, keep_event_ids=None,
                            min_date=None, n_chunks=16):
    """
    Считаем фичи чанками по customer_id.
    Каждый чанк обрабатывается на GPU через cuDF.
    """
    log('  Collecting unique customer_ids...')
    cids = lf_clean.select('customer_id').unique().collect().to_series().sort()
    log(f'  Unique customers: {len(cids):,}')

    chunk_size = len(cids) // n_chunks + 1
    chunks = [cids[i:i+chunk_size] for i in range(0, len(cids), chunk_size)]
    log(f'  Chunks: {len(chunks)}, ~{chunk_size:,} customers each')

    results = []
    for i, cid_chunk in enumerate(chunks):
        log(f'  Chunk {i+1}/{len(chunks)}: {len(cid_chunk):,} customers')

        cid_series = (cid_chunk.to_frame() if hasattr(cid_chunk, 'to_frame')
                      else pl.Series('customer_id', cid_chunk).to_frame())
        df_chunk = (
            lf_clean
            .join(cid_series.lazy(), on='customer_id', how='semi')
            .sort(['customer_id', 'event_dttm'])
            .collect()
        )
        log(f'    Loaded: {len(df_chunk):,} rows, {df_chunk.estimated_size("mb"):.0f} MB')

        # GPU processing
        gdf = process_chunk_gpu(df_chunk)
        del df_chunk
        gc.collect()

        # Filter needed rows (still on GPU)
        if keep_event_ids is not None:
            keep_series = cudf.Series(list(keep_event_ids))
            gdf = gdf[gdf['event_id'].isin(keep_series)]
        if min_date is not None:
            min_dt = np.datetime64(min_date)
            gdf = gdf[gdf['event_dttm'] >= min_dt]

        log(f'    After filter: {len(gdf):,} rows')
        if len(gdf) > 0:
            # GPU → arrow → polars
            result_pl = pl.from_arrow(gdf.to_arrow())
            results.append(result_pl)

        del gdf
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    result = pl.concat(results)
    log(f'  Total result: {len(result):,} rows')
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
log('=== TRAIN features ===')

train_files = sorted(PRETRAIN_TRAIN.glob('*.parquet'))
log(f'Files: {[f.name for f in train_files]}')

lf_all = clean(load_lazy(train_files))
min_train_date = '2024-10-01'

log('Building features in chunks (GPU)...')
df_train = build_features_chunked(
    lf_all,
    keep_event_ids=None,
    min_date=min_train_date,
    n_chunks=32,
)

# Присоединяем метки
labels = pl.read_parquet(DATA / 'train_labels.parquet')
df_train = df_train.join(labels.select(['event_id', 'target']), on='event_id', how='left')

log(f'Train: {len(df_train):,} rows')
log(f'target=1: {df_train.filter(pl.col("target")==1).height:,}')
log(f'target=0: {df_train.filter(pl.col("target")==0).height:,}')
log(f'target=null: {df_train["target"].is_null().sum():,}')

df_train.write_parquet(FEATURES_OUT / 'train_features.parquet')
log('Saved train_features.parquet')
del df_train; gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════
log('=== TEST features ===')

test_files = [PRETEST_TEST / 'pretest.parquet', PRETEST_TEST / 'test.parquet']
all_files = sorted(PRETRAIN_TRAIN.glob('*.parquet')) + test_files
log(f'All files for test context: {[f.name for f in all_files]}')

lf_full = clean(load_lazy(all_files))

test_event_ids = pl.read_parquet(PRETEST_TEST / 'test.parquet', columns=['event_id'])['event_id'].to_list()
test_event_ids_set = set(test_event_ids)
log(f'Test event_ids: {len(test_event_ids_set):,}')

log('Building features in chunks (GPU)...')
df_test = build_features_chunked(
    lf_full,
    keep_event_ids=test_event_ids_set,
    min_date=None,
    n_chunks=64,
)

log(f'Test: {len(df_test):,} rows')
df_test.write_parquet(FEATURES_OUT / 'test_features.parquet')
log('Saved test_features.parquet')

log('=== ALL DONE ===')
