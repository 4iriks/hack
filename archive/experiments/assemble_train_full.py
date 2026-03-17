"""
Assemble large training dataset from cached feature chunks.
Reads 85.7M pre-computed feature rows chunk by chunk (memory-safe).
Keeps all labeled + samples green at desired ratio.
"""
import polars as pl
import gc, time, os

ROOT = '/home/vadim/PyPr/hak'
CHUNKS_DIR = f'{ROOT}/features/_tmp_train'
LABELS_PATH = f'{ROOT}/main_data/train_labels.parquet'
OUT_PATH = f'{ROOT}/features/train_features_full.parquet'

GREEN_TO_FRAUD_RATIO = 50  # 50:1 green:fraud
SEED = 42

def ram_gb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1048576
    except:
        return 0.0
    return 0.0

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")} RAM:{ram_gb():.1f}GB] {msg}', flush=True)

# Load labels
labels = pl.read_parquet(LABELS_PATH)
n_fraud = (labels['target'] == 1).sum()
n_confirmed = (labels['target'] == 0).sum()
log(f'Labels: {len(labels):,} (fraud={n_fraud:,}, confirmed={n_confirmed:,})')

label_event_ids = set(labels['event_id'].to_list())
target_green = n_fraud * GREEN_TO_FRAUD_RATIO
log(f'Target green sample: {target_green:,} (ratio {GREEN_TO_FRAUD_RATIO}:1)')

# Count total green across chunks to compute sampling fraction
chunk_files = sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith('.parquet')])
log(f'Chunks: {len(chunk_files)}')

# First pass: count green rows per chunk
log('Pass 1: counting green rows...')
green_counts = []
for cf in chunk_files:
    path = os.path.join(CHUNKS_DIR, cf)
    df = pl.scan_parquet(path).select('event_id').collect()
    n_total = len(df)
    n_labeled = df.filter(pl.col('event_id').is_in(label_event_ids)).height
    n_green = n_total - n_labeled
    green_counts.append(n_green)
    del df; gc.collect()

total_green = sum(green_counts)
sample_fraction = min(target_green / total_green, 1.0)
log(f'Total green: {total_green:,}, sample fraction: {sample_fraction:.4f}')

# Second pass: extract labeled + sampled green
log('Pass 2: extracting data...')
results = []
total_labeled = 0
total_sampled_green = 0

for i, cf in enumerate(chunk_files):
    path = os.path.join(CHUNKS_DIR, cf)
    df = pl.read_parquet(path)

    # Split labeled vs green
    labeled = df.filter(pl.col('event_id').is_in(label_event_ids))
    green = df.filter(~pl.col('event_id').is_in(label_event_ids))

    # Sample green
    n_sample = max(1, int(len(green) * sample_fraction))
    green_sample = green.sample(n=n_sample, seed=SEED + i)

    # Combine
    chunk_result = pl.concat([labeled, green_sample])
    results.append(chunk_result)

    total_labeled += len(labeled)
    total_sampled_green += len(green_sample)

    log(f'  Chunk {i+1}/{len(chunk_files)}: {len(df):,} total, '
        f'{len(labeled):,} labeled, {len(green_sample):,} green sampled')

    del df, labeled, green, green_sample, chunk_result
    gc.collect()

# Concatenate all
log('Concatenating...')
df_all = pl.concat(results)
del results; gc.collect()

log(f'Total: {len(df_all):,} rows (labeled={total_labeled:,}, green={total_sampled_green:,})')

# Join labels
df_all = df_all.join(labels.select(['event_id', 'target']), on='event_id', how='left')

# Green rows get target=0
df_all = df_all.with_columns(
    pl.col('target').fill_null(0).cast(pl.Int8)
)

# Verify
n1 = (df_all['target'] == 1).sum()
n0 = (df_all['target'] == 0).sum()
log(f'Final: target=1: {n1:,}, target=0: {n0:,}, ratio={n0/n1:.1f}:1')

# Save
df_all.write_parquet(OUT_PATH)
file_size_mb = os.path.getsize(OUT_PATH) / 1048576
log(f'Saved: {OUT_PATH} ({file_size_mb:.0f} MB)')
log('DONE')
