"""
Полный пайплайн: feature engineering → train → submit
Memory-safe: обработка чанками по customer_id, файлы грузятся по одному.
Пиковое потребление RAM: ~5GB (безопасно для 32GB системы).
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from pathlib import Path
from datetime import datetime
import gc, json, time

ROOT           = Path('/home/vadim/PyPr/hak')
PRETRAIN_TRAIN = ROOT / 'Pre-train_Train'
PRETEST_TEST   = ROOT / 'Pre-test_Test'
DATA           = ROOT / 'main_data'
FEATURES_OUT   = ROOT / 'features'
MODELS_OUT     = ROOT / 'models'
SUBMIT_OUT     = ROOT / 'submissions'

for d in [FEATURES_OUT, MODELS_OUT, SUBMIT_OUT]:
    d.mkdir(exist_ok=True)

HIGH_RISK_TYPES = [12, 15, 10, 6, 3]
WINDOWS_SEC     = ['1h', '6h', '24h', '7d', '30d']

HIST_COLS = [
    'customer_id', 'event_id', 'event_dttm', 'operaton_amt',
    'phone_voip_call_state', 'mcc_code', 'channel_indicator_type',
    'currency_iso_cd', 'session_id',
]

HIST_FEAT_COLS = [
    'event_id',
    'cnt_1h', 'cnt_6h', 'cnt_24h', 'cnt_7d', 'cnt_30d',
    'amt_sum_1h', 'amt_sum_6h', 'amt_sum_24h', 'amt_sum_7d', 'amt_sum_30d',
    'secs_since_last', 'voip_cnt_24h',
    'is_new_mcc_code', 'is_new_channel_indicator_type', 'is_new_currency_iso_cd',
    'cum_unique_mcc_approx',
    'session_ops_before', 'session_amt_before',
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


# ─── helpers ────────────────────────────────

def _ram_gb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1048576
    except Exception:
        return 0.0
    return 0.0


def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")} RAM:{_ram_gb():.1f}GB] {msg}', flush=True)


def _parse_dt(df: pl.DataFrame) -> pl.DataFrame:
    if df['event_dttm'].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col('event_dttm').str.to_datetime('%Y-%m-%d %H:%M:%S')
        )
    return df


def _clean_types(df: pl.DataFrame) -> pl.DataFrame:
    """Приведение типов для train/test (full columns)."""
    df = _parse_dt(df)
    for col in ['compromised', 'developer_tools']:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Int8, strict=False).fill_null(0))
    df = df.with_columns(
        pl.col('mcc_code').cast(pl.Int32, strict=False),
        pl.col('session_id').cast(pl.Int64, strict=False),
        pl.col('battery').cast(pl.Float32, strict=False),
        pl.col('operaton_amt').fill_null(0.0),
        pl.col('phone_voip_call_state').fill_null(0),
        pl.col('web_rdp_connection').fill_null(0),
    )
    return df


def _clean_hist(df: pl.DataFrame) -> pl.DataFrame:
    """Приведение типов для minimal (hist) columns."""
    df = _parse_dt(df)
    df = df.with_columns(
        pl.col('mcc_code').cast(pl.Int32, strict=False),
        pl.col('session_id').cast(pl.Int64, strict=False),
        pl.col('operaton_amt').fill_null(0.0),
        pl.col('phone_voip_call_state').fill_null(0),
    )
    return df


def build_base_features(df: pl.DataFrame) -> pl.DataFrame:
    """Фичи из самой строки — история не нужна."""
    df = df.with_columns([
        pl.col('event_dttm').dt.hour().cast(pl.Int8).alias('hour'),
        pl.col('event_dttm').dt.weekday().cast(pl.Int8).alias('weekday'),
        pl.col('event_dttm').dt.month().cast(pl.Int8).alias('month'),
        (pl.col('event_dttm').dt.hour() < 6).cast(pl.Int8).alias('is_night'),
        pl.col('operaton_amt').log1p().alias('log_amount'),
        pl.col('operaton_amt').is_null().cast(pl.Int8).alias('is_null_amount'),
        pl.col('compromised').fill_null(0),
        pl.col('web_rdp_connection').fill_null(0),
        pl.col('phone_voip_call_state').fill_null(0),
        pl.col('developer_tools').fill_null(0),
        (
            pl.col('compromised').fill_null(0) +
            pl.col('web_rdp_connection').fill_null(0) +
            pl.col('phone_voip_call_state').fill_null(0) +
            pl.col('developer_tools').fill_null(0)
        ).cast(pl.Int8).alias('security_flags_sum'),
        pl.col('mcc_code').is_null().cast(pl.Int8).alias('is_null_mcc'),
        pl.col('mcc_code').fill_null(-1),
        pl.col('event_type_nm').is_in(HIGH_RISK_TYPES).cast(pl.Int8).alias('is_high_risk_type'),
        pl.col('battery').fill_null(-1.0),
    ])
    return df


def build_history_features(df_hist: pl.DataFrame) -> pl.DataFrame:
    """Rolling/novelty/session фичи. df_hist уже отсортирован по [customer_id, event_dttm]."""
    t0 = time.time()
    # Добавляем колонку единиц для подсчёта count через rolling_sum
    df_hist = df_hist.with_columns(pl.lit(1.0).alias('_ones'))
    vel_cols = []
    for w in WINDOWS_SEC:
        vel_cols += [
            pl.col('operaton_amt')
              .rolling_sum_by(by='event_dttm', window_size=w, closed='left')
              .over('customer_id').alias(f'amt_sum_{w}'),
            pl.col('_ones')
              .rolling_sum_by(by='event_dttm', window_size=w, closed='left')
              .over('customer_id').fill_null(0).cast(pl.Int32).alias(f'cnt_{w}'),
        ]
    vel_cols += [
        (
            (pl.col('event_dttm').dt.timestamp('ms') -
             pl.col('event_dttm').dt.timestamp('ms').shift(1).over('customer_id'))
            / 1000
        ).alias('secs_since_last'),
        pl.col('phone_voip_call_state')
          .rolling_sum_by(by='event_dttm', window_size='24h', closed='left')
          .over('customer_id').alias('voip_cnt_24h'),
    ]
    df_hist = df_hist.with_columns(vel_cols)

    nov_cols = []
    for col in ['mcc_code', 'channel_indicator_type', 'currency_iso_cd']:
        nov_cols.append(
            (pl.col(col).cum_count().over(['customer_id', col]) == 1)
              .cast(pl.Int8).alias(f'is_new_{col}')
        )
    nov_cols.append(
        pl.col('mcc_code').cum_count().over('customer_id').alias('cum_unique_mcc_approx')
    )
    df_hist = df_hist.with_columns(nov_cols)

    df_hist = df_hist.with_columns([
        (pl.col('event_id').cum_count().over('session_id') - 1)
          .cast(pl.Int32).alias('session_ops_before'),
        (pl.col('operaton_amt').cum_sum().over('session_id') - pl.col('operaton_amt'))
          .alias('session_amt_before'),
    ])

    log(f'    history features done ({time.time()-t0:.1f}s)')
    return df_hist.select(HIST_FEAT_COLS)


def get_all_customer_ids(files: list) -> pl.Series:
    """Собираем уникальных customer_id из всех файлов (по одному за раз)."""
    cids = set()
    for f in files:
        s = pl.read_parquet(f, columns=['customer_id'])['customer_id'].unique()
        cids.update(s.to_list())
        del s; gc.collect()
    return pl.Series('customer_id', sorted(cids))


def load_file_filtered(path: Path, cid_set: set, columns=None) -> pl.DataFrame:
    """Грузим один parquet-файл, фильтруем по customer_id. Peak RAM = 1 файл."""
    df = pl.read_parquet(path, columns=columns)
    df = df.filter(pl.col('customer_id').is_in(list(cid_set)))
    return df


# ─────────────────────────────────────────────
# STEP 1: Build train features (chunked)
# ─────────────────────────────────────────────

def build_train_features():
    train_feat_path = FEATURES_OUT / 'train_features.parquet'
    if train_feat_path.exists():
        log(f'Already exists: {train_feat_path} — skip')
        return

    pretrain_files = sorted(PRETRAIN_TRAIN.glob('pretrain*.parquet'))
    train_files    = sorted(PRETRAIN_TRAIN.glob('train*.parquet'))
    all_files      = pretrain_files + train_files
    log(f'pretrain: {len(pretrain_files)} files, train: {len(train_files)} files')

    log('Collecting unique customer_ids...')
    all_cids = get_all_customer_ids(all_files)
    n_customers = len(all_cids)
    log(f'Unique customers: {n_customers:,}')

    N_CHUNKS = 32
    chunk_size = n_customers // N_CHUNKS + 1
    tmp_dir = FEATURES_OUT / '_tmp_train'
    tmp_dir.mkdir(exist_ok=True)
    log(f'Processing in {N_CHUNKS} chunks of ~{chunk_size:,} customers')

    for ci in range(0, n_customers, chunk_size):
        chunk_idx = ci // chunk_size + 1
        tmp_path = tmp_dir / f'chunk_{chunk_idx:03d}.parquet'
        if tmp_path.exists():
            log(f'  Chunk {chunk_idx}/{N_CHUNKS}: already done — skip')
            continue

        chunk_cids = all_cids[ci:ci + chunk_size]
        cid_set = set(chunk_cids.to_list())
        log(f'  Chunk {chunk_idx}/{N_CHUNKS}: {len(cid_set):,} customers')

        # 1. Load history (pretrain minimal + train minimal) for this chunk
        hist_parts = []
        for f in pretrain_files:
            df = load_file_filtered(f, cid_set, columns=HIST_COLS)
            df = _clean_hist(df)
            hist_parts.append(df)
            del df

        # 2. Load train (full cols) for this chunk
        train_parts = []
        for f in train_files:
            df = load_file_filtered(f, cid_set)
            df = _clean_types(df)
            hist_parts.append(df.select(HIST_COLS))
            train_parts.append(df)
            del df

        gc.collect()

        # 3. History features on full timeline
        df_hist = pl.concat(hist_parts).sort(['customer_id', 'event_dttm'])
        del hist_parts; gc.collect()
        log(f'    hist: {len(df_hist):,} rows, {df_hist.estimated_size("mb"):.0f}MB')

        df_hist_feats = build_history_features(df_hist)
        del df_hist; gc.collect()

        # 4. Train base features
        df_train_chunk = pl.concat(train_parts).sort(['customer_id', 'event_dttm'])
        del train_parts; gc.collect()

        train_eids = df_train_chunk.select('event_id')
        df_hist_feats = train_eids.join(df_hist_feats, on='event_id', how='left')
        del train_eids

        df_train_base = build_base_features(df_train_chunk)
        del df_train_chunk; gc.collect()

        df_chunk = df_train_base.join(df_hist_feats, on='event_id', how='left')
        del df_train_base, df_hist_feats; gc.collect()

        # Сохраняем чанк на диск — освобождаем RAM
        df_chunk.write_parquet(tmp_path)
        del df_chunk; gc.collect()
        log(f'    saved {tmp_path.name}, {_ram_gb():.1f}GB RAM')

    # Собираем training-ready dataset: labeled + sampled nulls (по одному чанку)
    # Вместо загрузки 85M строк — берём только ~344K нужных
    log('Building training dataset from chunks...')
    chunk_files = sorted(tmp_dir.glob('chunk_*.parquet'))
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    train_start = datetime(2024, 10, 1)

    n_fraud = labels.filter(pl.col('target') == 1).height
    n_null_target = n_fraud * 5

    # Считаем общее число unlabeled для пропорционального семплирования
    label_eids = set(labels['event_id'].to_list())
    total_null = 0
    for cf in chunk_files:
        n = pl.scan_parquet(cf).filter(
            pl.col('event_dttm') >= train_start
        ).select(pl.len()).collect().item()
        total_null += n
    total_null -= len(labels)
    frac = n_null_target / max(total_null, 1)
    log(f'  fraud={n_fraud:,}  null_target={n_null_target:,}  frac={frac:.4f}')

    results = []
    for i, cf in enumerate(chunk_files):
        df = pl.read_parquet(cf)
        df = df.filter(pl.col('event_dttm') >= train_start)
        df = df.join(labels.select(['event_id', 'target']), on='event_id', how='left')

        labeled = df.filter(pl.col('target').is_not_null())
        df_null = df.filter(pl.col('target').is_null())

        n_sample = max(1, round(len(df_null) * frac))
        null_sampled = df_null.sample(n=min(n_sample, len(df_null)), seed=42 + i)
        null_sampled = null_sampled.with_columns(pl.lit(0).cast(pl.Int32).alias('target'))

        results.append(pl.concat([labeled, null_sampled]))
        del df, labeled, df_null, null_sampled; gc.collect()

    df_train = pl.concat(results)
    del results; gc.collect()
    log(f'Train rows: {len(df_train):,}')
    log(f'Target dist:\n{df_train["target"].value_counts().sort("target")}')

    df_train.write_parquet(train_feat_path)
    log(f'Saved → {train_feat_path}')
    del df_train; gc.collect()


# ─────────────────────────────────────────────
# STEP 2: Build test features (chunked)
# ─────────────────────────────────────────────

def build_test_features():
    test_feat_path = FEATURES_OUT / 'test_features.parquet'
    if test_feat_path.exists():
        log(f'Already exists: {test_feat_path} — skip')
        return

    pretrain_files = sorted(PRETRAIN_TRAIN.glob('pretrain*.parquet'))
    train_files    = sorted(PRETRAIN_TRAIN.glob('train*.parquet'))
    pretest_file   = PRETEST_TEST / 'pretest.parquet'
    test_file      = PRETEST_TEST / 'test.parquet'
    hist_files     = pretrain_files + train_files + [pretest_file]

    log('Collecting unique customer_ids for test...')
    test_cids = pl.read_parquet(test_file, columns=['customer_id'])['customer_id'].unique()
    cid_list = sorted(test_cids.to_list())
    n_customers = len(cid_list)
    log(f'Test customers: {n_customers:,}')

    N_CHUNKS = 48
    chunk_size = n_customers // N_CHUNKS + 1
    tmp_dir = FEATURES_OUT / '_tmp_test'
    tmp_dir.mkdir(exist_ok=True)
    log(f'Processing in {N_CHUNKS} chunks of ~{chunk_size:,} customers')

    for ci in range(0, n_customers, chunk_size):
        chunk_idx = ci // chunk_size + 1
        tmp_path = tmp_dir / f'chunk_{chunk_idx:03d}.parquet'
        if tmp_path.exists():
            log(f'  Chunk {chunk_idx}/{N_CHUNKS}: already done — skip')
            continue

        cid_set = set(cid_list[ci:ci + chunk_size])
        log(f'  Chunk {chunk_idx}/{N_CHUNKS}: {len(cid_set):,} customers')

        # 1. Load all history (pretrain + train + pretest) minimal
        hist_parts = []
        for f in hist_files:
            df = load_file_filtered(f, cid_set, columns=HIST_COLS)
            df = _clean_hist(df)
            hist_parts.append(df)
            del df

        # 2. Load test (full cols)
        df_test_chunk = load_file_filtered(test_file, cid_set)
        df_test_chunk = _clean_types(df_test_chunk)
        hist_parts.append(df_test_chunk.select(HIST_COLS))

        gc.collect()

        # 3. History features
        df_hist = pl.concat(hist_parts).sort(['customer_id', 'event_dttm'])
        del hist_parts; gc.collect()
        log(f'    hist: {len(df_hist):,} rows, {df_hist.estimated_size("mb"):.0f}MB')

        df_hist_feats = build_history_features(df_hist)
        del df_hist; gc.collect()

        # 4. Test base features
        test_eids = df_test_chunk.select('event_id')
        df_hist_feats = test_eids.join(df_hist_feats, on='event_id', how='left')
        del test_eids

        df_test_base = build_base_features(df_test_chunk)
        del df_test_chunk; gc.collect()

        df_chunk = df_test_base.join(df_hist_feats, on='event_id', how='left')
        del df_test_base, df_hist_feats; gc.collect()

        df_chunk.write_parquet(tmp_path)
        del df_chunk; gc.collect()
        log(f'    saved {tmp_path.name}, {_ram_gb():.1f}GB RAM')

    # Test — всего 633K строк, безопасно собрать в RAM
    log('Assembling test chunks...')
    chunk_files = sorted(tmp_dir.glob('chunk_*.parquet'))
    parts = []
    for cf in chunk_files:
        parts.append(pl.read_parquet(cf))
    df_test = pl.concat(parts)
    del parts; gc.collect()
    log(f'Test features: {len(df_test):,}')
    df_test.write_parquet(test_feat_path)
    log(f'Saved → {test_feat_path}')
    del df_test; gc.collect()


# ─────────────────────────────────────────────
# STEP 3: Training
# ─────────────────────────────────────────────

def train_models():
    train_feat_path = FEATURES_OUT / 'train_features.parquet'
    log('Loading training dataset...')
    df_dataset = pl.read_parquet(train_feat_path)

    n_fraud = df_dataset.filter(pl.col('target') == 1).height
    n_neg   = df_dataset.filter(pl.col('target') == 0).height
    log(f'Dataset: {len(df_dataset):,}  fraud={n_fraud:,}  neg={n_neg:,}')

    val_dt = datetime(2025, 4, 1)
    df_tr  = df_dataset.filter(pl.col('event_dttm') < val_dt)
    df_val = df_dataset.filter(pl.col('event_dttm') >= val_dt)
    log(f'Train split: {len(df_tr):,}  |  Val: {len(df_val):,}')
    log(f'Val fraud: {df_val.filter(pl.col("target")==1).height}')

    def to_xy(d):
        X = d.select(FEATURE_COLS).to_pandas()
        y = d['target'].to_numpy().astype(int)
        return X, y

    X_train, y_train = to_xy(df_tr)
    X_val,   y_val   = to_xy(df_val)
    del df_tr, df_val; gc.collect()

    # ── LightGBM GPU ──
    log('Training LightGBM GPU...')
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos = n_neg / n_pos
    log(f'  scale_pos_weight={scale_pos:.2f}  pos={n_pos:,}  neg={n_neg:,}')

    lgbm_model = lgb.LGBMClassifier(
        objective='binary',
        metric='average_precision',
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        scale_pos_weight=scale_pos,
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(150, verbose=True),
            lgb.log_evaluation(50),
        ],
    )
    lgbm_preds = lgbm_model.predict_proba(X_val)[:, 1]
    lgbm_score = average_precision_score(y_val, lgbm_preds)
    log(f'>>> LightGBM Val PR-AUC: {lgbm_score:.4f}')
    lgbm_model.booster_.save_model(str(MODELS_OUT / 'lgbm.txt'))

    # ── CatBoost GPU ──
    log('Training CatBoost GPU...')
    cat_model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.05,
        depth=8,
        task_type='GPU',
        devices='0',
        loss_function='Logloss',
        eval_metric='AUC',
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=150,
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    cat_preds  = cat_model.predict_proba(X_val)[:, 1]
    cat_score  = average_precision_score(y_val, cat_preds)
    log(f'>>> CatBoost Val PR-AUC: {cat_score:.4f}')
    cat_model.save_model(str(MODELS_OUT / 'catboost.cbm'))

    # ── Ensemble ──
    total     = lgbm_score + cat_score
    w_lgbm    = lgbm_score / total
    w_cat     = cat_score  / total
    ens_preds = lgbm_preds * w_lgbm + cat_preds * w_cat
    ens_score = average_precision_score(y_val, ens_preds)
    log(f'>>> Ensemble Val PR-AUC: {ens_score:.4f}')
    log(f'    lgbm={lgbm_score:.4f} w={w_lgbm:.2f}  cat={cat_score:.4f} w={w_cat:.2f}')
    weights = {'lgbm': w_lgbm, 'catboost': w_cat}
    with open(MODELS_OUT / 'weights.json', 'w') as f:
        json.dump(weights, f)

    # ── Final model on ALL data ──
    log('Training final LightGBM on all data...')
    X_full, y_full = to_xy(df_dataset)
    best_iter  = lgbm_model.best_iteration_
    final_lgbm = lgb.LGBMClassifier(
        objective='binary',
        metric='average_precision',
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        scale_pos_weight=scale_pos,
        n_estimators=int(best_iter * 1.1),
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    final_lgbm.fit(X_full, y_full)
    final_lgbm.booster_.save_model(str(MODELS_OUT / 'lgbm_final.txt'))
    log('Saved lgbm_final.txt')
    del X_full, y_full, df_dataset; gc.collect()

    return weights, cat_model


# ─────────────────────────────────────────────
# STEP 4: Predict & Submit
# ─────────────────────────────────────────────

def predict_and_submit(weights, cat_model):
    test_feat_path = FEATURES_OUT / 'test_features.parquet'
    log('Loading test features...')
    df_test   = pl.read_parquet(test_feat_path)
    X_test    = df_test.select(FEATURE_COLS).to_pandas()
    event_ids = df_test['event_id'].to_numpy()
    log(f'X_test: {X_test.shape}')

    final_lgbm_booster = lgb.Booster(model_file=str(MODELS_OUT / 'lgbm_final.txt'))
    lgbm_test = final_lgbm_booster.predict(X_test)
    cat_test  = cat_model.predict_proba(X_test)[:, 1]
    ens_test  = lgbm_test * weights['lgbm'] + cat_test * weights['catboost']
    log(f'Predictions: min={ens_test.min():.4f}  max={ens_test.max():.4f}  mean={ens_test.mean():.4f}')

    sample = pl.read_csv(DATA / 'sample_submit.csv')
    submit = pl.DataFrame({'event_id': event_ids, 'predict': ens_test})
    submit = sample.select('event_id').join(submit, on='event_id', how='left')

    n_null = submit['predict'].is_null().sum()
    if n_null > 0:
        median_pred = submit['predict'].drop_nulls().median()
        submit = submit.with_columns(pl.col('predict').fill_null(median_pred))
        log(f'Filled {n_null} missing with median')

    ts       = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = SUBMIT_OUT / f'submit_{ts}.csv'
    submit.write_csv(out_path)
    log(f'Saved: {out_path}')
    log(f'Rows: {len(submit):,}')


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    t_start = time.time()

    print('\n' + '='*60)
    print('STEP 1: Build train features (chunked, memory-safe)')
    print('='*60)
    build_train_features()

    print('\n' + '='*60)
    print('STEP 2: Build test features (chunked, memory-safe)')
    print('='*60)
    build_test_features()

    print('\n' + '='*60)
    print('STEP 3: Training (LightGBM GPU + CatBoost GPU)')
    print('='*60)
    weights, cat_model = train_models()

    print('\n' + '='*60)
    print('STEP 4: Predict & Submit')
    print('='*60)
    predict_and_submit(weights, cat_model)

    log(f'Total time: {(time.time()-t_start)/60:.1f} min')
    log('ALL DONE')
