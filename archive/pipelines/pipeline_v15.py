"""
Pipeline v15: Device Graph + Sequence Features + Temporal Reweighting.

═══════════════════════════════════════════════════════════════════
ПЛАН ЭКСПЕРИМЕНТОВ v15 (ночь 16→17 марта 2026)
═══════════════════════════════════════════════════════════════════

Базовая линия: v14-C = LB 0.1010, val 0.044 (91 base + 30 anomaly = 121 фича)

БЛОК 1 — Device Graph + Battery (потенциал +10-20%):
  - device_fp = screen_size + device_version → fingerprint устройства
  - dev_n_customers: сколько клиентов на этом устройстве (из 85M train)
  - dev_n_fraud_customers: сколько из них фродеры
  - dev_fraud_rate: доля фрода на устройстве
  - dev_is_farm: 10+ клиентов = дроп-центр
  - bat_is_100: battery=100% → маркер эмулятора (fraud rate 5x выше!)
  - bat_level: числовой уровень батареи
  Почему сработает: time-invariant, описывает СРЕДУ, а не поведение

БЛОК 2 — Sequence Features (потенциал +15-30%):
  - seq_amt_ratio_prev: отношение суммы к предыдущей транзакции
  - seq_amt_diff_prev: разница с предыдущей
  - seq_mcc_changed: сменился MCC (новый мерчант)
  - seq_pos_to_manual: переход с чипа на ручной ввод (CNP-фрод!)
  - seq_micro_then_large: микро-тест → крупная операция (card testing)
  - seq_velocity_accel: ускорение активности (1ч vs 6ч)
  - seq_spend_accel: ускорение трат
  Почему сработает: описывает ПАТТЕРН дня, а не точечную транзакцию

БЛОК 3 — Temporal Reweighting (потенциал +5-15%):
  - sample_weight = exp(-days_to_june_2025 / tau)
  - Последние месяцы train (Apr-May'25) весят больше → ближе к test
  - Тестируем tau = 60, 90, 180 дней
  Почему сработает: борется с distribution shift (adversarial AUC=1.0)

ABLATION (бесплатно на val):
  A: v14-C baseline (reference, ~0.044)
  B: v14-C + device           → вклад device graph
  C: v14-C + sequence         → вклад sequence
  D: v14-C + temporal weights → вклад reweighting (tau=60,90,180)
  E: v14-C + device + seq     → комбо фич
  F: v14-C + всё              → максимальный набор

САБМИТЫ ЗАВТРА (5 штук, по результатам ablation):
  1: лучший одиночный блок
  2: второй лучший
  3: третий
  4: комбо двух лучших
  5: всё что сработало
═══════════════════════════════════════════════════════════════════
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from pathlib import Path
from datetime import datetime
import gc, json, time, math

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_OUT  = ROOT / 'models_v15'
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
# БЛОК 1: Device Graph + Battery
# ═══════════════════════════════════════════════════════════

def build_device_graph():
    """Строим device_fp → n_customers из 85M train строк."""
    cache_path = FEATURES_IN / 'device_graph.parquet'
    if cache_path.exists():
        log('Device graph: загружаем из кэша')
        return pl.read_parquet(cache_path)

    log('=== СТРОИМ DEVICE GRAPH ИЗ 85M СТРОК ===')
    train_files = [
        RAW_TRAIN / 'train_part_1.parquet',
        RAW_TRAIN / 'train_part_2.parquet',
        RAW_TRAIN / 'train_part_3.parquet',
    ]

    # Считаем (device_fp → customer_id) пары
    all_pairs = []
    for f in train_files:
        log(f'  Сканирую {f.name}...')
        df = pl.read_parquet(f, columns=['customer_id', 'screen_size', 'device_system_version'])
        df = df.filter(pl.col('screen_size').is_not_null())
        df = df.with_columns(
            (pl.col('screen_size') + '_' + pl.col('device_system_version').fill_null('?')).alias('device_fp')
        )
        pairs = df.select(['device_fp', 'customer_id']).unique()
        all_pairs.append(pairs)
        del df; gc.collect()

    all_pairs = pl.concat(all_pairs).unique()
    log(f'  Уникальных (device_fp, customer_id) пар: {len(all_pairs):,}')

    # Агрегаты по device_fp
    device_graph = all_pairs.group_by('device_fp').agg([
        pl.col('customer_id').n_unique().alias('dev_n_customers'),
        pl.col('customer_id').alias('dev_customer_list'),
    ])

    # Добавляем fraud-связь: какие device_fp ассоциированы с fraud-клиентами
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_events = labels.filter(pl.col('target') == 1)['event_id'].to_list()

    # Находим customer_id fraud-клиентов
    fraud_customers = set()
    for f in train_files:
        df = pl.read_parquet(f, columns=['customer_id', 'event_id'])
        fc = df.filter(pl.col('event_id').is_in(fraud_events))['customer_id'].unique().to_list()
        fraud_customers.update(fc)
        del df; gc.collect()

    log(f'  Fraud customers: {len(fraud_customers):,}')

    # Для каждого device_fp: сколько fraud-клиентов
    def count_fraud_customers(cust_list):
        return sum(1 for c in cust_list if c in fraud_customers)

    device_graph = device_graph.with_columns(
        pl.col('dev_customer_list').map_elements(
            count_fraud_customers, return_dtype=pl.Int64
        ).alias('dev_n_fraud_customers')
    ).drop('dev_customer_list')

    device_graph = device_graph.with_columns(
        (pl.col('dev_n_fraud_customers') / pl.col('dev_n_customers').clip(lower_bound=1)).alias('dev_fraud_rate')
    )

    device_graph.write_parquet(cache_path)
    log(f'  Device graph: {len(device_graph):,} уникальных device_fp')
    log(f'  10+ customers: {(device_graph["dev_n_customers"] >= 10).sum()}')
    log(f'  С fraud: {(device_graph["dev_n_fraud_customers"] > 0).sum()}')

    return device_graph


def add_device_features(df, device_graph):
    """Добавляем device graph + battery фичи к датафрейму."""

    # Device fingerprint
    df = df.with_columns(
        pl.when(pl.col('screen_size').is_not_null())
        .then(pl.col('screen_size') + '_' + pl.col('device_system_version').fill_null('?'))
        .otherwise(pl.lit(None))
        .alias('device_fp')
    )

    # Join device graph
    df = df.join(device_graph.select(['device_fp', 'dev_n_customers', 'dev_n_fraud_customers', 'dev_fraud_rate']),
                 on='device_fp', how='left')

    df = df.with_columns([
        pl.col('dev_n_customers').fill_null(0).alias('dev_n_customers'),
        pl.col('dev_n_fraud_customers').fill_null(0).alias('dev_n_fraud_customers'),
        pl.col('dev_fraud_rate').fill_null(0.0).alias('dev_fraud_rate'),
        # Is "farm" device (10+ customers)
        (pl.col('dev_n_customers').fill_null(0) >= 10).cast(pl.Int8).alias('dev_is_farm'),
        # Log of n_customers
        pl.col('dev_n_customers').fill_null(0).cast(pl.Float64).log1p().alias('dev_log_n_customers'),
    ])

    # Battery: в preprocessed файлах = float32 (-1.0 = missing)
    # В сырых данных: string "100%", "60%", "not available", null
    # Preprocessed потерял строки → используем числовое значение
    bat_col = df['battery']
    if bat_col.dtype == pl.Float32 or bat_col.dtype == pl.Float64:
        # Preprocessed: -1.0 = missing, остальное - уровень (если есть)
        df = df.with_columns([
            (pl.col('battery') > 99.0).fill_null(False).cast(pl.Int8).alias('bat_is_100'),
            (pl.col('battery') >= 0.0).fill_null(False).cast(pl.Int8).alias('bat_has_real'),
            pl.when(pl.col('battery') >= 0.0).then(pl.col('battery')).otherwise(None).alias('bat_level'),
            pl.col('screen_size').is_not_null().cast(pl.Int8).alias('dev_has_info'),
        ])
    else:
        # Raw string: "100%", "not available", null
        df = df.with_columns([
            (pl.col('battery') == '100%').fill_null(False).cast(pl.Int8).alias('bat_is_100'),
            (pl.col('battery').is_not_null() &
             (pl.col('battery') != 'not available')).cast(pl.Int8).alias('bat_has_real'),
            pl.when(pl.col('battery').is_not_null() & (pl.col('battery') != 'not available'))
            .then(pl.col('battery').str.replace('%', '').str.strip_chars().cast(pl.Float64, strict=False))
            .otherwise(None).alias('bat_level'),
            pl.col('screen_size').is_not_null().cast(pl.Int8).alias('dev_has_info'),
        ])

    # Clean up temporary column
    if 'device_fp' in df.columns:
        df = df.drop('device_fp')

    return df


DEVICE_FEATURES = [
    'dev_n_customers', 'dev_n_fraud_customers', 'dev_fraud_rate',
    'dev_is_farm', 'dev_log_n_customers', 'dev_has_info',
    'bat_is_100', 'bat_has_real', 'bat_level',
]


# ═══════════════════════════════════════════════════════════
# БЛОК 2: Sequence Features (внутри дня)
# ═══════════════════════════════════════════════════════════

def build_prev_tx_lookup():
    """Из 32 чанков строим lookup: event_id → prev_amt, prev_mcc, prev_pos_cd, time_diff."""
    cache_path = FEATURES_IN / 'prev_tx_lookup.parquet'
    if cache_path.exists():
        log('Prev TX lookup: загружаем из кэша')
        return pl.read_parquet(cache_path)

    log('=== СТРОИМ PREV TX LOOKUP ИЗ ЧАНКОВ ===')
    chunk_dir = FEATURES_IN / '_tmp_train'
    chunks = sorted(chunk_dir.glob('chunk_*.parquet'))
    log(f'  Чанков: {len(chunks)}')

    # Нужно: для каждого event_id → предыдущая транзакция того же клиента
    # Чанки могут быть несортированы глобально, поэтому собираем всё
    all_results = []

    for i, chunk_path in enumerate(chunks):
        df = pl.read_parquet(chunk_path,
            columns=['customer_id', 'event_id', 'event_dttm', 'operaton_amt', 'mcc_code', 'pos_cd'])

        # Сортируем по customer_id, event_dttm
        df = df.sort(['customer_id', 'event_dttm'])

        # Предыдущая транзакция того же клиента
        df = df.with_columns([
            pl.col('operaton_amt').shift(1).over('customer_id').alias('prev_amt'),
            pl.col('mcc_code').shift(1).over('customer_id').alias('prev_mcc'),
            pl.col('pos_cd').shift(1).over('customer_id').alias('prev_pos_cd'),
            pl.col('event_dttm').shift(1).over('customer_id').alias('prev_dttm'),
        ])

        result = df.select(['event_id', 'prev_amt', 'prev_mcc', 'prev_pos_cd', 'prev_dttm',
                             'operaton_amt', 'mcc_code', 'pos_cd', 'event_dttm'])
        all_results.append(result)

        if (i + 1) % 8 == 0:
            log(f'  Обработано {i+1}/{len(chunks)} чанков')
        del df; gc.collect()

    log('  Собираем все чанки...')
    combined = pl.concat(all_results)
    del all_results; gc.collect()

    # ВАЖНО: чанки могут быть нарезаны по файлам, не по клиентам
    # Нужно пересортировать и пересчитать для граничных event_id
    log('  Глобальная сортировка по customer_id + event_dttm...')
    # Это тяжёлая операция на 85M строк, но нужна для корректности
    # Для экономии RAM, сохраним только event_id + фичи

    # Вообще-то, нам не нужна глобальная сортировка — достаточно
    # иметь prev_amt для каждого event_id из того же чанка.
    # Граничные ошибки (~3% event_id) — допустимый шум.

    # Считаем sequence features
    combined = combined.with_columns([
        # Ratio to previous amount
        (pl.col('operaton_amt') / pl.col('prev_amt').clip(lower_bound=1.0)).alias('seq_amt_ratio_prev'),

        # Diff from previous
        (pl.col('operaton_amt') - pl.col('prev_amt').fill_null(pl.col('operaton_amt'))).alias('seq_amt_diff_prev'),

        # MCC changed
        (pl.col('mcc_code') != pl.col('prev_mcc')).fill_null(False).cast(pl.Int8).alias('seq_mcc_changed'),

        # POS transition: non-manual → manual (чип → ручной ввод)
        ((pl.col('prev_pos_cd') != 1) & (pl.col('pos_cd') == 1)).fill_null(False).cast(pl.Int8).alias('seq_pos_to_manual'),

        # Card testing: prev_amt < 500 AND curr_amt > 10000
        ((pl.col('prev_amt') < 500) & (pl.col('operaton_amt') > 10000)).fill_null(False).cast(pl.Int8).alias('seq_micro_then_large'),
    ])

    result = combined.select([
        'event_id', 'seq_amt_ratio_prev', 'seq_amt_diff_prev',
        'seq_mcc_changed', 'seq_pos_to_manual', 'seq_micro_then_large',
    ])

    result.write_parquet(cache_path)
    log(f'  Prev TX lookup: {len(result):,} строк')
    return result


def build_test_sequence_features():
    """Sequence features для test — считаем напрямую из test.parquet."""
    cache_path = FEATURES_IN / 'test_seq_features.parquet'
    if cache_path.exists():
        log('Test sequence features: из кэша')
        return pl.read_parquet(cache_path)

    log('=== SEQUENCE FEATURES ДЛЯ TEST ===')
    df = pl.read_parquet(RAW_TEST / 'test.parquet',
        columns=['customer_id', 'event_id', 'event_dttm', 'operaton_amt', 'mcc_code', 'pos_cd'])

    df = df.sort(['customer_id', 'event_dttm'])

    df = df.with_columns([
        pl.col('operaton_amt').shift(1).over('customer_id').alias('prev_amt'),
        pl.col('mcc_code').shift(1).over('customer_id').alias('prev_mcc'),
        pl.col('pos_cd').shift(1).over('customer_id').alias('prev_pos_cd'),
    ])

    df = df.with_columns([
        (pl.col('operaton_amt') / pl.col('prev_amt').clip(lower_bound=1.0)).alias('seq_amt_ratio_prev'),
        (pl.col('operaton_amt') - pl.col('prev_amt').fill_null(pl.col('operaton_amt'))).alias('seq_amt_diff_prev'),
        (pl.col('mcc_code') != pl.col('prev_mcc')).fill_null(False).cast(pl.Int8).alias('seq_mcc_changed'),
        ((pl.col('prev_pos_cd') != 1) & (pl.col('pos_cd') == 1)).fill_null(False).cast(pl.Int8).alias('seq_pos_to_manual'),
        ((pl.col('prev_amt') < 500) & (pl.col('operaton_amt') > 10000)).fill_null(False).cast(pl.Int8).alias('seq_micro_then_large'),
    ])

    result = df.select([
        'event_id', 'seq_amt_ratio_prev', 'seq_amt_diff_prev',
        'seq_mcc_changed', 'seq_pos_to_manual', 'seq_micro_then_large',
    ])

    result.write_parquet(cache_path)
    log(f'  Test sequence features: {len(result):,} строк')
    return result


def add_sequence_features(df, seq_lookup):
    """Добавляем sequence features через join по event_id."""
    df = df.join(seq_lookup, on='event_id', how='left')

    # Fill nulls (первая транзакция клиента)
    df = df.with_columns([
        pl.col('seq_amt_ratio_prev').fill_null(1.0),
        pl.col('seq_amt_diff_prev').fill_null(0.0),
        pl.col('seq_mcc_changed').fill_null(0),
        pl.col('seq_pos_to_manual').fill_null(0),
        pl.col('seq_micro_then_large').fill_null(0),
    ])

    # Дополнительные фичи из существующих колонок
    if 'session_ops_before' in df.columns and 'session_amt_before' in df.columns:
        df = df.with_columns([
            # Средняя сумма в сессии ДО текущей транзакции
            (pl.col('operaton_amt') /
             (pl.col('session_amt_before') / pl.col('session_ops_before').clip(lower_bound=1) + 1.0)
            ).alias('seq_amt_vs_session_avg'),

            # Первая транзакция в сессии — большая сумма?
            ((pl.col('session_ops_before') == 0) &
             (pl.col('operaton_amt') > 50000)).cast(pl.Int8).alias('seq_first_tx_large'),
        ])

    if 'cnt_1h' in df.columns and 'cnt_6h' in df.columns:
        df = df.with_columns([
            # Ускорение активности: последний час vs 6ч среднее
            (pl.col('cnt_1h') / (pl.col('cnt_6h') / 6.0 + 0.01)).alias('seq_velocity_accel'),
        ])

    if 'amt_sum_1h' in df.columns and 'amt_sum_6h' in df.columns:
        df = df.with_columns([
            # Ускорение трат: последний час vs 6ч среднее
            (pl.col('amt_sum_1h') / (pl.col('amt_sum_6h') / 6.0 + 1.0)).alias('seq_spend_accel'),
        ])

    return df


SEQUENCE_FEATURES = [
    'seq_amt_ratio_prev', 'seq_amt_diff_prev',
    'seq_mcc_changed', 'seq_pos_to_manual', 'seq_micro_then_large',
    'seq_amt_vs_session_avg', 'seq_first_tx_large',
    'seq_velocity_accel', 'seq_spend_accel',
]


# ═══════════════════════════════════════════════════════════
# БЛОК 3: Temporal Reweighting
# ═══════════════════════════════════════════════════════════

# Test начинается примерно с Jun 2025
TEST_START_DATE = datetime(2025, 6, 1)

def compute_sample_weights(df, tau_days=90):
    """Вычисляем sample_weight = exp(-days_to_test / tau).
    Транзакции ближе к Jun 2025 получают больший вес."""

    dttm_col = df['event_dttm']
    if dttm_col.dtype == pl.Utf8:
        dates = pd.to_datetime(dttm_col.to_pandas().str[:10])
    else:
        # Уже datetime
        dates = dttm_col.to_pandas()
        if hasattr(dates.dt, 'date'):
            dates = pd.to_datetime(dates.dt.date)

    days_to_test = (TEST_START_DATE - dates).dt.days.clip(lower=0)
    weights = np.exp(-days_to_test.values / tau_days).astype(np.float32)

    # Нормализуем чтобы средний вес ≈ 1
    weights = weights / weights.mean()

    return weights


# ═══════════════════════════════════════════════════════════
# ОБУЧЕНИЕ
# ═══════════════════════════════════════════════════════════

# Import v9 + v14
import importlib.util

spec9 = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
v9 = importlib.util.module_from_spec(spec9)
spec9.loader.exec_module(v9)

spec14 = importlib.util.spec_from_file_location("v14", ROOT / "pipeline_v14.py")
v14 = importlib.util.module_from_spec(spec14)
spec14.loader.exec_module(v14)

seeds = [42, 123, 777]  # 3 seeds для ускорения ablation (5 seeds для финала)

LGBM_PARAMS = dict(
    objective='binary', metric='average_precision',
    device='gpu', gpu_platform_id=0, gpu_device_id=0,
    learning_rate=0.02,
    num_leaves=127, min_child_samples=200,
    subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
    n_jobs=4, verbose=-1,
)


def train_model(name, X_train, y_train, X_val, y_val,
                n_trees=5000, sample_weight=None):
    """3-seed LGBM ensemble. Фиксированные итерации (без early stopping)."""
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    spw = min(spw, 50.0)

    log(f'\n{"="*60}')
    log(f'Training: {name}')
    log(f'Features: {X_train.shape[1]}, Trees: {n_trees}, spw: {spw:.1f}')
    if sample_weight is not None:
        log(f'Sample weights: min={sample_weight.min():.3f}, max={sample_weight.max():.3f}')
    log(f'{"="*60}')

    preds = np.zeros(len(X_val), dtype=np.float64)
    seed_results = []
    importances = None

    for i, seed in enumerate(seeds):
        m = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed,
            scale_pos_weight=spw, n_estimators=n_trees)

        fit_params = dict(
            callbacks=[lgb.log_evaluation(0)],
        )
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight

        m.fit(X_train, y_train, **fit_params)

        # Predict на полном val для оценки
        p = m.predict_proba(X_val)[:, 1]
        preds += p
        prauc = average_precision_score(y_val, p)
        seed_results.append(prauc)
        log(f'  Seed {seed}: iter={n_trees}, val={prauc:.6f}')

        if importances is None:
            importances = m.feature_importances_.copy()
        else:
            importances += m.feature_importances_

        model_path = MODELS_OUT / f'{name}_s{seed}.txt'
        m.booster_.save_model(str(model_path))
        del m; gc.collect()

    preds /= len(seeds)
    ensemble_prauc = average_precision_score(y_val, preds)
    log(f'  Ensemble ({name}): val={ensemble_prauc:.6f}')

    # Top features
    importances = importances / len(seeds)
    feat_imp = sorted(zip(X_train.columns, importances), key=lambda x: -x[1])
    log(f'  Top-10:')
    total_imp = sum(importances)
    for fname, imp in feat_imp[:10]:
        log(f'    {fname:35s}: {imp/total_imp*100:.1f}%')

    # Новые фичи в топ?
    new_feats = [f for f in feat_imp if f[0].startswith(('dev_', 'bat_', 'seq_'))]
    if new_feats:
        log(f'  Новые фичи:')
        for fname, imp in new_feats[:10]:
            log(f'    {fname:35s}: {imp/total_imp*100:.2f}%')

    return ensemble_prauc, preds, seed_results


def generate_submission(name, feats, train_df_cols):
    """Генерация сабмита для эксперимента."""
    log(f'\n  Генерируем сабмит: {name}')

    # Загружаем test features
    test_df = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    test_df = test_df.unique(subset=['event_id'], keep='first')

    # v9 features
    old_profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    test_df = v9.add_features(test_df)
    test_df = v9.add_customer_profiles(test_df, old_profiles)

    # v14 anomaly features
    deep_profiles = pl.read_parquet(FEATURES_IN / 'deep_customer_profiles.parquet')
    mcc_profiles = pl.read_parquet(FEATURES_IN / 'customer_mcc_profiles.parquet')
    if mcc_profiles['mcc_code'].dtype != pl.Int32:
        mcc_profiles = mcc_profiles.with_columns(pl.col('mcc_code').cast(pl.Int32))
    test_df = v14.add_anomaly_features(test_df, deep_profiles, mcc_profiles)

    # v15 device features
    if any(f.startswith('dev_') or f.startswith('bat_') for f in feats):
        device_graph = pl.read_parquet(FEATURES_IN / 'device_graph.parquet')
        test_df = add_device_features(test_df, device_graph)

    # v15 sequence features
    if any(f.startswith('seq_') for f in feats):
        test_seq = build_test_sequence_features()
        test_df = add_sequence_features(test_df, test_seq)

    available_feats = [f for f in feats if f in test_df.columns]
    log(f'  Available features: {len(available_feats)}/{len(feats)}')

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

    path = SUBMIT_OUT / f'submit_v15_{name}_{ts}.csv'
    sub.write_csv(path)
    log(f'  Saved: {path.name}')
    return path


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_start = time.time()
    log('=== PIPELINE V15: DEVICE GRAPH + SEQUENCE + TEMPORAL ===')

    # ── Фаза 1: Строим все lookup-таблицы ──
    device_graph = build_device_graph()
    gc.collect()

    seq_lookup = build_prev_tx_lookup()
    gc.collect()

    log(f'\nRAM после lookup: {_ram_gb():.1f}GB')

    # ── Фаза 2: Загружаем данные и добавляем все фичи ──
    log('\n=== ЗАГРУЗКА ДАННЫХ ===')

    old_profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    deep_profiles = pl.read_parquet(FEATURES_IN / 'deep_customer_profiles.parquet')
    mcc_profiles = pl.read_parquet(FEATURES_IN / 'customer_mcc_profiles.parquet')
    if mcc_profiles['mcc_code'].dtype != pl.Int32:
        mcc_profiles = mcc_profiles.with_columns(pl.col('mcc_code').cast(pl.Int32))
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())

    # Val
    val_df = pl.read_parquet(FEATURES_IN / 'val_proper.parquet')
    val_df = val_df.with_columns(
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'))
    val_df = v9.add_features(val_df)
    val_df = v9.add_customer_profiles(val_df, old_profiles)
    val_df = v14.add_anomaly_features(val_df, deep_profiles, mcc_profiles)
    val_df = add_device_features(val_df, device_graph)
    val_df = add_sequence_features(val_df, seq_lookup)

    # Train
    val_ids = set(pl.read_parquet(FEATURES_IN / 'val_event_ids.parquet')['event_id'].to_list())
    train_df = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    train_df = train_df.filter(~pl.col('event_id').is_in(val_ids))
    train_df = v9.add_features(train_df)
    train_df = v9.add_customer_profiles(train_df, old_profiles)
    train_df = v14.add_anomaly_features(train_df, deep_profiles, mcc_profiles)
    train_df = add_device_features(train_df, device_graph)
    train_df = add_sequence_features(train_df, seq_lookup)

    del old_profiles, deep_profiles, mcc_profiles, device_graph, seq_lookup
    gc.collect()

    # ── Feature lists ──
    base_feats = [c for c in v9.FEATURE_COLS if c in train_df.columns and c in val_df.columns]
    anom_feats = [c for c in v14.ANOMALY_FEATURES if c in train_df.columns and c in val_df.columns]
    dev_feats = [c for c in DEVICE_FEATURES if c in train_df.columns and c in val_df.columns]
    seq_feats = [c for c in SEQUENCE_FEATURES if c in train_df.columns and c in val_df.columns]

    v14c_feats = base_feats + anom_feats  # = v14-C baseline (121 фич)
    all_feats = v14c_feats + dev_feats + seq_feats  # = всё

    y_train = train_df['target'].to_numpy().astype(int)
    y_val = val_df['is_fraud'].to_numpy().astype(int)

    log(f'\nTrain: {len(train_df):,}, Val: {len(val_df):,}')
    log(f'Val fraud: {y_val.sum()} ({100*y_val.mean():.4f}%)')
    log(f'Base: {len(base_feats)}, Anomaly: {len(anom_feats)}, Device: {len(dev_feats)}, Sequence: {len(seq_feats)}')
    log(f'v14-C total: {len(v14c_feats)}, v15 total: {len(all_feats)}')

    results = {}

    # Освобождаем Polars DataFrames — дальше работаем с pandas
    # Сначала конвертируем всё нужное
    log('\nКонвертация в pandas...')
    X_tr_all = train_df.select(all_feats).to_pandas().astype(np.float32)
    X_va_all = val_df.select(all_feats).to_pandas().astype(np.float32)

    # Освобождаем Polars
    del train_df, val_df; gc.collect()
    log(f'RAM после конверсии: {_ram_gb():.1f}GB')

    # ═══ ABLATION: 3 ЭКСПЕРИМЕНТА (фиксированные 5000 итераций) ═══

    # A: v14-C baseline (121 фич) — reference
    log('\n>>> EXP A: v14-C baseline (121 фич, 5K iter) <<<')
    prauc_a, _, _ = train_model('A_v14c_baseline',
        X_tr_all[v14c_feats], y_train, X_va_all[v14c_feats], y_val)
    results['A_v14c_baseline'] = prauc_a

    # B: v14-C + device + sequence (139 фич)
    log('\n>>> EXP B: ALL features (v14-C + device + sequence, 139 фич, 5K iter) <<<')
    prauc_b, _, _ = train_model('B_all_features',
        X_tr_all, y_train, X_va_all, y_val)
    results['B_all_features'] = prauc_b

    # D: v14-C + только device (130 фич)
    log('\n>>> EXP D: v14-C + ТОЛЬКО device фичи (130, 5K iter) <<<')
    d_feats = v14c_feats + dev_feats
    prauc_d, _, _ = train_model('D_device_only',
        X_tr_all[d_feats], y_train, X_va_all[d_feats], y_val)
    results['D_device_only'] = prauc_d

    del X_tr_all, X_va_all; gc.collect()

    # ═══ ИТОГИ ═══
    log(f'\n{"="*60}')
    log('ИТОГИ V15 ABLATION')
    log(f'{"="*60}')

    sorted_res = sorted(results.items(), key=lambda x: -x[1])
    for name, prauc in sorted_res:
        delta = 100 * (prauc / prauc_a - 1) if prauc_a > 0 else 0
        marker = ' <<<' if prauc > prauc_a else ''
        log(f'{name:25s}: val={prauc:.6f} ({delta:+.1f}% vs baseline){marker}')

    # Сохраняем результаты
    with open(MODELS_OUT / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Генерируем сабмиты для лучших экспериментов
    log('\n=== ГЕНЕРАЦИЯ САБМИТОВ ===')
    feat_map = {
        'A_v14c_baseline': v14c_feats,
        'B_all_features': all_feats,
        'C_all_temporal': all_feats,
        'D_device_only': v14c_feats + dev_feats,
    }

    for name, prauc in sorted_res[:3]:
        generate_submission(name, feat_map[name], [])

    total_min = (time.time() - t_start) / 60
    log(f'\nВсего: {total_min:.0f} мин')
    log('DONE')
