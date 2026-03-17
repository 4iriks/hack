"""
Pipeline v12: Full-Day Train + Hard Negatives + Day-Level Features + Stacking.

Последовательные эксперименты, одно изменение за раз:
  Шаг 2: Полные дни в train (вместо случайных строк)
  Шаг 3: + Hard negative mining (подозрительные green)
  Шаг 4: + Day-level фичи (контекст дня)
  Шаг 5: + Day-level стекинг (2я модель поверх)
  Шаг 6: + Customer-day boost (пост-процессинг)

Baseline: v10 val = 0.039, LB = 0.097
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from pathlib import Path
from datetime import datetime
import gc, json, time, os

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
CHUNKS_DIR  = FEATURES_IN / '_tmp_train'
MODELS_OUT  = ROOT / 'models_v12'
SUBMIT_OUT  = ROOT / 'submissions'

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

# Import feature engineering from v9
import importlib.util
spec = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
v9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9)

seeds = [42, 123, 777, 2024, 31337]

LGBM_PARAMS = dict(
    objective='binary', metric='average_precision',
    device='gpu', gpu_platform_id=0, gpu_device_id=0,
    learning_rate=0.02,
    num_leaves=127, min_child_samples=50,
    subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
    n_jobs=4, verbose=-1,
)


# ═══════════════════════════════════════════════════════
# Шаг 2: Собрать train из ПОЛНЫХ ДНЕЙ
# ═══════════════════════════════════════════════════════

def build_fullday_train(green_day_ratio=10, green_seed=42):
    """
    Собрать train из полных дней:
    - Для labeled клиентов: ВСЕ строки в день с событием
    - Для green клиентов: случайные полные дни (ratio green_days : labeled_days)

    Исключаем val event_ids.
    """
    cache_path = FEATURES_IN / 'train_fullday.parquet'
    if cache_path.exists():
        log(f'Загружаю кэш: {cache_path}')
        return pl.read_parquet(cache_path)

    t0 = time.time()
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    val_ids = set(pl.read_parquet(FEATURES_IN / 'val_event_ids.parquet')['event_id'].to_list())
    label_ids = set(labels['event_id'].to_list())
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())
    confirmed_ids = set(labels.filter(pl.col('target') == 0)['event_id'].to_list())
    labeled_custs = set(labels['customer_id'].to_list())

    chunk_files = sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith('.parquet')])
    log(f'Шаг 2: Сборка full-day train из {len(chunk_files)} чанков')

    # Проход 1: Найти (customer_id, date) для labeled событий
    log('Проход 1: Ищем даты labeled событий...')
    labeled_dates = set()  # (customer_id, date)
    for i, cf in enumerate(chunk_files):
        df = pl.read_parquet(CHUNKS_DIR / cf, columns=['customer_id', 'event_id', 'event_dttm'])
        # Находим labeled события (исключая val)
        labeled_rows = df.filter(
            pl.col('event_id').is_in(label_ids) & ~pl.col('event_id').is_in(val_ids)
        )
        if len(labeled_rows) > 0:
            dates = labeled_rows.with_columns(
                pl.col('event_dttm').cast(pl.Date).alias('date')
            ).select(['customer_id', 'date']).unique()
            for row in dates.iter_rows():
                labeled_dates.add(row)
        del df, labeled_rows; gc.collect()
        if (i + 1) % 8 == 0:
            log(f'  Чанк {i+1}/{len(chunk_files)}, найдено {len(labeled_dates)} labeled (cust, date)')

    log(f'Labeled (customer, date) пар: {len(labeled_dates)}')

    # Проход 1.5: Собрать все (customer_id, date) для green клиентов
    log('Проход 1.5: Собираем green (customer, date) пары...')
    green_dates = []  # list of (customer_id, date)
    for i, cf in enumerate(chunk_files):
        df = pl.read_parquet(CHUNKS_DIR / cf, columns=['customer_id', 'event_dttm'])
        df = df.with_columns(pl.col('event_dttm').cast(pl.Date).alias('date'))
        cust_dates = df.select(['customer_id', 'date']).unique()
        # Оставляем только green клиентов (НЕ labeled)
        green_cd = cust_dates.filter(~pl.col('customer_id').is_in(labeled_custs))
        for row in green_cd.iter_rows():
            green_dates.append(row)
        del df, cust_dates, green_cd; gc.collect()
        if (i + 1) % 8 == 0:
            log(f'  Чанк {i+1}/{len(chunk_files)}, green (cust,date) пар: {len(green_dates)}, RAM: {_ram_gb():.1f}GB')

    log(f'Green (customer, date) пар: {len(green_dates)}')

    # Сэмплируем green дни
    np.random.seed(green_seed)
    n_green_days = min(len(labeled_dates) * green_day_ratio, len(green_dates))
    green_idx = np.random.choice(len(green_dates), size=n_green_days, replace=False)
    sampled_green = set(green_dates[i] for i in green_idx)
    del green_dates; gc.collect()
    log(f'Сэмплировано green дней: {len(sampled_green)} (ratio {green_day_ratio}:1)')

    # Объединяем: все labeled + sampled green дни
    all_target_days = labeled_dates | sampled_green
    # Преобразуем в DataFrame для джоина
    target_days_df = pl.DataFrame(
        list(all_target_days),
        schema={'customer_id': pl.Int64, 'date': pl.Date},
        orient='row'
    )
    log(f'Всего (cust, date) для извлечения: {len(target_days_df)}')
    del all_target_days, sampled_green, labeled_dates; gc.collect()

    # Проход 2: Извлечь ВСЕ строки для целевых (customer_id, date)
    log('Проход 2: Извлекаем полные дни...')
    parts = []
    total_rows = 0
    for i, cf in enumerate(chunk_files):
        df = pl.read_parquet(CHUNKS_DIR / cf)
        df = df.with_columns(pl.col('event_dttm').cast(pl.Date).alias('_date'))
        matched = df.join(
            target_days_df,
            left_on=['customer_id', '_date'],
            right_on=['customer_id', 'date'],
            how='inner'
        ).drop('_date')
        # Исключаем val event_ids
        matched = matched.filter(~pl.col('event_id').is_in(val_ids))
        if len(matched) > 0:
            parts.append(matched)
            total_rows += len(matched)
        del df, matched; gc.collect()
        if (i + 1) % 8 == 0:
            log(f'  Чанк {i+1}/{len(chunk_files)}: {total_rows:,} строк, RAM: {_ram_gb():.1f}GB')

    train_df = pl.concat(parts)
    del parts; gc.collect()

    # Добавляем target
    train_df = train_df.with_columns(
        pl.when(pl.col('event_id').is_in(fraud_ids)).then(1)
        .when(pl.col('event_id').is_in(confirmed_ids)).then(0)
        .otherwise(0)
        .cast(pl.Int8).alias('target')
    )

    n_fraud = (train_df['target'] == 1).sum()
    n_total = len(train_df)
    n_custs = train_df['customer_id'].n_unique()
    log(f'Full-day train: {n_total:,} строк, {n_fraud:,} fraud, {n_custs:,} клиентов')
    log(f'Fraud rate: {100*n_fraud/n_total:.4f}%')

    # Сохраняем кэш
    train_df.write_parquet(cache_path)
    log(f'Сохранён: {cache_path} ({time.time()-t0:.0f}s)')

    return train_df


# ═══════════════════════════════════════════════════════
# Шаг 4: Day-level фичи
# ═══════════════════════════════════════════════════════

def add_day_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Добавить фичи, описывающие контекст ВСЕГО дня клиента.
    Считаются из строк того же DataFrame (согласованно с train).
    """
    df = df.with_columns(pl.col('event_dttm').cast(pl.Date).alias('_day'))

    # Агрегаты дня для каждого (customer_id, day)
    day_agg = df.group_by(['customer_id', '_day']).agg([
        pl.len().alias('day_n_ops'),
        pl.col('operaton_amt').sum().alias('day_total_amt'),
        pl.col('operaton_amt').max().alias('day_max_amt'),
        pl.col('operaton_amt').mean().alias('day_mean_amt'),
        pl.col('operaton_amt').std().alias('day_std_amt'),
        pl.col('mcc_code').n_unique().alias('day_n_unique_mcc'),
        pl.col('event_type_nm').n_unique().alias('day_n_unique_event_type'),
        pl.col('channel_indicator_type').n_unique().alias('day_n_unique_channel'),
        pl.col('hour').min().alias('day_first_hour'),
        pl.col('hour').max().alias('day_last_hour'),
        (pl.col('hour').max() - pl.col('hour').min()).alias('day_hour_span'),
        pl.col('compromised').sum().alias('day_compromised_sum'),
        pl.col('web_rdp_connection').sum().alias('day_rdp_sum'),
        pl.col('phone_voip_call_state').sum().alias('day_voip_sum'),
        pl.col('developer_tools').sum().alias('day_devtools_sum'),
    ])

    # Джоиним обратно
    df = df.join(day_agg, on=['customer_id', '_day'], how='left')

    # Производные day-level фичи
    new_cols = []
    # Текущая транзакция относительно дня
    if 'day_total_amt' in df.columns and 'operaton_amt' in df.columns:
        new_cols.append(
            (pl.col('operaton_amt') / (pl.col('day_total_amt') + 1)).alias('day_amt_fraction'))
    if 'day_max_amt' in df.columns and 'operaton_amt' in df.columns:
        new_cols.append(
            (pl.col('operaton_amt') >= pl.col('day_max_amt')).cast(pl.Int8).alias('day_is_max_tx'))
    # День необычный для клиента?
    if 'day_n_ops' in df.columns and 'cnt_30d' in df.columns:
        new_cols.append(
            (pl.col('day_n_ops').cast(pl.Float64) / (pl.col('cnt_30d') / 30 + 0.1)).alias('day_ops_vs_avg'))
    if 'day_total_amt' in df.columns and 'amt_sum_30d' in df.columns:
        new_cols.append(
            (pl.col('day_total_amt') / (pl.col('amt_sum_30d') / 30 + 1)).alias('day_amt_vs_avg'))
    # Security flags за день
    new_cols.append(
        (pl.col('day_compromised_sum') + pl.col('day_rdp_sum') +
         pl.col('day_voip_sum') + pl.col('day_devtools_sum')).alias('day_security_sum'))

    if new_cols:
        df = df.with_columns(new_cols)

    df = df.drop('_day')
    return df


DAY_FEATURE_COLS = [
    'day_n_ops', 'day_total_amt', 'day_max_amt', 'day_mean_amt', 'day_std_amt',
    'day_n_unique_mcc', 'day_n_unique_event_type', 'day_n_unique_channel',
    'day_first_hour', 'day_last_hour', 'day_hour_span',
    'day_compromised_sum', 'day_rdp_sum', 'day_voip_sum', 'day_devtools_sum',
    'day_amt_fraction', 'day_is_max_tx', 'day_ops_vs_avg', 'day_amt_vs_avg',
    'day_security_sum',
]


# ═══════════════════════════════════════════════════════
# Обучение и оценка
# ═══════════════════════════════════════════════════════

def load_val():
    """Загрузить proper val (из v10)."""
    val_df = pl.read_parquet(FEATURES_IN / 'val_proper.parquet')
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())
    val_df = val_df.with_columns(
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'))
    return val_df


def prepare_features(df, profiles, use_day_features=False):
    """Подготовить фичи: v9 base + опционально day-level."""
    df = v9.add_features(df)
    df = v9.add_customer_profiles(df, profiles)
    if use_day_features:
        df = add_day_features(df)
    return df


def get_feature_list(use_day_features=False):
    """Список фичей для модели."""
    feats = list(v9.FEATURE_COLS)
    if use_day_features:
        feats += DAY_FEATURE_COLS
    return feats


def train_lgbm(X_tr, y_tr, X_val, y_val, seed, spw, n_estimators=50000, tag=''):
    """Обучить один LGBM с early stopping."""
    m = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed,
        scale_pos_weight=spw, n_estimators=n_estimators)
    m.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(0)])
    best_iter = m.best_iteration_
    preds = m.predict_proba(X_val)[:, 1]
    prauc = average_precision_score(y_val, preds)
    log(f'  {tag} Seed {seed}: iter={best_iter}, val={prauc:.6f}')
    return m, preds, best_iter


def run_experiment(train_df, val_df, profiles, use_day_features=False, tag='', save_prefix=''):
    """Обучить 5 seeds, вернуть ensemble val PR-AUC."""
    t0 = time.time()
    log(f'\n{"="*60}')
    log(f'Эксперимент: {tag}')
    log(f'{"="*60}')

    # Фичи
    train_feat = prepare_features(train_df.clone(), profiles, use_day_features)
    val_feat = prepare_features(val_df.clone(), profiles, use_day_features)

    feat_list = get_feature_list(use_day_features)
    available = [c for c in feat_list if c in train_feat.columns and c in val_feat.columns]
    log(f'Фичей: {len(available)}')

    X_tr = train_feat.select(available).to_pandas().astype(np.float32)
    y_tr = train_feat['target'].to_numpy().astype(int)
    X_val_np = val_feat.select(available).to_pandas().astype(np.float32)
    y_val_np = val_feat['is_fraud'].to_numpy().astype(int)

    spw_raw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    spw = min(spw_raw, 50.0)  # GPU падает при spw > ~60
    log(f'Train: {X_tr.shape}, fraud={y_tr.sum():,}, spw_raw={spw_raw:.1f}, spw_capped={spw:.1f}')
    log(f'Val: {X_val_np.shape}, fraud={y_val_np.sum():,}')

    del train_feat; gc.collect()

    # Обучение 5 seeds
    all_preds = np.zeros(len(X_val_np), dtype=np.float64)
    models = []
    for seed in seeds:
        m, preds, best_iter = train_lgbm(
            X_tr, y_tr, X_val_np, y_val_np, seed, spw, tag=tag)
        all_preds += preds
        if save_prefix:
            m.booster_.save_model(str(MODELS_OUT / f'{save_prefix}_s{seed}.txt'))
        models.append(m)
        del m; gc.collect()

    all_preds /= len(seeds)
    ensemble_prauc = average_precision_score(y_val_np, all_preds)
    log(f'{tag} Ensemble val PR-AUC = {ensemble_prauc:.6f}')
    log(f'{tag} Время: {(time.time()-t0)/60:.1f} мин')

    return ensemble_prauc, all_preds, available, val_feat


# ═══════════════════════════════════════════════════════
# Шаг 3: Hard Negative Mining
# ═══════════════════════════════════════════════════════

def add_hard_negatives(train_df, profiles, top_pct=5):
    """
    Заменить часть random green на hard negatives.
    Используем v10 модель для скоринга green строк, берём top N%.
    """
    log(f'\nШаг 3: Hard Negative Mining (top {top_pct}% green)')

    # Загружаем v10 модель
    v10_dir = ROOT / 'models_v10'

    # Подготовим фичи для скоринга
    train_feat = prepare_features(train_df.clone(), profiles, use_day_features=False)
    v10_feats = [c for c in v9.FEATURE_COLS if c in train_feat.columns]

    # Скорим green строки v10 моделью
    green_mask = train_df['target'].to_numpy() == 0
    labeled_mask = ~green_mask

    X_green = train_feat.filter(pl.col('target') == 0).select(v10_feats).to_pandas().astype(np.float32)

    v10_preds = np.mean([
        lgb.Booster(model_file=str(v10_dir / f'm1_lgbm_s{s}.txt')).predict(X_green)
        for s in seeds], axis=0)

    # Берём top N% как hard negatives
    threshold = np.percentile(v10_preds, 100 - top_pct)
    hard_mask = v10_preds >= threshold
    n_hard = hard_mask.sum()
    log(f'  Green строк: {len(X_green):,}, hard negatives (top {top_pct}%): {n_hard:,}')
    log(f'  Порог скора: {threshold:.6f}')

    # Собираем: все labeled + hard negatives
    green_df = train_df.filter(pl.col('target') == 0)
    hard_green = green_df.filter(
        pl.Series('_hard', hard_mask)
    )
    labeled_df = train_df.filter(pl.col('target') == 1)

    # Также добавляем random green (чтобы модель видела и лёгкие примеры)
    easy_green = green_df.filter(~pl.Series('_hard', hard_mask))
    n_easy_sample = min(len(labeled_df) * 5, len(easy_green))  # 5:1 easy green
    easy_sample = easy_green.sample(n=n_easy_sample, seed=42)

    new_train = pl.concat([labeled_df, hard_green, easy_sample])
    new_train = new_train.sample(fraction=1.0, seed=42)  # shuffle

    n_new = len(new_train)
    n_fraud = (new_train['target'] == 1).sum()
    log(f'  Новый train: {n_new:,} строк, {n_fraud:,} fraud')
    log(f'  Hard green: {len(hard_green):,}, Easy green sample: {len(easy_sample):,}')

    del train_feat, X_green, v10_preds, green_df, easy_green; gc.collect()
    return new_train


# ═══════════════════════════════════════════════════════
# Шаг 5: Day-level стекинг
# ═══════════════════════════════════════════════════════

def day_level_stacking(val_feat, val_preds, y_val):
    """
    Вторая модель на уровне (customer, day).
    Вход: агрегаты скоров base модели + day-level фичи.
    """
    log(f'\nШаг 5: Day-level стекинг')

    val_df = val_feat.with_columns([
        pl.Series('base_score', val_preds),
        pl.Series('is_fraud', y_val),
        pl.col('event_dttm').cast(pl.Date).alias('_day'),
    ])

    # Агрегируем скоры base модели по (customer, day)
    day_scores = val_df.group_by(['customer_id', '_day']).agg([
        pl.col('base_score').max().alias('day_score_max'),
        pl.col('base_score').mean().alias('day_score_mean'),
        pl.col('base_score').std().alias('day_score_std'),
        pl.col('base_score').min().alias('day_score_min'),
        (pl.col('base_score').max() - pl.col('base_score').min()).alias('day_score_range'),
        pl.col('base_score').quantile(0.9).alias('day_score_p90'),
        pl.len().alias('day_n_scored'),
        pl.col('is_fraud').max().alias('day_has_fraud'),
    ])

    # Джоиним обратно к каждой транзакции
    val_stacked = val_df.join(day_scores, on=['customer_id', '_day'], how='left')

    # Финальный скор: base_score × day_score_max
    stacked_score = val_stacked['base_score'].to_numpy() * val_stacked['day_score_max'].to_numpy()
    prauc = average_precision_score(y_val, stacked_score)
    log(f'  score × day_max: val={prauc:.6f}')

    # Другие комбинации
    s2 = val_stacked['base_score'].to_numpy() * (val_stacked['day_score_mean'].to_numpy() ** 0.5)
    prauc2 = average_precision_score(y_val, s2)
    log(f'  score × day_mean^0.5: val={prauc2:.6f}')

    s3 = val_stacked['base_score'].to_numpy() + val_stacked['day_score_max'].to_numpy()
    prauc3 = average_precision_score(y_val, s3)
    log(f'  score + day_max: val={prauc3:.6f}')

    # Оптимальный alpha для score × day_max^alpha
    from scipy.optimize import minimize_scalar
    base = val_stacked['base_score'].to_numpy()
    day_max = val_stacked['day_score_max'].to_numpy()

    def neg_prauc(alpha):
        s = base * (day_max ** alpha)
        return -average_precision_score(y_val, s)

    opt = minimize_scalar(neg_prauc, bounds=(0.1, 3.0), method='bounded')
    best_alpha = opt.x
    best_prauc = -opt.fun
    log(f'  score × day_max^{best_alpha:.2f}: val={best_prauc:.6f} ← оптимум')

    return best_prauc, best_alpha


# ═══════════════════════════════════════════════════════
# Шаг 6: Customer-day boost
# ═══════════════════════════════════════════════════════

def customer_day_boost(val_df, preds, y_val):
    """Post-processing: score × customer_max."""
    log(f'\nШаг 6: Customer-day boost')
    customer_ids = val_df['customer_id'].to_numpy()
    unique_custs = np.unique(customer_ids)

    boosted = preds.copy()
    for cid in unique_custs:
        mask = customer_ids == cid
        cust_max = preds[mask].max()
        boosted[mask] = preds[mask] * cust_max

    prauc = average_precision_score(y_val, boosted)
    log(f'  score × cust_max: val={prauc:.6f}')

    # С оптимальным alpha
    from scipy.optimize import minimize_scalar
    def neg_prauc(alpha):
        b = preds.copy()
        for cid in unique_custs:
            mask = customer_ids == cid
            cust_max = preds[mask].max()
            b[mask] = preds[mask] * (cust_max ** alpha)
        return -average_precision_score(y_val, b)

    opt = minimize_scalar(neg_prauc, bounds=(0.1, 3.0), method='bounded')
    best_alpha = opt.x
    best_prauc = -opt.fun
    log(f'  score × cust_max^{best_alpha:.2f}: val={best_prauc:.6f} ← оптимум')

    return best_prauc, best_alpha


# ═══════════════════════════════════════════════════════
# MAIN: Последовательные эксперименты
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    t_total = time.time()
    results = {}

    profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    val_df = load_val()

    log(f'Val: {len(val_df):,} строк, {val_df["is_fraud"].sum():,} fraud')

    # ─── Шаг 2: Полные дни ───
    log('\n' + '='*60)
    log('ШАГ 2: Train из полных дней')
    log('='*60)

    train_fullday = build_fullday_train(green_day_ratio=10)

    prauc_s2, preds_s2, feats_s2, val_feat_s2 = run_experiment(
        train_fullday, val_df, profiles,
        use_day_features=False,
        tag='Шаг2_fullday',
        save_prefix='s2_lgbm'
    )
    results['step2_fullday'] = prauc_s2
    log(f'>>> Шаг 2 vs baseline: {prauc_s2:.6f} vs 0.039 ({"+" if prauc_s2 > 0.039 else "-"}{abs(prauc_s2-0.039):.6f})')

    # ─── Шаг 3: Hard negatives ───
    log('\n' + '='*60)
    log('ШАГ 3: + Hard Negative Mining')
    log('='*60)

    train_hard = add_hard_negatives(train_fullday, profiles, top_pct=5)

    prauc_s3, preds_s3, feats_s3, val_feat_s3 = run_experiment(
        train_hard, val_df, profiles,
        use_day_features=False,
        tag='Шаг3_hard_neg',
        save_prefix='s3_lgbm'
    )
    results['step3_hard_neg'] = prauc_s3

    # Выбираем лучший из шагов 2-3
    if prauc_s3 > prauc_s2:
        best_train = train_hard
        best_preds = preds_s3
        best_feats = feats_s3
        best_val_feat = val_feat_s3
        best_tag = 'Шаг3'
        best_prauc = prauc_s3
        log(f'>>> Hard negatives ПОМОГЛИ: {prauc_s3:.6f} > {prauc_s2:.6f}')
    else:
        best_train = train_fullday
        best_preds = preds_s2
        best_feats = feats_s2
        best_val_feat = val_feat_s2
        best_tag = 'Шаг2'
        best_prauc = prauc_s2
        log(f'>>> Hard negatives НЕ помогли: {prauc_s3:.6f} <= {prauc_s2:.6f}, откат к Шагу 2')

    del train_hard; gc.collect()

    # ─── Шаг 4: Day-level фичи ───
    log('\n' + '='*60)
    log('ШАГ 4: + Day-level фичи')
    log('='*60)

    prauc_s4, preds_s4, feats_s4, val_feat_s4 = run_experiment(
        best_train, val_df, profiles,
        use_day_features=True,
        tag='Шаг4_day_feats',
        save_prefix='s4_lgbm'
    )
    results['step4_day_feats'] = prauc_s4

    if prauc_s4 > best_prauc:
        best_preds = preds_s4
        best_feats = feats_s4
        best_val_feat = val_feat_s4
        best_tag = 'Шаг4'
        best_prauc = prauc_s4
        log(f'>>> Day-level фичи ПОМОГЛИ: {prauc_s4:.6f}')
    else:
        log(f'>>> Day-level фичи НЕ помогли: {prauc_s4:.6f} <= {best_prauc:.6f}, откат')

    # ─── Шаг 5: Day-level стекинг ───
    prauc_s5, alpha_s5 = day_level_stacking(best_val_feat, best_preds,
                                             val_df['is_fraud'].to_numpy().astype(int))
    results['step5_stacking'] = prauc_s5
    results['step5_alpha'] = alpha_s5

    # ─── Шаг 6: Customer-day boost ───
    prauc_s6, alpha_s6 = customer_day_boost(val_df, best_preds,
                                             val_df['is_fraud'].to_numpy().astype(int))
    results['step6_boost'] = prauc_s6
    results['step6_alpha'] = alpha_s6

    # ═══════════════════════════════════════════════════════
    # Итоги
    # ═══════════════════════════════════════════════════════
    log('\n' + '='*60)
    log('ИТОГИ ВСЕХ ЭКСПЕРИМЕНТОВ')
    log('='*60)
    log(f'  Baseline v10:           val = 0.039000')
    for name, prauc in sorted(results.items(), key=lambda x: -x[1] if isinstance(x[1], float) and x[1] > 0.001 else 0):
        if isinstance(prauc, float) and prauc > 0.001:
            delta = prauc - 0.039
            log(f'  {name:25s}: val = {prauc:.6f} ({delta:+.6f}, {100*delta/0.039:+.1f}%)')

    # Сохраняем результаты
    with open(MODELS_OUT / 'experiment_results.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (float, np.floating)) else v
                   for k, v in results.items()}, f, indent=2)

    log(f'\nОбщее время: {(time.time()-t_total)/60:.1f} мин')
    log('DONE')
