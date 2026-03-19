"""
v18b: Гибрид — v14-C pipeline + honest_dormancy_days.

Стратегия:
  - Все профили (prof_*, anom_*) — "жирные" из deep_customer_profiles (как v14-C)
  - dormancy_days — честная: time since LAST known transaction (shift(1) в train,
    last pretest/train epoch для test)
  - Распределения dormancy совпадают в train и test → модель применяет правильные сплиты

Train/Val dormancy: из honest_pit_profiles.parquet (PIT shift(1))
Test dormancy: test_epoch - max(epoch from pretrain+train+pretest WHERE epoch < test_epoch)
"""
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from pathlib import Path
from datetime import datetime
import gc, time, importlib.util

ROOT        = Path('/home/vadim/PyPr/hak')
RAW         = ROOT / 'Pre-train_Train'
PRETEST_DIR = ROOT / 'Pre-test_Test'
FEATURES    = ROOT / 'features'
DATA        = ROOT / 'main_data'
MODELS_OUT  = ROOT / 'models_v18b'
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


def build_test_dormancy_lookup():
    """
    Для каждого test event_id вычисляет honest dormancy:
    test_epoch - last_known_epoch (из pretrain+train+pretest, строго < test_epoch)
    """
    log('Строим test dormancy lookup...')

    # 1. Загрузка test для получения (customer_id, event_id, epoch)
    test = pl.read_parquet(PRETEST_DIR / 'test.parquet',
                           columns=['customer_id', 'event_id', 'event_dttm'])
    test = test.with_columns(
        pl.col('event_dttm').str.to_datetime('%Y-%m-%d %H:%M:%S')
            .dt.epoch('s').alias('test_epoch')
    )
    log(f'  Test: {len(test):,} строк')

    # 2. Для каждого клиента — начало их test-дня (min epoch)
    #    Все pretest/train/pretrain данные ДО этого момента = "прошлое"
    test_day_start = test.group_by('customer_id').agg(
        pl.col('test_epoch').min().alias('test_day_start')
    )

    # 3. Pretest: последняя транзакция ПЕРЕД test-днём
    log('  Сканирую pretest...')
    pretest = pl.read_parquet(PRETEST_DIR / 'pretest.parquet',
                              columns=['customer_id', 'event_dttm'])
    pretest = pretest.with_columns(
        pl.col('event_dttm').str.to_datetime('%Y-%m-%d %H:%M:%S')
            .dt.epoch('s').alias('epoch')
    ).drop('event_dttm')

    # Джойн с test_day_start, фильтр epoch < test_day_start
    pretest = pretest.join(test_day_start, on='customer_id', how='inner')
    pretest = pretest.filter(pl.col('epoch') < pl.col('test_day_start'))
    last_pretest = pretest.group_by('customer_id').agg(
        pl.col('epoch').max().alias('last_pretest_epoch')
    )
    del pretest; gc.collect()
    log(f'  Pretest last epoch: {len(last_pretest):,} клиентов')

    # 4. Train: последняя транзакция (до May 2025)
    log('  Сканирую train файлы для last epoch...')
    train_files = sorted(RAW.glob('train_*.parquet'))
    train_lasts = []
    for f in train_files:
        df = pl.read_parquet(f, columns=['customer_id', 'event_dttm'])
        df = df.with_columns(
            pl.col('event_dttm').str.to_datetime('%Y-%m-%d %H:%M:%S')
                .dt.epoch('s').alias('epoch')
        ).drop('event_dttm')
        agg = df.group_by('customer_id').agg(pl.col('epoch').max().alias('last_train_epoch'))
        train_lasts.append(agg)
        del df; gc.collect()

    last_train = pl.concat(train_lasts).group_by('customer_id').agg(
        pl.col('last_train_epoch').max()
    )
    del train_lasts; gc.collect()
    log(f'  Train last epoch: {len(last_train):,} клиентов')

    # 5. Pretrain: последняя транзакция (до Sep 2024) — fallback
    log('  Сканирую pretrain файлы для last epoch...')
    pretrain_files = sorted(RAW.glob('pretrain_*.parquet'))
    pt_lasts = []
    for f in pretrain_files:
        df = pl.read_parquet(f, columns=['customer_id', 'event_dttm'])
        df = df.with_columns(
            pl.col('event_dttm').str.to_datetime('%Y-%m-%d %H:%M:%S')
                .dt.epoch('s').alias('epoch')
        ).drop('event_dttm')
        agg = df.group_by('customer_id').agg(pl.col('epoch').max().alias('last_pretrain_epoch'))
        pt_lasts.append(agg)
        del df; gc.collect()

    last_pretrain = pl.concat(pt_lasts).group_by('customer_id').agg(
        pl.col('last_pretrain_epoch').max()
    )
    del pt_lasts; gc.collect()

    # 6. Мержим: приоритет pretest > train > pretrain
    lookup = test_day_start.join(last_pretest, on='customer_id', how='left')
    lookup = lookup.join(last_train, on='customer_id', how='left')
    lookup = lookup.join(last_pretrain, on='customer_id', how='left')

    lookup = lookup.with_columns(
        pl.coalesce([
            pl.col('last_pretest_epoch'),
            pl.col('last_train_epoch'),
            pl.col('last_pretrain_epoch'),
        ]).alias('last_known_epoch')
    )
    del last_pretest, last_train, last_pretrain; gc.collect()

    # Статистика
    has_pretest = lookup.filter(pl.col('last_pretest_epoch').is_not_null()).height
    has_train = lookup.filter(
        pl.col('last_pretest_epoch').is_null() & pl.col('last_train_epoch').is_not_null()
    ).height
    log(f'  Источник last_known: pretest={has_pretest:,}, train={has_train:,}, '
        f'pretrain={lookup.height - has_pretest - has_train:,}')

    # 7. Джойн с test → dormancy per event_id
    test = test.join(lookup.select(['customer_id', 'last_known_epoch']),
                     on='customer_id', how='left')
    test = test.with_columns(
        pl.when(pl.col('last_known_epoch').is_not_null())
        .then((pl.col('test_epoch') - pl.col('last_known_epoch')).cast(pl.Float64) / 86400.0)
        .otherwise(0.0)
        .alias('test_honest_dormancy')
    )

    result = test.select(['event_id', 'test_honest_dormancy'])
    dormancy_mean = result['test_honest_dormancy'].mean()
    dormancy_median = result['test_honest_dormancy'].median()
    log(f'  Test dormancy: mean={dormancy_mean:.2f} days, median={dormancy_median:.2f} days')

    return result


if __name__ == '__main__':
    t0 = time.time()
    log('═══════════════════════════════════════════════════════')
    log('V18b: HYBRID — v14-C profiles + honest dormancy')
    log('═══════════════════════════════════════════════════════')

    # Загрузка модулей v9, v14
    spec9 = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
    v9 = importlib.util.module_from_spec(spec9)
    spec9.loader.exec_module(v9)

    spec14 = importlib.util.spec_from_file_location("v14", ROOT / "pipeline_v14.py")
    v14 = importlib.util.module_from_spec(spec14)
    spec14.loader.exec_module(v14)

    # ── Загрузка данных (как v14-C) ──
    log('Загрузка данных...')
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())
    val_ids = set(pl.read_parquet(FEATURES / 'val_event_ids.parquet')['event_id'].to_list())

    train_df = pl.read_parquet(FEATURES / 'train_features_full.parquet')
    train_df = train_df.filter(~pl.col('event_id').is_in(val_ids))
    val_df = pl.read_parquet(FEATURES / 'val_proper.parquet')
    val_df = val_df.with_columns(
        pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud')
    )
    log(f'  Train: {len(train_df):,}, Val: {len(val_df):,}')

    # ── Стандартный v14-C pipeline: full profiles ──
    log('v14-C features (full profiles)...')
    profiles = pl.read_parquet(FEATURES / 'customer_profiles.parquet')
    deep = pl.read_parquet(FEATURES / 'deep_customer_profiles.parquet')
    mcc_prof = pl.read_parquet(FEATURES / 'customer_mcc_profiles.parquet')
    if mcc_prof['mcc_code'].dtype != pl.Int32:
        mcc_prof = mcc_prof.with_columns(pl.col('mcc_code').cast(pl.Int32))

    train_df = v9.add_features(train_df)
    train_df = v9.add_customer_profiles(train_df, profiles)
    train_df = v14.add_anomaly_features(train_df, deep, mcc_prof)

    val_df = v9.add_features(val_df)
    val_df = v9.add_customer_profiles(val_df, profiles)
    val_df = v14.add_anomaly_features(val_df, deep, mcc_prof)

    # ── Подмена dormancy_days на honest версию (train/val) ──
    log('Подмена dormancy на honest PIT...')
    honest = pl.read_parquet(FEATURES / 'honest_pit_profiles.parquet')
    honest_dorm = honest.select(['event_id', 'honest_dormancy_days'])
    del honest; gc.collect()

    # Train
    train_df = train_df.rename({'dormancy_days': 'dormancy_days_old'})
    train_df = train_df.join(honest_dorm, on='event_id', how='left')
    train_df = train_df.with_columns(
        pl.col('honest_dormancy_days').fill_null(pl.col('dormancy_days_old'))
          .alias('dormancy_days')
    )

    # Val
    val_df = val_df.rename({'dormancy_days': 'dormancy_days_old'})
    val_df = val_df.join(honest_dorm, on='event_id', how='left')
    val_df = val_df.with_columns(
        pl.col('honest_dormancy_days').fill_null(pl.col('dormancy_days_old'))
          .alias('dormancy_days')
    )
    del honest_dorm; gc.collect()

    old_mean = train_df['dormancy_days_old'].head(100_000).mean()
    new_mean = train_df['dormancy_days'].head(100_000).mean()
    log(f'  Train dormancy: old={old_mean:.1f} → honest={new_mean:.1f} days')

    # ── Feature list ──
    base_feats = [c for c in v9.FEATURE_COLS if c in train_df.columns and c in val_df.columns]
    anom_feats = [c for c in v14.ANOMALY_FEATURES if c in train_df.columns and c in val_df.columns]
    all_feats = base_feats + anom_feats
    log(f'Фичи: {len(all_feats)} (base={len(base_feats)}, anomaly={len(anom_feats)})')

    # ── LGBM: 5 seeds ──
    log('\nОбучение LGBM (5 seeds)...')
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
        m.booster_.save_model(str(MODELS_OUT / f'hybrid_s{seed}.txt'))
        del m; gc.collect()

    preds_val /= len(seeds)
    ensemble_prauc = average_precision_score(y_val, preds_val)

    log(f'\n{"="*60}')
    log(f'V18b HYBRID ENSEMBLE: val PR-AUC = {ensemble_prauc:.6f}')
    log(f'v14-C baseline:                     0.044314')
    log(f'Delta:                              +{(ensemble_prauc/0.044314-1)*100:.1f}%')
    log(f'{"="*60}')

    # ── Test submission с honest dormancy ──
    log('\nГенерация test submission...')

    # Строим test dormancy lookup (pretrain+train+pretest)
    test_dorm_lookup = build_test_dormancy_lookup()

    # Test features (как v14-C)
    test_df = pl.read_parquet(FEATURES / 'test_features.parquet')
    test_df = test_df.unique(subset=['event_id'], keep='first')
    test_df = v9.add_features(test_df)
    test_profiles = pl.read_parquet(FEATURES / 'customer_profiles.parquet')
    test_df = v9.add_customer_profiles(test_df, test_profiles)
    test_deep = pl.read_parquet(FEATURES / 'deep_customer_profiles.parquet')
    test_mcc = pl.read_parquet(FEATURES / 'customer_mcc_profiles.parquet')
    if test_mcc['mcc_code'].dtype != pl.Int32:
        test_mcc = test_mcc.with_columns(pl.col('mcc_code').cast(pl.Int32))
    test_df = v14.add_anomaly_features(test_df, test_deep, test_mcc)
    del test_deep, test_mcc, test_profiles; gc.collect()

    # Подмена test dormancy
    test_df = test_df.rename({'dormancy_days': 'dormancy_days_old'})
    test_df = test_df.join(test_dorm_lookup, on='event_id', how='left')
    test_df = test_df.with_columns(
        pl.col('test_honest_dormancy').fill_null(pl.col('dormancy_days_old'))
          .alias('dormancy_days')
    )

    old_test_mean = test_df['dormancy_days_old'].mean()
    new_test_mean = test_df['dormancy_days'].mean()
    log(f'  Test dormancy: old={old_test_mean:.1f} → honest={new_test_mean:.1f} days')

    X_test = test_df.select(all_feats).to_pandas().values.astype(np.float32)
    event_ids_test = test_df['event_id'].to_numpy()

    preds_test = np.mean([
        lgb.Booster(model_file=str(MODELS_OUT / f'hybrid_s{s}.txt')).predict(X_test)
        for s in seeds], axis=0)

    sample = pl.read_csv(DATA / 'sample_submit.csv')
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    sub = pl.DataFrame({'event_id': event_ids_test, 'predict': preds_test.astype(np.float64)})
    sub = sample.select('event_id').join(sub, on='event_id', how='left')
    if sub['predict'].is_null().sum() > 0:
        sub = sub.with_columns(pl.col('predict').fill_null(sub['predict'].drop_nulls().median()))
    path = SUBMIT_OUT / f'submit_v18b_hybrid_{ts}.csv'
    sub.write_csv(path)
    log(f'Сабмит: {path.name}')

    log(f'\nВремя: {(time.time()-t0)/60:.1f} мин')
    log('ГОТОВО')
