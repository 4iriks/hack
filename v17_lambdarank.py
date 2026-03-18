"""
v17-LR: LambdaRank — оптимизация ранжирования напрямую.

PR-AUC = метрика ранжирования, а мы оптимизируем binary CE.
LambdaRank напрямую оптимизирует NDCG/MAP через попарные сравнения.
Группируем по customer_id — ранжируем транзакции ВНУТРИ клиента.
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from pathlib import Path
from datetime import datetime
import gc, json, time, importlib.util

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_OUT  = ROOT / 'models_v17'
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

spec9 = importlib.util.spec_from_file_location("v9", ROOT / "pipeline_v9.py")
v9 = importlib.util.module_from_spec(spec9)
spec9.loader.exec_module(v9)

spec14 = importlib.util.spec_from_file_location("v14", ROOT / "pipeline_v14.py")
v14 = importlib.util.module_from_spec(spec14)
spec14.loader.exec_module(v14)

seeds = [42, 123, 777, 2024, 31337]


if __name__ == '__main__':
    t_start = time.time()
    log('=== V17-LR: LAMBDARANK ===')

    # Загрузка данных (как v14-C)
    old_deep_profs = pl.read_parquet(FEATURES_IN / 'deep_customer_profiles.parquet')
    old_mcc_profs = pl.read_parquet(FEATURES_IN / 'customer_mcc_profiles.parquet')
    if old_mcc_profs['mcc_code'].dtype != pl.Int32:
        old_mcc_profs = old_mcc_profs.with_columns(pl.col('mcc_code').cast(pl.Int32))
    old_profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())
    val_ids = set(pl.read_parquet(FEATURES_IN / 'val_event_ids.parquet')['event_id'].to_list())

    log('Загрузка данных...')
    train_df = pl.read_parquet(FEATURES_IN / 'train_features_full.parquet')
    train_df = train_df.filter(~pl.col('event_id').is_in(val_ids))
    train_df = v9.add_features(train_df)
    train_df = v9.add_customer_profiles(train_df, old_profiles)
    train_df = v14.add_anomaly_features(train_df, old_deep_profs, old_mcc_profs)

    val_df = pl.read_parquet(FEATURES_IN / 'val_proper.parquet')
    val_df = val_df.with_columns(pl.col('event_id').is_in(fraud_ids).cast(pl.Int8).alias('is_fraud'))
    val_df = v9.add_features(val_df)
    val_df = v9.add_customer_profiles(val_df, old_profiles)
    val_df = v14.add_anomaly_features(val_df, old_deep_profs, old_mcc_profs)

    test_df = pl.read_parquet(FEATURES_IN / 'test_features.parquet')
    test_df = test_df.unique(subset=['event_id'], keep='first')
    test_df = v9.add_features(test_df)
    test_df = v9.add_customer_profiles(test_df, old_profiles)
    test_df = v14.add_anomaly_features(test_df, old_deep_profs, old_mcc_profs)

    base_feats = [c for c in v9.FEATURE_COLS if c in train_df.columns and c in val_df.columns and c in test_df.columns]
    anom_feats = [c for c in v14.ANOMALY_FEATURES if c in train_df.columns and c in val_df.columns and c in test_df.columns]
    all_feats = base_feats + anom_feats

    y_train = train_df['target'].to_numpy().astype(int)
    y_val = val_df['is_fraud'].to_numpy().astype(int)
    event_ids_test = test_df['event_id'].to_numpy()

    log(f'Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}')
    log(f'Фичи: {len(all_feats)}')

    # Группы по customer_id для LambdaRank
    train_customers = train_df['customer_id'].to_numpy()
    val_customers = val_df['customer_id'].to_numpy()

    # Подсчёт размеров групп (LambdaRank требует массив group_sizes)
    log('Подсчёт групп клиентов...')

    def compute_groups(customers):
        """Возвращает массив размеров групп (сколько строк у каждого клиента подряд)."""
        # Нужно отсортировать по customer_id и вернуть group sizes
        order = np.argsort(customers, kind='stable')
        sorted_custs = customers[order]
        # Находим границы групп
        changes = np.where(sorted_custs[1:] != sorted_custs[:-1])[0] + 1
        boundaries = np.concatenate([[0], changes, [len(sorted_custs)]])
        groups = np.diff(boundaries)
        return order, groups

    tr_order, tr_groups = compute_groups(train_customers)
    va_order, va_groups = compute_groups(val_customers)

    log(f'  Train: {len(tr_groups):,} групп (клиентов), мин={tr_groups.min()}, макс={tr_groups.max()}, медиана={int(np.median(tr_groups))}')
    log(f'  Val: {len(va_groups):,} групп, мин={va_groups.min()}, макс={va_groups.max()}, медиана={int(np.median(va_groups))}')

    X_tr = train_df.select(all_feats).to_pandas().values.astype(np.float32)
    X_va = val_df.select(all_feats).to_pandas().values.astype(np.float32)
    X_te = test_df.select(all_feats).to_pandas().values.astype(np.float32)

    # Сортируем по customer_id
    X_tr_sorted = X_tr[tr_order]
    y_tr_sorted = y_train[tr_order]
    X_va_sorted = X_va[va_order]
    y_va_sorted = y_val[va_order]

    del train_df, val_df; gc.collect()

    # ── Параметры LambdaRank ──
    lr_params = dict(
        objective='lambdarank',
        metric='ndcg',
        ndcg_eval_at=[5, 10, 50],
        device='gpu', gpu_platform_id=0, gpu_device_id=0,
        learning_rate=0.02,
        num_leaves=127, min_child_samples=200,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
        n_jobs=4, verbose=-1,
        label_gain=[0, 1],  # бинарная relevance: 0=не фрод, 1=фрод
    )

    # ── Также тестируем MAP objective ──
    results = {}

    for obj_name, obj_type in [('lambdarank', 'lambdarank'), ('rank_xendcg', 'rank_xendcg')]:
        log(f'\n{"="*60}')
        log(f'Эксперимент: {obj_name} (121 фич)')
        log(f'{"="*60}')

        params = dict(lr_params)
        params['objective'] = obj_type

        preds_orig_order = np.zeros(len(X_va), dtype=np.float64)

        for seed in seeds:
            params['random_state'] = seed

            dtrain = lgb.Dataset(X_tr_sorted, label=y_tr_sorted, group=tr_groups)
            dval = lgb.Dataset(X_va_sorted, label=y_va_sorted, group=va_groups, reference=dtrain)

            bst = lgb.train(
                params, dtrain, num_boost_round=10000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)]
            )

            # Предсказания на отсортированном val
            p_sorted = bst.predict(X_va_sorted)
            # Обратная сортировка в оригинальный порядок
            p_orig = np.empty_like(p_sorted)
            p_orig[va_order] = p_sorted

            prauc = average_precision_score(y_val, p_orig)
            log(f'  Seed {seed}: iter={bst.best_iteration}, val={prauc:.6f}')

            preds_orig_order += p_orig
            bst.save_model(str(MODELS_OUT / f'LR_{obj_name}_s{seed}.txt'))
            del bst; gc.collect()

        preds_orig_order /= len(seeds)
        ensemble_prauc = average_precision_score(y_val, preds_orig_order)
        log(f'  Ensemble ({obj_name}): val={ensemble_prauc:.6f}')
        results[obj_name] = ensemble_prauc

    # ── Также: binary baseline для сравнения (те же данные, свежее обучение) ──
    log(f'\n{"="*60}')
    log(f'Baseline: binary (для сравнения)')
    log(f'{"="*60}')

    bin_params = dict(
        objective='binary', metric='average_precision',
        device='gpu', gpu_platform_id=0, gpu_device_id=0,
        learning_rate=0.02,
        num_leaves=127, min_child_samples=200,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
        n_jobs=4, verbose=-1,
    )
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    spw = min(spw, 50.0)

    preds_bin = np.zeros(len(X_va), dtype=np.float64)
    for seed in seeds:
        m = lgb.LGBMClassifier(**bin_params, random_state=seed,
            scale_pos_weight=spw, n_estimators=10000)
        m.fit(X_tr, y_train,
              eval_set=[(X_va, y_val)],
              callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)])
        p = m.predict_proba(X_va)[:, 1]
        preds_bin += p
        prauc = average_precision_score(y_val, p)
        log(f'  Seed {seed}: iter={m.best_iteration_}, val={prauc:.6f}')
        del m; gc.collect()

    preds_bin /= len(seeds)
    bin_prauc = average_precision_score(y_val, preds_bin)
    log(f'  Ensemble (binary): val={bin_prauc:.6f}')
    results['binary_baseline'] = bin_prauc

    # ── Итоги ──
    log(f'\n{"="*60}')
    log('РЕЗУЛЬТАТЫ')
    log(f'{"="*60}')
    for name, prauc in sorted(results.items(), key=lambda x: -x[1]):
        log(f'  {name}: val={prauc:.6f}')

    # ── Сабмит лучшего ──
    best_name = max(results, key=results.get)
    if best_name != 'binary_baseline':
        log(f'\n=== САБМИТ: {best_name} ===')
        # Для LambdaRank нужен predict на неотсортированных данных
        preds_test = np.mean([
            lgb.Booster(model_file=str(MODELS_OUT / f'LR_{best_name}_s{s}.txt')).predict(X_te)
            for s in seeds], axis=0)

        sample = pl.read_csv(DATA / 'sample_submit.csv')
        ts = datetime.now().strftime('%Y%m%d_%H%M')
        sub = pl.DataFrame({'event_id': event_ids_test, 'predict': preds_test.astype(np.float64)})
        sub = sample.select('event_id').join(sub, on='event_id', how='left')
        if sub['predict'].is_null().sum() > 0:
            sub = sub.with_columns(pl.col('predict').fill_null(sub['predict'].drop_nulls().median()))
        path = SUBMIT_OUT / f'submit_v17_LR_{best_name}_{ts}.csv'
        sub.write_csv(path)
        log(f'Сохранён: {path.name}')
    else:
        log('\nBinary baseline лучше — LambdaRank не помог.')

    with open(MODELS_OUT / 'results_v17_lr.json', 'w') as f:
        json.dump(results, f, indent=2)

    log(f'\nВремя: {(time.time()-t_start)/60:.0f} мин')
    log('ГОТОВО')
