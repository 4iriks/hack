"""
v16-IPW: Adversarial Importance-Weighted training.

Идея: обучить adversarial модель (train vs test), получить P(test|x),
затем взвесить train samples: w = P(test|x) / P(train|x) = p/(1-p).
Clipped max=10-20 чтобы не давать единичным примерам огромный вес.

Это заставляет LGBM фокусироваться на train примерах, похожих на test,
тем самым уменьшая эффект distribution shift.

121 фич (как v14-C), те же параметры, но sample_weight вместо uniform.
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
MODELS_OUT  = ROOT / 'models_v16'
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

LGBM_PARAMS = dict(
    objective='binary', metric='average_precision',
    device='gpu', gpu_platform_id=0, gpu_device_id=0,
    learning_rate=0.02,
    num_leaves=127, min_child_samples=200,
    subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
    n_jobs=4, verbose=-1,
)

CLIP_MAX = 10.0  # Max weight for IPW


def train_adversarial(X_train, X_test, feats):
    """Train adversarial classifier with 5-fold CV to get OOF probabilities.
    Returns P(test|x) for each train sample (out-of-fold).
    """
    from sklearn.model_selection import StratifiedKFold
    log('Training adversarial model (5-fold CV for soft probabilities)...')

    # Subsample for speed
    n_sub = min(300_000, len(X_train))
    rng = np.random.RandomState(42)
    tr_idx = rng.choice(len(X_train), n_sub, replace=False)
    X_tr_sub = X_train[tr_idx]

    n_te = min(300_000, len(X_test))
    te_idx = rng.choice(len(X_test), n_te, replace=False)
    X_te_sub = X_test[te_idx]

    X_adv = np.vstack([X_tr_sub, X_te_sub])
    y_adv = np.concatenate([np.zeros(len(X_tr_sub)), np.ones(len(X_te_sub))])

    # Very weak model to avoid perfect separation
    adv_params = dict(
        objective='binary', metric='auc',
        device='gpu', gpu_platform_id=0, gpu_device_id=0,
        learning_rate=0.1, num_leaves=15, max_depth=4,
        min_child_samples=500, subsample=0.5, colsample_bytree=0.3,
        reg_alpha=5.0, reg_lambda=10.0,
        n_estimators=100, n_jobs=4, verbose=-1,
    )

    # 5-fold CV for out-of-fold predictions
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_adv))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_adv, y_adv)):
        m = lgb.LGBMClassifier(**adv_params, random_state=fold)
        m.fit(X_adv[train_idx], y_adv[train_idx])
        oof_preds[val_idx] = m.predict_proba(X_adv[val_idx])[:, 1]
        del m

    # OOF AUC
    from sklearn.metrics import roc_auc_score
    oof_auc = roc_auc_score(y_adv, oof_preds)
    log(f'  Adversarial OOF AUC: {oof_auc:.4f}')

    # Get OOF probabilities for the train subset
    p_train_sub = oof_preds[:len(X_tr_sub)]
    log(f'  P(test|train) stats: mean={p_train_sub.mean():.4f}, std={p_train_sub.std():.4f}, '
        f'min={p_train_sub.min():.4f}, max={p_train_sub.max():.4f}')

    # Now train final model on full adversarial data to predict ALL train samples
    m_final = lgb.LGBMClassifier(**adv_params, random_state=42)
    m_final.fit(X_adv, y_adv)

    # Predict P(test|x) for ALL train samples
    p_test = m_final.predict_proba(X_train)[:, 1]

    # Top features driving shift
    imp = pd.Series(m_final.feature_importances_, index=feats).sort_values(ascending=False)
    log(f'  Top shift features: {dict(imp.head(5))}')
    log(f'  Full P(test|train) stats: mean={p_test.mean():.4f}, std={p_test.std():.4f}, '
        f'min={p_test.min():.4f}, max={p_test.max():.4f}')

    del m_final, X_adv, y_adv; gc.collect()
    return p_test


def compute_ipw_weights(p_test, clip_max=CLIP_MAX):
    """Compute importance weights: w = p/(1-p), clipped."""
    p_clipped = np.clip(p_test, 0.01, 0.99)
    weights = p_clipped / (1 - p_clipped)
    weights = np.clip(weights, 0.01, clip_max)
    # Normalize to mean=1
    weights = weights / weights.mean()
    return weights


def train_model_weighted(name, X_train, y_train, X_val, y_val, sample_weight,
                         n_trees=10000, patience=300):
    """Train 5-seed LGBM ensemble with sample weights."""
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    spw = min(spw, 50.0)

    log(f'\n{"="*60}')
    log(f'Training: {name} ({X_train.shape[1]} feats, weighted)')
    log(f'  Weight stats: mean={sample_weight.mean():.2f}, std={sample_weight.std():.2f}, '
        f'min={sample_weight.min():.3f}, max={sample_weight.max():.3f}')
    log(f'{"="*60}')

    preds = np.zeros(len(X_val), dtype=np.float64)
    for seed in seeds:
        m = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed,
            scale_pos_weight=spw, n_estimators=n_trees)
        m.fit(X_train, y_train, sample_weight=sample_weight,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)])
        p = m.predict_proba(X_val)[:, 1]
        preds += p
        prauc = average_precision_score(y_val, p)
        log(f'  Seed {seed}: iter={m.best_iteration_}, val={prauc:.6f}')
        m.booster_.save_model(str(MODELS_OUT / f'{name}_s{seed}.txt'))
        del m; gc.collect()

    preds /= len(seeds)
    ensemble_prauc = average_precision_score(y_val, preds)
    log(f'  Ensemble ({name}): val={ensemble_prauc:.6f}')
    return ensemble_prauc


if __name__ == '__main__':
    t_start = time.time()
    log('=== V16-IPW: ADVERSARIAL IMPORTANCE WEIGHTING ===')

    # Load
    old_deep_profs = pl.read_parquet(FEATURES_IN / 'deep_customer_profiles.parquet')
    old_mcc_profs = pl.read_parquet(FEATURES_IN / 'customer_mcc_profiles.parquet')
    if old_mcc_profs['mcc_code'].dtype != pl.Int32:
        old_mcc_profs = old_mcc_profs.with_columns(pl.col('mcc_code').cast(pl.Int32))
    old_profiles = pl.read_parquet(FEATURES_IN / 'customer_profiles.parquet')
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    fraud_ids = set(labels.filter(pl.col('target') == 1)['event_id'].to_list())
    val_ids = set(pl.read_parquet(FEATURES_IN / 'val_event_ids.parquet')['event_id'].to_list())

    log('Loading data...')
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
    log(f'Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}')
    log(f'Features: {len(all_feats)}')

    X_tr = train_df.select(all_feats).to_pandas().values.astype(np.float32)
    X_va = val_df.select(all_feats).to_pandas().values.astype(np.float32)
    X_te = test_df.select(all_feats).to_pandas().values.astype(np.float32)
    event_ids = test_df['event_id'].to_numpy()

    del train_df, val_df; gc.collect()

    # Train adversarial model
    p_test = train_adversarial(X_tr, X_te, all_feats)

    # Try different clip values
    results = {}
    for clip in [5.0, 10.0, 20.0]:
        name = f'IPW_clip{int(clip)}'
        weights = compute_ipw_weights(p_test, clip_max=clip)
        prauc = train_model_weighted(name, X_tr, y_train, X_va, y_val, weights)
        results[name] = prauc

    # Find best
    best_name = max(results, key=results.get)
    best_prauc = results[best_name]
    log(f'\nBest: {best_name} (val={best_prauc:.6f})')
    log(f'v14-C ref: 0.044')
    for n, p in sorted(results.items(), key=lambda x: -x[1]):
        log(f'  {n}: val={p:.6f} ({100*(p/0.044-1):+.1f}%)')

    # Generate submission for best
    log('\n=== SUBMIT ===')
    preds = np.mean([
        lgb.Booster(model_file=str(MODELS_OUT / f'{best_name}_s{s}.txt')).predict(X_te)
        for s in seeds], axis=0)

    sample = pl.read_csv(DATA / 'sample_submit.csv')
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    sub = pl.DataFrame({'event_id': event_ids, 'predict': preds.astype(np.float64)})
    sub = sample.select('event_id').join(sub, on='event_id', how='left')
    n_null = sub['predict'].is_null().sum()
    if n_null > 0:
        sub = sub.with_columns(pl.col('predict').fill_null(sub['predict'].drop_nulls().median()))

    path = SUBMIT_OUT / f'submit_v16_{best_name}_{ts}.csv'
    sub.write_csv(path)

    # Also save best clip=10 separately (our default)
    if best_name != 'IPW_clip10':
        preds10 = np.mean([
            lgb.Booster(model_file=str(MODELS_OUT / f'IPW_clip10_s{s}.txt')).predict(X_te)
            for s in seeds], axis=0)
        sub10 = pl.DataFrame({'event_id': event_ids, 'predict': preds10.astype(np.float64)})
        sub10 = sample.select('event_id').join(sub10, on='event_id', how='left')
        if sub10['predict'].is_null().sum() > 0:
            sub10 = sub10.with_columns(pl.col('predict').fill_null(sub10['predict'].drop_nulls().median()))
        path10 = SUBMIT_OUT / f'submit_v16_IPW_clip10_{ts}.csv'
        sub10.write_csv(path10)
        log(f'Saved: {path10.name}')

    log(f'Saved: {path.name}')
    log(f'Time: {(time.time()-t_start)/60:.0f} min')

    with open(MODELS_OUT / 'results_v16_ipw.json', 'w') as f:
        json.dump(results, f, indent=2)

    log('DONE')
