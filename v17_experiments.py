"""
v17: Three approaches to beat v14-C (LB=0.1010):
  A) Blend v14-C + IPW_clip20 predictions (no training)
  B) Red vs Yellow score as feature (fraud vs confirmed mini-model)
  C) Focal Loss (custom objective for hard examples)
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from datetime import datetime
import gc, json, time, importlib.util

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'
MODELS_V14  = ROOT / 'models_v14'
MODELS_V16  = ROOT / 'models_v16'
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

LGBM_PARAMS = dict(
    objective='binary', metric='average_precision',
    device='gpu', gpu_platform_id=0, gpu_device_id=0,
    learning_rate=0.02,
    num_leaves=127, min_child_samples=200,
    subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=3.0, max_bin=255,
    n_jobs=4, verbose=-1,
)


# ═══════════════════════════════════════════════════════════
# EXP A: Blend v14-C + IPW_clip20
# ═══════════════════════════════════════════════════════════

def exp_blend(X_va, y_val, X_te, event_ids, all_feats):
    """Blend v14-C and IPW_clip20 predictions."""
    log('\n>>> EXP A: BLEND v14-C + IPW_clip20 <<<')

    # Load v14-C predictions on val and test
    v14c_val = np.mean([
        lgb.Booster(model_file=str(MODELS_V14 / f'C_base_plus_anomaly_s{s}.txt')).predict(X_va)
        for s in seeds], axis=0)
    v14c_test = np.mean([
        lgb.Booster(model_file=str(MODELS_V14 / f'C_base_plus_anomaly_s{s}.txt')).predict(X_te)
        for s in seeds], axis=0)

    # Load IPW_clip20 predictions
    ipw_val = np.mean([
        lgb.Booster(model_file=str(MODELS_V16 / f'IPW_clip20_s{s}.txt')).predict(X_va)
        for s in seeds], axis=0)
    ipw_test = np.mean([
        lgb.Booster(model_file=str(MODELS_V16 / f'IPW_clip20_s{s}.txt')).predict(X_te)
        for s in seeds], axis=0)

    prauc_v14c = average_precision_score(y_val, v14c_val)
    prauc_ipw = average_precision_score(y_val, ipw_val)
    log(f'  v14-C val: {prauc_v14c:.6f}')
    log(f'  IPW_clip20 val: {prauc_ipw:.6f}')

    # Test different blend weights
    best_w, best_prauc = 0, 0
    for w in np.arange(0.0, 1.05, 0.05):
        blend = w * v14c_val + (1 - w) * ipw_val
        prauc = average_precision_score(y_val, blend)
        if prauc > best_prauc:
            best_w, best_prauc = w, prauc

    log(f'  Best blend: w_v14c={best_w:.2f}, val={best_prauc:.6f}')

    # Also test geometric mean (rank-based)
    from scipy.stats import rankdata
    r1 = rankdata(v14c_val) / len(v14c_val)
    r2 = rankdata(ipw_val) / len(ipw_val)
    geo_blend = np.sqrt(r1 * r2)
    prauc_geo = average_precision_score(y_val, geo_blend)
    log(f'  Geometric rank blend: val={prauc_geo:.6f}')

    # Generate submissions for best arithmetic + geometric
    results = {}
    sample = pl.read_csv(DATA / 'sample_submit.csv')
    ts = datetime.now().strftime('%Y%m%d_%H%M')

    # Arithmetic blend
    blend_test = best_w * v14c_test + (1 - best_w) * ipw_test
    sub = pl.DataFrame({'event_id': event_ids, 'predict': blend_test.astype(np.float64)})
    sub = sample.select('event_id').join(sub, on='event_id', how='left')
    if sub['predict'].is_null().sum() > 0:
        sub = sub.with_columns(pl.col('predict').fill_null(sub['predict'].drop_nulls().median()))
    path = SUBMIT_OUT / f'submit_v17_blend_arith_{ts}.csv'
    sub.write_csv(path)
    log(f'  Saved: {path.name}')
    results['blend_arith'] = best_prauc

    # Simple 50/50
    blend50 = 0.5 * v14c_val + 0.5 * ipw_val
    prauc_50 = average_precision_score(y_val, blend50)
    log(f'  50/50 blend: val={prauc_50:.6f}')
    blend50_test = 0.5 * v14c_test + 0.5 * ipw_test
    sub50 = pl.DataFrame({'event_id': event_ids, 'predict': blend50_test.astype(np.float64)})
    sub50 = sample.select('event_id').join(sub50, on='event_id', how='left')
    if sub50['predict'].is_null().sum() > 0:
        sub50 = sub50.with_columns(pl.col('predict').fill_null(sub50['predict'].drop_nulls().median()))
    path50 = SUBMIT_OUT / f'submit_v17_blend_50_50_{ts}.csv'
    sub50.write_csv(path50)
    log(f'  Saved: {path50.name}')
    results['blend_50_50'] = prauc_50

    return results


# ═══════════════════════════════════════════════════════════
# EXP B: Red vs Yellow score as feature
# ═══════════════════════════════════════════════════════════

def train_red_vs_yellow(train_df, val_df, test_df, all_feats, labels_df):
    """Train fraud vs confirmed model, use OOF predictions as feature."""
    log('\n>>> EXP B: RED VS YELLOW SCORE <<<')

    # Get ONLY labeled data (87K: fraud + confirmed), NOT green
    label_event_ids = set(labels_df['event_id'].to_list())
    labeled_df = train_df.filter(pl.col('event_id').is_in(label_event_ids))

    n_fraud = (labeled_df['target'] == 1).sum()
    n_confirmed = (labeled_df['target'] == 0).sum()
    log(f'  Labeled: {len(labeled_df):,} (fraud={n_fraud}, confirmed={n_confirmed})')

    # Get features for labeled data
    avail = [f for f in all_feats if f in labeled_df.columns]
    X_labeled = labeled_df.select(avail).to_pandas().values.astype(np.float32)
    y_labeled = labeled_df['target'].to_numpy().astype(int)

    # 5-fold CV to get OOF predictions for ALL train data
    log('  Training 5-fold CV on labeled data...')
    rvs_params = dict(
        objective='binary', metric='auc',
        device='gpu', gpu_platform_id=0, gpu_device_id=0,
        learning_rate=0.05, num_leaves=63, min_child_samples=100,
        subsample=0.7, colsample_bytree=0.5,
        reg_alpha=1.0, reg_lambda=5.0,
        n_estimators=2000, n_jobs=4, verbose=-1,
    )

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_labeled))
    models = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_labeled, y_labeled)):
        m = lgb.LGBMClassifier(**rvs_params, random_state=fold)
        m.fit(X_labeled[tr_idx], y_labeled[tr_idx],
              eval_set=[(X_labeled[va_idx], y_labeled[va_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof_preds[va_idx] = m.predict_proba(X_labeled[va_idx])[:, 1]
        models.append(m)
        log(f'    Fold {fold}: iter={m.best_iteration_}, AUC on labeled val')

    from sklearn.metrics import roc_auc_score
    oof_auc = roc_auc_score(y_labeled, oof_preds)
    log(f'  OOF AUC (fraud vs confirmed): {oof_auc:.4f}')

    # Now predict red_vs_yellow score for ALL data (train, val, test)
    X_tr_full = train_df.select(avail).to_pandas().values.astype(np.float32)
    X_va_full = val_df.select(avail).to_pandas().values.astype(np.float32)
    X_te_full = test_df.select(avail).to_pandas().values.astype(np.float32)

    rvs_train = np.mean([m.predict_proba(X_tr_full)[:, 1] for m in models], axis=0)
    rvs_val = np.mean([m.predict_proba(X_va_full)[:, 1] for m in models], axis=0)
    rvs_test = np.mean([m.predict_proba(X_te_full)[:, 1] for m in models], axis=0)

    log(f'  RvY score stats (train): mean={rvs_train.mean():.4f}, std={rvs_train.std():.4f}')
    log(f'  RvY score stats (val): mean={rvs_val.mean():.4f}, std={rvs_val.std():.4f}')
    log(f'  RvY score stats (test): mean={rvs_test.mean():.4f}, std={rvs_test.std():.4f}')

    del X_tr_full, X_va_full, X_te_full, models; gc.collect()
    return rvs_train, rvs_val, rvs_test


# ═══════════════════════════════════════════════════════════
# EXP C: Focal Loss
# ═══════════════════════════════════════════════════════════

def focal_loss_objective(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal loss: -alpha * (1-p)^gamma * log(p) for positives."""
    p = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid
    grad = p - y_true  # same as binary CE gradient

    # Focal weighting
    p_t = np.where(y_true == 1, p, 1 - p)
    focal_weight = alpha * (1 - p_t) ** gamma

    grad = focal_weight * grad
    hess = focal_weight * p * (1 - p)
    hess = np.maximum(hess, 1e-7)  # stability
    return grad, hess


def focal_loss_obj(y_pred, dataset):
    y_true = dataset.get_label()
    return focal_loss_objective(y_true, y_pred, gamma=2.0, alpha=0.25)


def focal_loss_eval(y_pred, dataset):
    y_true = dataset.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    prauc = average_precision_score(y_true, p)
    return 'pr_auc', prauc, True


def train_model(name, X_train, y_train, X_val, y_val, n_trees=10000, patience=300,
                custom_obj=None, custom_eval=None, use_cpu=False):
    """Train 5-seed LGBM ensemble, optionally with custom objective."""
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    spw = min(spw, 50.0)

    log(f'\n{"="*60}')
    log(f'Training: {name} ({X_train.shape[1]} feats{"  CPU" if use_cpu else ""})')
    log(f'{"="*60}')

    params = dict(LGBM_PARAMS)
    if use_cpu:
        params.pop('device', None)
        params.pop('gpu_platform_id', None)
        params.pop('gpu_device_id', None)
        params['n_jobs'] = -1
    if custom_obj is not None:
        # Remove objective from params, use custom
        params.pop('objective', None)
        params.pop('metric', None)

    preds = np.zeros(len(X_val), dtype=np.float64)
    for seed in seeds:
        if custom_obj is not None:
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            bst_params = {k: v for k, v in params.items()
                         if k not in ('n_estimators',)}
            bst_params['random_state'] = seed
            bst_params['scale_pos_weight'] = spw
            bst = lgb.train(
                bst_params, dtrain, num_boost_round=n_trees,
                valid_sets=[dval], fobj=custom_obj, feval=custom_eval,
                callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)]
            )
            p = 1.0 / (1.0 + np.exp(-bst.predict(X_val)))  # sigmoid for custom obj
            prauc = average_precision_score(y_val, p)
            log(f'  Seed {seed}: iter={bst.best_iteration}, val={prauc:.6f}')
            bst.save_model(str(MODELS_OUT / f'{name}_s{seed}.txt'))
            preds += p
            del bst; gc.collect()
        else:
            m = lgb.LGBMClassifier(**params, random_state=seed,
                scale_pos_weight=spw, n_estimators=n_trees)
            m.fit(X_train, y_train,
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
    log('=== V17: BLEND + RED_VS_YELLOW + FOCAL LOSS ===')

    # Load data (same as v14-C)
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
    event_ids = test_df['event_id'].to_numpy()

    log(f'Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}')
    log(f'Features: {len(all_feats)}')

    X_tr = train_df.select(all_feats).to_pandas().values.astype(np.float32)
    X_va = val_df.select(all_feats).to_pandas().values.astype(np.float32)
    X_te = test_df.select(all_feats).to_pandas().values.astype(np.float32)

    results = {
        'A_blend_50_50': 0.048482,  # Already computed
        'A_blend_arith': 0.051048,  # w_v14c=0, just IPW_clip20
    }

    # ── EXP B: Red vs Yellow score as feature ──
    rvs_train, rvs_val, rvs_test = train_red_vs_yellow(train_df, val_df, test_df, all_feats, labels)

    # Add RvY as extra feature (use pandas DataFrame to keep feature names)
    import pandas as pd
    feats_rvs = all_feats + ['rvs_score']
    X_tr_rvs = pd.DataFrame(X_tr, columns=all_feats)
    X_tr_rvs['rvs_score'] = rvs_train.astype(np.float32)
    X_va_rvs = pd.DataFrame(X_va, columns=all_feats)
    X_va_rvs['rvs_score'] = rvs_val.astype(np.float32)
    X_te_rvs = pd.DataFrame(X_te, columns=all_feats)
    X_te_rvs['rvs_score'] = rvs_test.astype(np.float32)

    results['B_red_vs_yellow'] = train_model(
        'B_red_vs_yellow', X_tr_rvs, y_train, X_va_rvs, y_val, use_cpu=True)

    del X_tr_rvs, X_va_rvs; gc.collect()

    # ── EXP C: Focal Loss ──
    results['C_focal_loss'] = train_model(
        'C_focal_loss', X_tr, y_train, X_va, y_val,
        custom_obj=focal_loss_obj, custom_eval=focal_loss_eval)

    # ── Generate submissions ──
    log('\n=== SUBMISSIONS ===')
    sample = pl.read_csv(DATA / 'sample_submit.csv')
    ts = datetime.now().strftime('%Y%m%d_%H%M')

    # B: Red vs Yellow
    X_te_rvs_df = pd.DataFrame(X_te, columns=all_feats)
    X_te_rvs_df['rvs_score'] = rvs_test.astype(np.float32)
    preds_b = np.mean([
        lgb.Booster(model_file=str(MODELS_OUT / f'B_red_vs_yellow_s{s}.txt')).predict(
            X_te_rvs_df)
        for s in seeds], axis=0)
    sub_b = pl.DataFrame({'event_id': event_ids, 'predict': preds_b.astype(np.float64)})
    sub_b = sample.select('event_id').join(sub_b, on='event_id', how='left')
    if sub_b['predict'].is_null().sum() > 0:
        sub_b = sub_b.with_columns(pl.col('predict').fill_null(sub_b['predict'].drop_nulls().median()))
    path_b = SUBMIT_OUT / f'submit_v17_red_vs_yellow_{ts}.csv'
    sub_b.write_csv(path_b)
    log(f'Saved: {path_b.name}')

    # C: Focal Loss (need sigmoid for custom obj predictions)
    preds_c_raw = np.mean([
        lgb.Booster(model_file=str(MODELS_OUT / f'C_focal_loss_s{s}.txt')).predict(X_te)
        for s in seeds], axis=0)
    preds_c = 1.0 / (1.0 + np.exp(-preds_c_raw))
    sub_c = pl.DataFrame({'event_id': event_ids, 'predict': preds_c.astype(np.float64)})
    sub_c = sample.select('event_id').join(sub_c, on='event_id', how='left')
    if sub_c['predict'].is_null().sum() > 0:
        sub_c = sub_c.with_columns(pl.col('predict').fill_null(sub_c['predict'].drop_nulls().median()))
    path_c = SUBMIT_OUT / f'submit_v17_focal_loss_{ts}.csv'
    sub_c.write_csv(path_c)
    log(f'Saved: {path_c.name}')

    # Summary
    log(f'\n{"="*60}')
    log('RESULTS v17')
    log(f'{"="*60}')
    log(f'v14-C reference: val=0.044, LB=0.1010')
    for name, prauc in sorted(results.items(), key=lambda x: -x[1]):
        delta = 100 * (prauc / 0.044 - 1)
        log(f'  {name}: val={prauc:.6f} ({delta:+.1f}% vs v14-C)')

    with open(MODELS_OUT / 'results_v17.json', 'w') as f:
        json.dump(results, f, indent=2)

    log(f'\nTime: {(time.time()-t_start)/60:.0f} min')
    log('DONE')
