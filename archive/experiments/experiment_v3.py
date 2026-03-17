"""
Experiment v3: тестируем разные neg ratios + target/frequency encoding.
Всё документируем.
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from pathlib import Path
from datetime import datetime
import gc, time, json

ROOT        = Path('/home/vadim/PyPr/hak')
DATA        = ROOT / 'main_data'
FEATURES_IN = ROOT / 'features'

from pipeline_v2 import log, add_v2_features, FEATURE_COLS_V2


def load_train_with_ratio(neg_ratio: int):
    """Пересобираем train из чанков с заданным neg_ratio."""
    tmp_dir = FEATURES_IN / '_tmp_train'
    chunk_files = sorted(tmp_dir.glob('chunk_*.parquet'))
    labels = pl.read_parquet(DATA / 'train_labels.parquet')
    train_start = datetime(2024, 10, 1)

    n_fraud = labels.filter(pl.col('target') == 1).height
    n_null_target = n_fraud * neg_ratio

    total_null = 0
    for cf in chunk_files:
        n = pl.scan_parquet(cf).filter(
            pl.col('event_dttm') >= train_start
        ).select(pl.len()).collect().item()
        total_null += n
    total_null -= len(labels)
    frac = min(n_null_target / max(total_null, 1), 1.0)

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
    return df_train


def add_target_encoding(df_train: pl.DataFrame, df_val: pl.DataFrame,
                        cols: list, target_col='target', smoothing=10):
    """Target encoding с Leave-One-Out для train, global для val."""
    global_mean = df_train[target_col].mean()

    for col in cols:
        # Stats per category from train
        stats = df_train.group_by(col).agg([
            pl.col(target_col).sum().alias('_te_sum'),
            pl.col(target_col).count().alias('_te_cnt'),
        ])

        # LOO for train: (sum_all - target_i) / (cnt_all - 1)
        df_train = df_train.join(stats, on=col, how='left')
        df_train = df_train.with_columns([
            (
                (pl.col('_te_sum') - pl.col(target_col)) /
                (pl.col('_te_cnt') - 1 + smoothing) +
                global_mean * smoothing / (pl.col('_te_cnt') - 1 + smoothing)
            ).alias(f'te_{col}')
        ]).drop(['_te_sum', '_te_cnt'])

        # For val: direct mapping
        stats = stats.with_columns([
            (pl.col('_te_sum') / (pl.col('_te_cnt') + smoothing) +
             global_mean * smoothing / (pl.col('_te_cnt') + smoothing))
              .alias(f'te_{col}')
        ]).select([col, f'te_{col}'])

        df_val = df_val.join(stats, on=col, how='left')
        df_val = df_val.with_columns(
            pl.col(f'te_{col}').fill_null(global_mean)
        )

    return df_train, df_val


def add_frequency_encoding(df: pl.DataFrame, cols: list, total: int = None):
    """Frequency encoding: count / total."""
    if total is None:
        total = len(df)
    for col in cols:
        counts = df.group_by(col).agg(pl.len().alias(f'freq_{col}'))
        df = df.join(counts, on=col, how='left')
        df = df.with_columns(
            (pl.col(f'freq_{col}') / total).alias(f'freq_{col}')
        )
    return df


def quick_eval(X_train, y_train, X_val, y_val, label=""):
    """Quick LightGBM eval to test feature sets."""
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos = n_neg / n_pos

    model = lgb.LGBMClassifier(
        objective='binary', metric='average_precision',
        device='gpu', gpu_platform_id=0, gpu_device_id=0,
        scale_pos_weight=scale_pos,
        n_estimators=3000, learning_rate=0.03,
        num_leaves=255, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=2.0, max_bin=255,
        random_state=42, n_jobs=4, verbose=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])

    preds = model.predict_proba(X_val)[:, 1]
    score = average_precision_score(y_val, preds)
    log(f'[{label}] PR-AUC={score:.4f}  best_iter={model.best_iteration_}  features={X_train.shape[1]}')
    return score, model


def main():
    results = {}

    # ═══════════════════════════════════════════
    # EXP 1: Test different neg ratios
    # ═══════════════════════════════════════════
    print('\n' + '='*60)
    print('EXP 1: Negative ratio experiments')
    print('='*60)

    val_dt = datetime(2025, 4, 1)

    for ratio in [5, 10, 20, 50]:
        log(f'--- Ratio {ratio}:1 ---')
        df = load_train_with_ratio(ratio)
        df = add_v2_features(df)

        available_feats = [c for c in FEATURE_COLS_V2 if c in df.columns]

        df_tr = df.filter(pl.col('event_dttm') < val_dt)
        df_val = df.filter(pl.col('event_dttm') >= val_dt)

        X_train = df_tr.select(available_feats).to_pandas()
        y_train = df_tr['target'].to_numpy().astype(int)
        X_val = df_val.select(available_feats).to_pandas()
        y_val = df_val['target'].to_numpy().astype(int)

        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        log(f'  Train: {len(df_tr):,} (pos={n_pos:,}, neg={n_neg:,})')
        log(f'  Val:   {len(df_val):,}')

        score, _ = quick_eval(X_train, y_train, X_val, y_val, f'ratio_{ratio}')
        results[f'ratio_{ratio}'] = score

        del df, df_tr, df_val, X_train, y_train, X_val, y_val; gc.collect()

    best_ratio_key = max(results, key=results.get)
    best_ratio = int(best_ratio_key.split('_')[1])
    log(f'\n>>> Best ratio: {best_ratio}:1 (PR-AUC={results[best_ratio_key]:.4f})')

    # ═══════════════════════════════════════════
    # EXP 2: Target encoding + frequency encoding
    # ═══════════════════════════════════════════
    print('\n' + '='*60)
    print('EXP 2: Target + Frequency encoding')
    print('='*60)

    df = load_train_with_ratio(best_ratio)
    df = add_v2_features(df)

    available_feats = [c for c in FEATURE_COLS_V2 if c in df.columns]

    df_tr = df.filter(pl.col('event_dttm') < val_dt)
    df_val = df.filter(pl.col('event_dttm') >= val_dt)

    # Add target encoding
    te_cols = ['mcc_code', 'event_type_nm', 'event_desc', 'channel_indicator_type',
               'channel_indicator_sub_type', 'currency_iso_cd', 'operating_system_type', 'pos_cd']
    te_cols = [c for c in te_cols if c in df.columns]
    log(f'Target encoding cols: {te_cols}')

    df_tr, df_val = add_target_encoding(df_tr, df_val, te_cols)
    te_feat_names = [f'te_{c}' for c in te_cols]

    # Add frequency encoding
    freq_cols = ['mcc_code', 'event_type_nm', 'event_desc', 'channel_indicator_type']
    freq_cols = [c for c in freq_cols if c in df.columns]
    df_tr = add_frequency_encoding(df_tr, freq_cols)
    df_val = add_frequency_encoding(df_val, freq_cols, total=len(df_tr))
    freq_feat_names = [f'freq_{c}' for c in freq_cols]

    extended_feats = available_feats + te_feat_names + freq_feat_names
    # Make sure all are in both
    extended_feats = [c for c in extended_feats if c in df_tr.columns and c in df_val.columns]

    X_train = df_tr.select(extended_feats).to_pandas()
    y_train = df_tr['target'].to_numpy().astype(int)
    X_val = df_val.select(extended_feats).to_pandas()
    y_val = df_val['target'].to_numpy().astype(int)

    log(f'Extended features: {len(extended_feats)}')
    score_te, _ = quick_eval(X_train, y_train, X_val, y_val, 'target+freq_enc')
    results['target+freq_enc'] = score_te

    # Also test without target encoding (just freq)
    feats_freq_only = available_feats + freq_feat_names
    feats_freq_only = [c for c in feats_freq_only if c in df_tr.columns and c in df_val.columns]
    X_train_fo = df_tr.select(feats_freq_only).to_pandas()
    X_val_fo = df_val.select(feats_freq_only).to_pandas()
    score_fo, _ = quick_eval(X_train_fo, y_train, X_val_fo, y_val, 'freq_only')
    results['freq_only'] = score_fo

    del df, df_tr, df_val, X_train, y_train, X_val, y_val; gc.collect()

    # ═══════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════
    print('\n' + '='*60)
    print('EXPERIMENT SUMMARY')
    print('='*60)
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        marker = ' <<<' if v == max(results.values()) else ''
        print(f'  {k:25s}  PR-AUC={v:.4f}{marker}')

    with open(ROOT / 'experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    log('Results saved to experiment_results.json')


if __name__ == '__main__':
    main()
