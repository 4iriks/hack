# Data Fusion 2026 "Страж" — Fraud Detection Pipeline

## Задача
Детекция фрода в банковских транзакциях. PR-AUC метрика.
3 класса: fraud (target=1), confirmed (target=0), unlabeled (normal).
100K клиентов, 200M+ операций, 4 периода (pretrain/train/pretest/test).
Test = последний случайный день каждого клиента (633,683 ops).
LB: public=30% (недели 1,3,5), private=70% (остальные 7 недель).

## Текущее состояние (2026-03-19)
- Лучший LB: **0.1010** (pipeline_v14-C, 121 фич, early stopping)
- v18 (honest point-in-time): val=0.0439, LB=0.0944 (-6.5%, mismatch)
- v18b (hybrid: v14-C + honest dormancy + test fix): val=0.0492, 5 seeds, submit ready, ожидает LB
- Потолок текущего подхода (LGBM + 121 фич): ~0.101
- Реалистичный потенциал с новыми стратегиями: 0.12-0.14
- Kaggle API подключён (KGAT токен в ~/.bashrc)

## Полная LB история (отсортирована по PR-AUC)
| Submit | PR-AUC | Strategy | Notes |
|--------|--------|----------|-------|
| **v14-C** | **0.1010** | **base+anomaly 121 фич, ES** | **РЕКОРД** |
| v18 | 0.0944 | honest PIT profiles, 121 фич | val≈v14-C, LB-6.5% mismatch |
| v18b | ??? | v14-C + honest dormancy + test fix | val=0.049, ожидает LB |
| v16-B | 0.0965 | +5 pretest anom фич, 126 total | val↑21% LB↓4.5% |
| v15-B | 0.0989 | 139 фич, 5K fixed iter | device+seq, fixed iter вредит |
| v14-D | 0.0984 | anomaly+20K trees | 20K = переобучение |
| v16-IPW50 | 0.0992 | 121 фич, IPW clip=50 | clip20 лучше |
| v16-IPW20 | 0.1009 | 121 фич, IPW clip=20 | ≈рекорд! оптимальный clip |
| v16-IPW100 | 0.0980 | 121 фич, IPW clip=100 | слишком агрессивно |
| v16-IPW10 | 0.0981 | 121 фич, IPW clip=10 | val↑14.7% LB↓2.9% |
| v15-A | 0.0981 | 121 фич, 5K fixed iter | fixed iter = -2.9%! |
| v10-M1 | 0.0973 | proper val, 5 seeds | бывший рекорд |
| v13-C | 0.0951 | all feats, 30K trees | 30K = переобучение |
| v8 | 0.0951 | labeled+green 50:1 | 2.6M строк |
| v6 | 0.0945 | labeled+green 5:1 | 90 фич, 5 сидов |
| v3f | 0.0940 | labeled+green 5:1 | 41 фич, 3 сида |
| v9-E | 0.0932 | two-stage blend | хуже v8 |
| v9-A | 0.0929 | two-stage susp×fraud | хуже v8 |
| v10+boost | 0.0910 | v10 + cust_max^3.13 | boost вредит |
| v1 | 0.0904 | labeled+green | baseline |
| v12b-M1 | 0.0894 | hard neg + cust boost | val↑ LB↓ |
| v11 | 0.0754 | new feats + 3 models | ПРОВАЛ |
| v2 | 0.0741 | +25 фич | ПРОВАЛ |
| v5 | 0.0679 | labeled-only | ПРОВАЛ |

## Доказанные правила (v1→v16)
1. ВСЕГДА early stopping на полном val (523K) — НЕ субсемпл, НЕ фиксированные iter
2. НЕ добавлять фичи бездумно — каждый раз ухудшало LB (v2, v11, v15, v16-B)
3. НЕ customer boost — утечка, -6.5% на LB
4. НЕ labeled-only — 0.068, провал
5. НЕ больше 10K деревьев — переобучение на shift
6. Val ≠ LB: val↑ часто → LB↓ (частично из-за temporal leakage в prof_*/anom_*)
7. Time-invariant per-customer фичи — единственный подтверждённый прирост на LB
8. Базовая стратегия: labeled+green (fraud vs confirmed+sampled_normal)
9. НЕ pretest-based фичи — val↑21% но LB↓4.5%, утечка через клиентов (v16-B)
10. НЕ z-score velocity — деревья не нуждаются в нормализации, ломает фичи (v16-Z)
11. IPW: clip20=0.1009 (≈рекорд), clip10=0.0981, clip50=0.0992, clip100=0.0980. Парабола: clip20=оптимум, выше/ниже хуже
12. TEMPORAL LEAKAGE: deep profiles (prof_*, anom_*) из 177M ops содержат будущее для val. Честный val=0.031 vs утёкший=0.044 (-30%). cust_* чистые (pretrain only)
13. HONEST PIT (v18): LB=0.0944 (-6.5% vs v14-C). Причина: train на аппроксимациях (median≈mean), test на точных profiles → mismatch. Leaked фичи ближе к test по распределению

## Доказанные факты
- Distribution shift: adversarial AUC=1.0 между train/test
- dormancy_days: shift=2.13σ, 98.8% adversarial importance
- Shift ПЕРВАСИВНЫЙ: удаление/нормализация/IPW фич не помогает
- Customer-level PR-AUC=0.214 vs transaction-level=0.039
- battery = -1.0 во ВСЕХ preprocessed строках (бесполезен)
- RDP×VoIP = 0 случаев в train
- Pretest profiles: val+22% но LB-4.5% (утечка, доказано v16-B)
- Top shift features (adversarial): dormancy_days, cum_unique_mcc, month, day_of_month, session_amt
- **TEMPORAL LEAKAGE в deep profiles**: prof_* и anom_* считались по pretrain+train (177M ops). Val-транзакция в ноябре "видит" профиль до мая = утечка. Честный val = 0.031 vs утёкший 0.044 (-30%). cust_* чистые (pretrain only). Скрипт: analytics/02_honest_val.py
- **v18 PIT: LB=0.0944 (-6.5%)**: честные PIT-аппроксимации при обучении + точные deep profiles на test = mismatch. Leaked фичи парадоксально лучше для LB — они ближе к test-распределению
- **dormancy_days в v14-C СЛОМАНА**: формула event_dttm - cust_last_epoch(pretrain) даёт ~129 дней (константа на клиента). Honest dormancy (shift(1)) = 0.3 дня (реальный интервал). Quick test +13.5% val. Но test dormancy тоже нужно пересчитать (старая = 270+ дней)

## Что не работает в данных
- battery (все -1.0), compromised (все null), developer_tools (все null)
- RDP+VoIP combo (0 случаев), session sharing (14 fraud из 6289)
- Device "фермы" (0.4% importance), pos_cd downgrade (0.0% importance)
- Sequence features (0.1-0.7% importance, 0 прирост LB)

## Что осталось попробовать
1. **v18b hybrid** — ЗАВЕРШЕНО: val=0.0492, test dormancy fix (279→2 days), submit ready, ожидает LB
2. **LambdaRank** — ranking objective вместо binary CE (начато, не завершено)
3. **Focal Loss** — custom objective для hard examples (начато, не завершено)
4. **Другое ratio негативов** — 10:1, 20:1, 100:1 вместо 50:1
5. **Стекинг разных "взглядов"** — velocity-only + anomaly-only + profile-only → мета-модель
6. **Трансформер** — LBSF на последовательностях клиентов (Kaggle P100)

## Исчерпанные подходы (v16-v18)
- Pretest profiles: val+21% LB-4.5% (утечка)
- Z-score velocity: вредит деревьям
- **Honest PIT profiles (v18)**: val=0.044 (≈v14-C), LB=0.094 (-6.5%). Аппроксимации train↔test mismatch
- IPW: clip20=0.1009 (≈рекорд, не лучше), парабола clip10-100
- Blend v14-C + IPW: бесполезен (val≠LB)
- Red vs Yellow фича: val ХУЖЕ baseline
- Drift features: не завершено, скорее всего вредит

## Инфраструктура
- Локально: 32GB RAM, RTX 3080 Ti 12GB, LGBM GPU ~10мин/seed
- Kaggle: P100 16GB, 30GB RAM, 30ч/нед GPU (для transformer/тяжёлых)
- Стек: Python 3, LightGBM GPU, Polars, pandas, numpy, sklearn

## Правила работы
- Пиши код в новые файлы (pipeline_v16.py и т.д.), не трогай старые
- Логируй все эксперименты и результаты в FINDINGS.md
- При создании сабмита — сохраняй в submissions/ с временной меткой
- Проверяй RAM/GPU перед тяжёлыми процессами, не роняй ПК

## Структура данных
- main_data/ — train_labels.parquet (87K меток), sample_submit.csv
- Pre-train_Train/ — pretrain (3×600MB, Oct'23-Sep'24) + train (3×650MB, Oct'24-May'25)
- Pre-test_Test/ — pretest.parquet (14M, Jun-Aug'25) + test.parquet (633K)
- features/ — кэши: train_features_full (2.6M), val_proper (523K), test_features (633K),
  customer_profiles, deep_customer_profiles, customer_mcc_profiles, pretest_profiles,
  device_graph, prev_tx_lookup, honest_pit_profiles, _tmp_train/ (32 чанка)
- models_v*/ — сохранённые модели
- submissions/ — CSV для LB

## Ключевые файлы
- pipeline_v14.py — РЕКОРД (LB=0.101): anomaly features + LGBM + ES
- pipeline_v10.py — proper val + baseline LGBM (LB=0.097)
- pipeline_v9.py — add_features() и FEATURE_COLS (91 фича)
- pipeline_v15.py — device+sequence эксперименты (не помогло)
- v16_ipw.py / v16_ipw2.py — IPW эксперименты (clip5-100)
- v17_experiments.py — blend + RvY + focal loss
- v17_lambdarank.py — LambdaRank эксперимент
- v18_honest_features.py — Point-in-Time честные фичи, val=0.0439, chunked 87M rows
- v18b_hybrid_test.py — Гибрид: v14-C profiles + honest dormancy + test dormancy fix
- analytics/01_fundamentals.ipynb — фундаментальный аудит пайплайна
- analytics/02_honest_val.py — пересчёт профилей без temporal leakage, доказательство утечки -30%
- analytics/feature_analysis.md — справочник всех 121 фичей + данные для графиков
- FINDINGS.md — полный лог всех экспериментов
- REPORT.md — сводный отчёт проекта

## Доступные колонки данных
customer_id, event_id, event_dttm, event_type_nm, event_desc,
channel_indicator_type, channel_indicator_sub_type, operaton_amt,
currency_iso_cd, mcc_code, pos_cd, accept_language, browser_language,
timezone, session_id, operating_system_type, battery,
device_system_version, screen_size, developer_tools,
phone_voip_call_state, web_rdp_connection, compromised
