# Data Fusion 2026 "Страж" — Fraud Detection Pipeline

## Задача
Детекция фрода в банковских транзакциях. PR-AUC метрика.
3 класса: fraud (target=1), confirmed (target=0), unlabeled (normal).
100K клиентов, 200M+ операций, 4 периода (pretrain/train/pretest/test).
Test = последний случайный день каждого клиента (633,683 ops).
LB: public=30% (недели 1,3,5), private=70% (остальные 7 недель).

## Текущее состояние (2026-03-17)
- Лучший LB: **0.1010** (pipeline_v14-C, 121 фич, early stopping)
- Потолок текущего подхода (LGBM + 121 фич): ~0.101
- Реалистичный потенциал с новыми стратегиями: 0.12-0.14
- Kaggle API подключён (KGAT токен в ~/.bashrc)

## Полная LB история (отсортирована по PR-AUC)
| Submit | PR-AUC | Strategy | Notes |
|--------|--------|----------|-------|
| **v14-C** | **0.1010** | **base+anomaly 121 фич, ES** | **РЕКОРД** |
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
6. Val ≠ LB: val↑ часто → LB↓ (кроме time-invariant фич)
7. Time-invariant per-customer фичи — единственный подтверждённый прирост на LB
8. Базовая стратегия: labeled+green (fraud vs confirmed+sampled_normal)
9. НЕ pretest-based фичи — val↑21% но LB↓4.5%, утечка через клиентов (v16-B)
10. НЕ z-score velocity — деревья не нуждаются в нормализации, ломает фичи (v16-Z)
11. IPW: clip20=0.1009 (≈рекорд), clip10=0.0981, clip50=0.0992, clip100=0.0980. Парабола: clip20=оптимум, выше/ниже хуже

## Доказанные факты
- Distribution shift: adversarial AUC=1.0 между train/test
- dormancy_days: shift=2.13σ, 98.8% adversarial importance
- Shift ПЕРВАСИВНЫЙ: удаление/нормализация/IPW фич не помогает
- Customer-level PR-AUC=0.214 vs transaction-level=0.039
- Калибровка val→LB стабильна: коэффициент ~2.3
- battery = -1.0 во ВСЕХ preprocessed строках (бесполезен)
- RDP×VoIP = 0 случаев в train
- Pretest profiles: val+22% но LB-4.5% (утечка, доказано v16-B)
- Top shift features (adversarial): dormancy_days, cum_unique_mcc, month, day_of_month, session_amt

## Что не работает в данных
- battery (все -1.0), compromised (все null), developer_tools (все null)
- RDP+VoIP combo (0 случаев), session sharing (14 fraud из 6289)
- Device "фермы" (0.4% importance), pos_cd downgrade (0.0% importance)
- Sequence features (0.1-0.7% importance, 0 прирост LB)

## План v16+ (приоритеты)
1. **Pretest profiles + drift фичи** — пересчёт customer profiles из 14M pretest (Jun-Aug'25 = эпоха теста)
2. **Z-score нормализация по месяцу** — убирает temporal shift из velocity фичей
3. **Red vs Yellow score как фича** — LGBM на 87K (fraud vs confirmed), predict как доп. фича
4. **Adversarial IPW clipped** — w=p/(1-p), clip max=10-20, борьба с shift через веса
5. **Focal Loss** — custom objective для LGBM, фокус на hard examples
6. **Двухстадийный** — green-vs-notgreen → red-vs-yellow (НЕ как v9!)

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
  device_graph, prev_tx_lookup, _tmp_train/ (32 чанка)
- models_v*/ — сохранённые модели
- submissions/ — CSV для LB

## Ключевые файлы
- pipeline_v14.py — РЕКОРД (LB=0.101): anomaly features + LGBM + ES
- pipeline_v10.py — proper val + baseline LGBM (LB=0.097)
- pipeline_v9.py — add_features() и FEATURE_COLS (91 фича)
- pipeline_v15.py — device+sequence эксперименты (не помогло)
- FINDINGS.md — полный лог всех экспериментов
- REPORT.md — сводный отчёт проекта

## Доступные колонки данных
customer_id, event_id, event_dttm, event_type_nm, event_desc,
channel_indicator_type, channel_indicator_sub_type, operaton_amt,
currency_iso_cd, mcc_code, pos_cd, accept_language, browser_language,
timezone, session_id, operating_system_type, battery,
device_system_version, screen_size, developer_tools,
phone_voip_call_state, web_rdp_connection, compromised
