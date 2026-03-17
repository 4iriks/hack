# Data Fusion 2026 "Страж" — Полный отчёт
# Fraud Detection Pipeline: путь от 0.090 до 0.101
# Дата: 17 марта 2026

---

## 1. ЗАДАЧА

Детекция мошеннических банковских транзакций.
- Метрика: **PR-AUC** (Precision-Recall Area Under Curve)
- 3 класса: fraud (target=1), confirmed (target=0), unlabeled (normal)
- ~100K клиентов, 200M+ операций, 4 периода данных
- Test = последний случайный день каждого клиента (633,683 операций)
- Лидерборд: public=30% (недели 1,3,5), private=70% (остальные 7 недель)

---

## 2. ДАННЫЕ

### 2.1 Структура (23 колонки на транзакцию)
```
customer_id              — ID клиента
event_id                 — ID транзакции (уникальный)
event_dttm               — дата-время ("2024-10-01 05:29:14")
event_type_nm            — тип события (закодирован)
event_desc               — описание события (закодирован)
channel_indicator_type   — канал (мобайл/веб/POS)
channel_indicator_sub_type — подтип канала
operaton_amt             — сумма операции
currency_iso_cd          — валюта
mcc_code                 — категория мерчанта
pos_cd                   — тип терминала (чип/ручной/CNP)
accept_language          — язык браузера (часто null)
browser_language         — язык (часто null)
timezone                 — часовой пояс (часто null)
session_id               — ID сессии (часто null)
operating_system_type    — ОС (часто null)
battery                  — заряд батареи (часто null / -1.0)
device_system_version    — версия устройства (часто null)
screen_size              — размер экрана (часто null)
developer_tools          — dev tools (часто null)
phone_voip_call_state    — VoIP звонок (0/1)
web_rdp_connection       — RDP подключение (0/1)
compromised              — скомпрометирован (часто null)
```

### 2.2 Периоды
| Период | Даты | Строк | Файлы | Используем |
|--------|------|-------|-------|-----------|
| Pretrain | Oct'23 → Sep'24 | 90M | 3 × 600MB | Для customer profiles |
| Train | Oct'24 → May'25 | 85M | 3 × 650MB | Обучение модели |
| Pretest | Jun'25 → Aug'25 | 14M | 1 × 324MB | НЕ используем (пока) |
| Test | случайный день | 633K | 1 × 17MB | Предсказываем |

### 2.3 Метки (train_labels.parquet)
- target=1 (fraud): 51,438 транзакций
- target=0 (confirmed): 36,076 транзакций — подозрительные, но подтверждённые клиентом
- unlabeled: ~84,900,000 — считаются нормальными
- Итого помечено: 87,514 из 85M (0.1%)

### 2.4 Ключевое свойство данных
- 100% клиентов из test есть в train (94,207 из 94,241)
- Медиана 1700 транзакций на клиента → достаточно для профилирования
- Distribution shift: adversarial AUC=1.0 между train и test (идеальное разделение)

---

## 3. ТЕКУЩИЙ ЛУЧШИЙ ПОДХОД (v14-C, LB=0.1010)

### 3.1 Сборка обучающих данных
Из 85M строк train берём:
- ВСЕ 87K помеченных (51K fraud + 36K confirmed)
- ~2.6M случайных unlabeled (normal) в пропорции ~50:1
- Итого: 2.66M строк
- Стратегия: "labeled+green" (fraud vs confirmed + sampled normal)

### 3.2 Validation
- Из полных 85M train берём 1 случайный день каждого клиента (имитация теста)
- 523K строк, 430 fraud (0.082%)
- Используется для early stopping (критически важно!)

### 3.3 Feature Engineering (121 фича)

**91 базовая фича:**
- Временные: hour, weekday, month, is_night
- Суммы: log_amount, is_null_amount
- Velocity: cnt_1h/6h/24h/7d/30d, amt_sum_1h/6h/24h/7d/30d
- Ratios: amt_ratio_7d_30d, cnt_ratio_24h_7d, cnt_ratio_7d_30d
- Сессия: session_ops_before, session_amt_before
- Новизна: is_new_mcc, is_new_channel, cum_unique_mcc
- Безопасность: security_flags_sum, voip_cnt_24h
- Время между: secs_since_last, dormancy_days
- Категории: event_type_nm, event_desc, channel_*, pos_cd, mcc_code

**16 customer profiles (из pretrain, 90M строк):**
- cust_avg/std/med/max/p95_amt — статистики сумм
- cust_n_tx, cust_n_unique_mcc/channel — объёмы
- cust_avg_hour, cust_hour_std, cust_pct_night — временные паттерны
- cust_pct_high_risk, cust_tenure_days — риск и стаж
- cust_avg_gap_sec, cust_last_epoch — частота

**30 per-customer anomaly фич (ГЛАВНЫЙ ПРОРЫВ v14-C):**
Профили из 177M строк (pretrain+train), для каждой транзакции — насколько она аномальна для ЭТОГО клиента:
- anom_amt_zscore — z-score суммы vs история клиента
- anom_amt_vs_median/p95/p99 — ratio к привычным суммам
- anom_amt_above_max — сумма больше исторического максимума
- anom_amt_iqr_score — outlier по IQR
- anom_hour_zscore — нетипичный час для клиента
- anom_night_unusual — ночь для "дневного" клиента
- anom_weekend_unusual — выходной для "будничного" клиента
- anom_mcc_novel — MCC которого клиент НИКОГДА не использовал
- anom_mcc_familiarity — log(кол-во транзакций клиента в этом MCC)
- anom_amt_vs_mcc_typical — сумма vs обычная для клиента в ЭТОМ MCC
- anom_gap_zscore — нетипичный gap между транзакциями
- anom_daily_velocity — транзакций за день vs обычное
- anom_hourly_burst — burst активности за час

### 3.4 Модель
- LightGBM GPU, binary classification
- scale_pos_weight=50 (компенсация дисбаланса fraud:normal)
- learning_rate=0.02, num_leaves=127, min_child_samples=200
- subsample=0.7, colsample_bytree=0.6
- n_estimators=10000, early_stopping patience=300
- Early stopping на полном val (523K) — критически важно!
- 3-5 seed ансамбль → среднее предсказаний

### 3.5 Предсказание
- Те же 121 фич для test (633K строк)
- Среднее 3-5 моделей → CSV

---

## 4. ХРОНОЛОГИЯ ЭКСПЕРИМЕНТОВ

### 4.1 Полная LB таблица (отсортирована по PR-AUC)
| # | Версия | LB PR-AUC | Val PR-AUC | Стратегия | Результат |
|---|--------|----------|-----------|-----------|-----------|
| 1 | **v14-C** | **0.1010** | **0.044** | **base+anomaly 121 фич, ES** | **РЕКОРД** |
| 2 | v15-B | 0.0989 | 0.045 | 139 фич, 5K fixed iter | device+seq |
| 3 | v14-D | 0.0984 | 0.055 | anomaly+20K trees | переобучение |
| 4 | v15-A | 0.0981 | 0.045 | 121 фич, 5K fixed iter | fixed iter вредит |
| 5 | v10-M1 | 0.0973 | 0.039 | proper val, 5 seeds | бывший рекорд |
| 6 | v13-C | 0.0951 | 0.045 | all feats, 30K trees | переобучение |
| 7 | v8 | 0.0951 | 0.31* | labeled+green 50:1 | 2.6M строк |
| 8 | v6 | 0.0945 | — | labeled+green 5:1 | 90 фич, 5 сидов |
| 9 | v3f | 0.0940 | — | labeled+green 5:1 | 41 фич, 3 сида |
| 10 | v9-E | 0.0932 | 0.32* | two-stage blend | хуже v8 |
| 11 | v9-A | 0.0929 | 0.27* | two-stage susp×fraud | хуже v8 |
| 12 | v10+boost | 0.0910 | 0.044 | v10 + cust_max^3.13 | boost вредит |
| 13 | v1 | 0.0904 | — | labeled+green | baseline |
| 14 | v12b-M1 | 0.0894 | 0.051 | hard neg + customer boost | val↑ LB↓ |
| 15 | v11 | 0.0754 | 0.034 | new feats + 3 models | провал |
| 16 | v2 | 0.0741 | — | +25 фич | провал |
| 17 | v5 | 0.0679 | — | labeled-only | провал |

*Val до proper validation (random split, завышен)

### 4.2 Этапы развития

**Этап 1: Baseline (v1→v3f, LB 0.090→0.094)**
- v1: первый baseline, labeled+green стратегия, 41 фича → LB=0.090
- v2: +25 фич → LB=0.074 (ХУЖЕ! больше фич = больше шума)
- v3f: multi-seed, deduplication → LB=0.094
- v5: labeled-only (без normal) → LB=0.068 (ПРОВАЛ)
- v6: 90 фич, 5 seeds → LB=0.095

**Этап 2: Масштабирование данных (v8-v9, LB 0.095)**
- v8: полный train 2.6M строк (50:1 sampling), 3 модели → LB=0.095
- v9: two-stage модели (suspicious × fraud) → LB=0.093 (хуже!)
- Вывод: больше данных не помогает, усложнение модели не помогает

**Этап 3: Proper Validation (v10, LB=0.097)**
- Создали правильный val: 1 случайный день каждого клиента из 85M строк
- 523K строк, 430 fraud — имитация теста
- Early stopping на этом val → стабильные модели
- LB=0.097 (+2.3% от v8)
- Это "бесплатный" прирост от правильной инфраструктуры

**Этап 4: Пост-процессинг (v10+boost, v12b, LB=0.089-0.091)**
- Customer boost: score × cust_max^3.13 → val+13%, LB-6.5% (ХУЖЕ!)
- Hard negative mining: val+16%, с boost val+35% → LB-8.1% (ЕЩЁ ХУЖЕ!)
- ВЫВОД: val систематически врёт для пост-процессинга. Val ↑ → LB ↓.

**Этап 5: Adversarial Validation (исследование)**
- Train vs Test AUC = 1.0000 — ПОЛНОЕ разделение
- Главный виновник: dormancy_days (shift=2.13σ, 98.8% adversarial importance)
- Shift ПЕРВАСИВНЫЙ: удаление фич не помогает, нормализация не помогает
- Причина: train = random sample, test = полные дни из более позднего периода
- Velocity фичи (cnt_6h, cnt_24h) систематически выше в test

**Этап 6: Борьба со shift (v13, LB=0.095)**
- 30K деревьев: val+15%, LB-2.3% → переобучение на shift
- Удаление leaky фич: не помогает (фичи нужны для fraud detection)
- Сильная регуляризация: val-21% → убивает модель
- ВЫВОД: нельзя бороться с shift удалением или регуляризацией

**Этап 7: Per-Customer Anomaly (v14, LB=0.101) ← ПРОРЫВ**
- Deep customer profiles из 177M строк (39 колонок)
- 18 anomaly features: z-score суммы, необычный час, новый MCC и т.д.
- val+14%, LB+3.8% → ПЕРВОЕ подтверждённое улучшение на LB!
- Почему работает: time-invariant, per-customer, не пост-процессинг
- 20K деревьев (v14-D): val+41% но LB-1.1% → переобучение

**Этап 8: Device + Sequence фичи (v15, LB=0.098-0.099)**
- Device graph: fingerprint устройства, fraud rate, "фермы" → 0.4-0.8% importance
- Sequence: переходы между транзакциями, card testing → 0.1-0.7% importance
- Battery: все значения -1.0 в preprocessed → бесполезно (0.0% importance)
- Temporal reweighting: tau=90 → хуже на val
- Фиксированные 5K iter vs early stopping: -2.9% на LB!
- ВЫВОД: добавление фич не помогает, early stopping критичен

---

## 5. КЛЮЧЕВЫЕ НАХОДКИ

### 5.1 Что РАБОТАЕТ (доказано на LB)
| Подход | Прирост LB | Версия |
|--------|-----------|--------|
| Per-customer anomaly features | +3.8% (0.097→0.101) | v14-C |
| Proper validation (1 день/клиент) | +2.3% (0.095→0.097) | v10 |
| labeled+green стратегия | базовая | v1+ |
| Multi-seed ансамбль (3-5 seeds) | +1-2% | v3f→v6 |
| Early stopping на полном val | -2.9% без него | v15 |

### 5.2 Что НЕ РАБОТАЕТ (доказано на LB)
| Подход | Эффект LB | Версия |
|--------|----------|--------|
| Добавление фич (device, sequence) | -2% до -26% | v2, v11, v15 |
| Customer boost (пост-процессинг) | -6.5% до -8% | v10+boost, v12b |
| Больше деревьев (20K, 30K) | -2% до -6% | v13, v14-D |
| Temporal reweighting (tau=90) | хуже на val | v15 |
| Фиксированные итерации (5K) | -2.9% | v15-A |
| labeled-only (без normal) | -32% | v5 |
| Two-stage модели | -2% | v9 |
| Hard negative mining | -8% | v12b |

### 5.3 Что проверяли в данных
| Сигнал | Результат |
|--------|----------|
| battery="100%" → эмулятор | В raw данных 5x fraud rate, но в preprocessed ВСЕ = -1.0. Бесполезен |
| RDP + VoIP combo → соц. инженерия | 0 случаев в train |
| Session sharing (2+ клиентов) | 6289 сессий, всего 14 fraud → слишком мало |
| Device "фермы" (10+ клиентов) | 1366 в train, 0.4% importance → слабо |
| pos_cd downgrade (чип→ручной) | 0.0% importance |
| compromised column | Почти всё null |
| pos_cd=1 (ручной ввод) | 5.7x lift fraud vs confirmed (используется) |
| phone_voip_call_state | 11x lift fraud vs confirmed (используется) |

### 5.4 Главная проблема: Distribution Shift
- Adversarial AUC=1.0 между train и test — модель ИДЕАЛЬНО различает периоды
- dormancy_days: train mean=129, test mean=279 (shift=2.13σ)
- Velocity фичи (cnt_6h/24h) выше в test (полные дни vs random sample)
- Shift ПЕРВАСИВНЫЙ: нельзя убрать удалением фич или нормализацией
- Единственное что работает через shift: TIME-INVARIANT per-customer фичи

### 5.5 Паттерн "Val ↑ → LB ↓"
Каждое "улучшение" на val ухудшало LB, КРОМЕ:
- Proper validation (v10): инфраструктурное улучшение
- Per-customer anomaly (v14-C): time-invariant фичи

Причина: val имеет "утечку" — модель обучалась на ДРУГИХ днях ТЕХ ЖЕ клиентов.
Любой пост-процессинг или feature engineering, эксплуатирующий знакомство с клиентами,
показывает ложный прирост на val но не обобщается на LB.

---

## 6. ПРАВИЛА (извлечённые уроки)

1. **ВСЕГДА** early stopping на полном val (523K) — не субсемпл, не фиксированные iter
2. **НЕ** добавлять фичи бездумно — каждый раз ухудшало LB
3. **НЕ** customer boost / пост-процессинг — утечка через клиентов
4. **НЕ** увеличивать деревья >10K — переобучение на shift
5. **НЕ** labeled-only — LB=0.068, провал
6. **НЕ** доверять val для пост-процессинга — val врёт
7. **ТОЛЬКО** time-invariant per-customer фичи дали подтверждённый прирост на LB
8. Калибровка val→LB стабильна: коэффициент ~2.3 (val=0.044 → LB=0.101)

---

## 7. ТЕХНИЧЕСКАЯ ИНФРАСТРУКТУРА

### 7.1 Железо
- RAM: 32GB
- GPU: RTX 3080 Ti 12GB
- OS: Linux

### 7.2 Стек
- Python 3, LightGBM (GPU), Polars, pandas, numpy, scikit-learn
- Полный pipeline: ~10 мин на 1 seed (2.6M строк, GPU LGBM)
- 3 seeds × 1 exp = ~30 мин

### 7.3 Файловая структура
```
main_data/              — исходные данные
  train_labels.parquet  — 87K меток
  sample_submit.csv     — формат сабмита

Pre-train_Train/        — pretrain (3×600MB) + train (3×650MB)
Pre-test_Test/          — pretest (324MB) + test (17MB)

features/               — кэшированные фичи
  train_features_full.parquet  — 2.6M строк (assembled)
  val_proper.parquet           — 523K строк (proper val)
  test_features.parquet        — 633K строк
  customer_profiles.parquet    — 16 фич из pretrain
  deep_customer_profiles.parquet — 39 фич из pretrain+train
  customer_mcc_profiles.parquet  — per-customer-MCC профили
  device_graph.parquet         — device fingerprints
  prev_tx_lookup.parquet       — sequence features (85M строк)
  pretest_profiles.parquet     — профили из pretest (НЕ ИСПОЛЬЗУЮТСЯ)
  _tmp_train/                  — 32 чанка по ~2.7M строк

models_v*/              — сохранённые модели по версиям
submissions/            — CSV для загрузки на лидерборд

pipeline_v*.py          — код каждой версии (v1→v15)
FINDINGS.md             — детальный лог всех экспериментов
CLAUDE.md               — инструкции и контекст проекта
```

---

## 8. АНАЛИЗ ОШИБОК (из v10)

### 8.1 Customer-level vs Transaction-level
- Customer-level PR-AUC = 0.214 (модель ХОРОШО находит fraud-клиентов)
- Transaction-level PR-AUC = 0.039 (но тонет в их обычных транзакциях)
- 210 fraud-клиентов в val, в среднем 16.5 транзакций/день

### 8.2 Пропущенный fraud = "тихие" транзакции
- Пойманные: amt=5.9M, amt_zscore=+5.7, secs_since_last=3ч (явно аномальные)
- Пропущенные: amt=2.1M, amt_zscore=-0.6, secs_since_last=9ч (неотличимы от нормы)

### 8.3 False Positives
- Top-430 предсказаний: 39 fraud + 2 confirmed + 389 green
- Модель НЕ путает fraud с confirmed — путает с обычными (green)

### 8.4 Сегменты
- VoIP=1: PR-AUC=0.124 (3× лучше средней)
- Суммы 10K-50K: PR-AUC=0.074
- Ночь: PR-AUC=0.067
- Вечер: PR-AUC=0.027 (худший)

### 8.5 Feature Importance (v14-C, топ-10)
| # | Фича | Importance |
|---|------|-----------|
| 1 | dormancy_days | 2.6% |
| 2 | cust_pct_high_risk | 2.3% |
| 3 | cust_tenure_days | 2.3% |
| 4 | prof_hour_entropy | 2.3% |
| 5 | cust_avg_hour | 2.2% |
| 6 | cust_hour_std | 2.1% |
| 7 | prof_mcc_entropy | 2.1% |
| 8 | cust_pct_night | 2.1% |
| 9 | prof_tx_per_day | 2.1% |
| 10 | cum_unique_mcc_approx | 2.0% |

Anomaly фичи не в топ-10, но суммарно дают ~8% importance и именно они дали прирост LB.

---

## 9. ПЛАН ДАЛЬНЕЙШИХ ДЕЙСТВИЙ (v16+)

### 9.1 Приоритет 1: Pretest Profiles + Drift (потенциал LB 0.11+)
Пересчитать customer profiles из 14M pretest строк (Jun-Aug'25).
Pretest = тот же период что test → distribution shift для profile-фичей исчезает.

Drift фичи: сравнение pretrain vs pretest нормы клиента:
- drift_amt = pretest_avg / pretrain_avg (рост/падение трат)
- drift_mcc_change (сменились ли любимые MCC)
- drift_hour_shift (сместилось ли время активности)

Сложность: средняя, ~2-3 часа.

### 9.2 Приоритет 2: Z-score нормализация по месяцу
Вместо raw cnt_24h=10 → (10 - month_mean) / month_std.
Убирает временной тренд из базовых фичей.

Сложность: ~1 час.

### 9.3 Приоритет 3: Red vs Yellow score как фича
Маленький LGBM на 87K (fraud vs confirmed), его predict как доп. фича.
НЕ labeled-only модель, а именно фича из вспомогательной модели.

Сложность: ~30 мин.

### 9.4 Приоритет 4: Adversarial IPW (clipped)
Классификатор train vs test, вес = p/(1-p) с clipping max=10.
Заставляет LGBM игнорировать фичи, специфичные для train-периода.

Сложность: ~2 часа. Риск: extreme weights из-за AUC=1.0.

---

## 10. ГЛАВНЫЕ ВЫВОДЫ

1. **Distribution shift — корневая проблема.** AUC=1.0 между train и test. Любой подход,
   не учитывающий shift, обречён на провал или маргинальный прирост.

2. **Time-invariant per-customer фичи — единственный доказанный путь.**
   "Насколько эта транзакция аномальна для ЭТОГО клиента" работает и в train, и в test.

3. **Val ненадёжен для предсказания LB.** Прирост на val часто = падение на LB.
   Единственное исключение: time-invariant фичи (v14-C).

4. **Простота побеждает сложность.** 121 фича + LGBM + early stopping > 139 фич,
   two-stage модели, hard negative mining, пост-процессинг.

5. **Early stopping критичен.** Фиксированные итерации дали -2.9% на LB.
   Полный val (523K) обязателен.

6. **Потолок текущего подхода ~0.101.** Для прорыва нужна смена парадигмы:
   pretest profiles (актуальные нормы клиентов вместо годичной давности).
