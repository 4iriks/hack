# Полный справочник фичей и данных v14-C
## Data Fusion 2026 "Страж" — Fraud Detection

> Модель-рекордсмен: v14-C, LB PR-AUC = 0.1010
> 121 фича = 91 базовая (v9) + 30 anomaly/profile (v14)
> Ансамбль 5 seeds, LGBM GPU, early stopping 300



---

# 1. ДАННЫЕ ДЛЯ ГРАФИКОВ

---

## 1.1 Логика разбиения (Validation Strategy)

### Временные диапазоны
```
Train assembled:  2024-10-01 00:00:05  →  2025-05-31 23:59:54   (8 месяцев)
Val proper:       2024-10-01 00:00:47  →  2025-05-31 23:57:22   (8 месяцев)
Test:             2025-06-01 00:00:13  →  2025-08-09 23:58:25   (70 дней)
```

### Размеры выборок
```
Train assembled:    2,659,398 строк
Val proper:           523,280 строк
Test:                 633,683 строк (уникальных event_id)
```

### Gap между выборками
```
Train → Val:    НЕТ (val выбирается из train периода, один случайный день/клиент)
Train → Test:   0 дней (test начинается на следующий день после train)
```

### Пересечение клиентов
```
Train клиентов:   99,952
Val клиентов:    100,000
Test клиентов:    94,241

Train ∩ Val:     99,952  (100.0% от val)
Train ∩ Test:    94,207  (100.0% от test)
Val ∩ Test:      94,241  (100.0% от test)
```
**Вывод:** 100% клиентов пересекаются — per-customer фичи работают.
5,759 клиентов из val отсутствуют в test (ушли/неактивны в Jun-Aug'25).

### Fraud в val
```
Val fraud:          430 из 523,280  (0.0822%)
Val fraud даты:     2024-10-03 → 2025-05-30
Val non-fraud даты: 2024-10-01 → 2025-05-31
```

### Состав Train assembled
```
Fraud (red):         51,172  ( 1.92%)  — target=1
Confirmed (yellow):  35,506  ( 1.34%)  — target=0 (labeled)
Green (unlabeled): 2,572,720  (96.74%)  — sampled normal ops
ИТОГО:            2,659,398

Negatives = yellow + green = 2,608,226
Ratio neg/pos = 51:1
scale_pos_weight = min(51, 50) = 50.0
```

### Схема разбиения
```
┌──────────────────── Oct'24 ────────────────── May'25 ──┐    ┌── Jun'25 ── Aug'25 ──┐
│                    TRAIN PERIOD                         │    │    TEST PERIOD        │
│  85M операций, 100K клиентов                           │    │  633K ops, 94K clients│
│                                                        │    │                       │
│  Train assembled: 2.66M строк (сэмпл)                 │    │  test.parquet         │
│  Val proper: 523K строк (1 день/клиент)                │    │  (1 день/клиент)      │
│                                                        │    │                       │
│  Метки: 87,514 (51K fraud + 36K confirmed)             │    │  Без меток            │
│  Green: ~85M обычных операций                          │    │                       │
└────────────────────────────────────────────────────────┘    └───────────────────────┘
                         ↕ 0 дней gap                              ↕
                    100% overlap клиентов                      94K из 100K
```

---

## 1.3 Feature Importance (Топ-20)

Модель: v14-C, 5 seeds, importance type = gain (суммарный прирост качества от сплитов)

```
 #   Фича                          Gain %   Cum %    CV     Категория
─────────────────────────────────────────────────────────────────────────
 1   anom_hourly_burst               6.41%    6.4%   0.32   anomaly
 2   mcc_code                        3.52%    9.9%   0.13   other
 3   prof_gap_std                    3.16%   13.1%   0.24   profile
 4   amt_ratio_7d_30d                2.61%   15.7%   0.21   velocity
 5   amt_sum_1h                      2.61%   18.3%   0.70   velocity
 6   dormancy_days                   2.22%   20.5%   0.30   customer
 7   event_desc                      2.18%   22.7%   0.07   other
 8   anom_daily_velocity             2.04%   24.7%   0.17   anomaly
 9   prof_hour_entropy               1.92%   26.7%   0.32   profile
10   prof_gap_mean                   1.90%   28.6%   0.16   profile
11   prof_tx_per_day                 1.83%   30.4%   0.27   profile
12   amt_sum_6h                      1.82%   32.2%   0.18   velocity
13   anom_amt_iqr_score              1.77%   34.0%   0.35   anomaly
14   amt_sum_24h                     1.75%   35.7%   0.41   velocity
15   cust_p95_amt                    1.61%   37.3%   0.33   customer
16   cust_pct_high_risk              1.60%   39.0%   0.36   customer
17   prof_amt_iqr                    1.54%   40.5%   0.34   profile
18   cust_med_amt                    1.49%   42.0%   0.36   customer
19   prof_mcc_entropy                1.47%   43.5%   0.36   profile
20   cust_avg_hour                   1.36%   44.8%   0.40   customer
```

### Покрытие
```
Топ-10:   28.6% всего gain
Топ-20:   44.8% всего gain
Нижние 50: 8.9% всего gain   ← "балласт"
```

### Importance по категориям (все 121 фича)
```
velocity (cnt_*, amt_sum_*, ratio_*):    ~30%
anomaly/profile (anom_*, prof_*):        ~35%
customer_profile (cust_*):               ~15%
temporal (hour, weekday, month):          ~8%
other (mcc_code, event_desc, flags):     ~12%
```

### Стабильность (CV между seeds)
```
Стабильные (CV < 0.2):    mcc_code (0.13), prof_gap_mean (0.16),
                           anom_daily_velocity (0.17), event_desc (0.07)
Нестабильные (CV > 0.5):  amt_sum_1h (0.70) — дерево сильно зависит от seed
```

---

## 1.4 Метрики: Local Val vs Лидерборд

```
Модель               Val PR-AUC    LB PR-AUC    Ratio    Delta vs рекорд
──────────────────────────────────────────────────────────────────────────
v14-C (РЕКОРД)         0.0440       0.1010      2.30       0.0%
v16-IPW20               0.0510       0.1009      1.98      -0.1%
v16-IPW50               0.0510       0.0992      1.95      -1.8%
v15-B                   0.0450       0.0989      2.20      -2.1%
v14-D                   0.0550       0.0984      1.79      -2.6%
v16-IPW10               0.0500       0.0981      1.96      -2.9%
v15-A                   0.0450       0.0981      2.18      -2.9%
v16-IPW100              0.0520       0.0980      1.88      -3.0%
v10-M1                  0.0390       0.0973      2.49      -3.7%
v16-B (pretest)         0.0540       0.0965      1.79      -4.5%
v13-C                   0.0450       0.0951      2.11      -5.8%
v11                     0.0340       0.0754      2.22     -25.3%
```

### Ключевые наблюдения
```
Средний ratio LB/Val:  ~2.1x  (LB всегда выше val)
Причина:  test fraud rate выше val fraud rate (≈0.17% vs 0.08%)
          + другие клиенты/периоды

АНТИКОРРЕЛЯЦИЯ:  Val↑ часто → LB↓
  v16-B:    val 0.054 (+23%) → LB 0.0965 (-4.5%)  ← pretest утечка
  v14-D:    val 0.055 (+25%) → LB 0.0984 (-2.6%)  ← переобучение 20K деревьев
  v16-IPW:  val 0.051 (+16%) → LB 0.1009 (-0.1%)  ← IPW не помогает на LB
```

### PR-кривая v14-C на Val
```
Точки на кривой (Recall → Precision):
  Recall=0.10  →  Precision ≈ 0.035
  Recall=0.25  →  Precision ≈ 0.015
  Recall=0.50  →  Precision ≈ 0.005

Val PR-AUC = 0.0440
Val fraud rate = 0.0822% (430 из 523,280)
```



---

# 2. ПОЛНЫЙ СПРАВОЧНИК ФИЧЕЙ (121)

---

## 2.1 Исходные колонки данных (из parquet)

Эти колонки приходят напрямую из данных. Часть используется как фичи, часть — для вычислений.

| Фича | Тип | Описание |
|------|-----|----------|
| `customer_id` | int | ID клиента (НЕ фича, для группировки) |
| `event_id` | int | ID операции (НЕ фича, для submission) |
| `event_dttm` | datetime | Дата/время операции |
| `operaton_amt` | float | Сумма операции в рублях |
| `mcc_code` | int | MCC-код торговой точки (категория покупки) |
| `pos_cd` | int | POS condition code (способ ввода карты) |
| `event_type_nm` | int | Тип события (закодирован числом) |
| `event_desc` | int | Описание события (закодирован числом) |
| `channel_indicator_type` | int | Канал операции (банкомат, онлайн, POS и т.д.) |
| `channel_indicator_sub_type` | int | Подтип канала |
| `currency_iso_cd` | int | Код валюты (ISO) |
| `operating_system_type` | int | ОС устройства (Android, iOS, Web...) |
| `timezone` | int | Часовой пояс клиента |
| `phone_voip_call_state` | int | Состояние VoIP-звонка (мошенник по телефону?) |
| `web_rdp_connection` | int | RDP-подключение (удалённый доступ?) |
| `compromised` | int | Устройство скомпрометировано? (ВСЕ null) |
| `developer_tools` | int | Dev Tools открыты? (ВСЕ null) |
| `battery` | float | Уровень заряда (ВСЕ -1.0, бесполезен) |

---

## 2.2 Предвычисленные фичи (из pipeline сборки)

Эти колонки вычисляются при создании train/val/test parquet файлов из сырых данных.

### Временные

| Фича | Формула | Описание |
|------|---------|----------|
| `hour` | `event_dttm.hour()` | Час операции (0-23) |
| `weekday` | `event_dttm.weekday()` | День недели (0=Пн, 6=Вс) |
| `month` | `event_dttm.month()` | Месяц (1-12) |
| `is_night` | `hour >= 22 OR hour < 6` | Ночная операция (1/0) |

### Логарифм суммы

| Фича | Формула | Описание |
|------|---------|----------|
| `log_amount` | `log(operaton_amt + 1)` | Логарифм суммы (сглаживает выбросы) |

### Velocity (скорость операций)

| Фича | Формула | Описание |
|------|---------|----------|
| `cnt_1h` | rolling count | Кол-во операций клиента за последний 1 час |
| `cnt_6h` | rolling count | Кол-во операций клиента за последние 6 часов |
| `cnt_24h` | rolling count | Кол-во операций клиента за последние 24 часа |
| `cnt_7d` | rolling count | Кол-во операций клиента за последние 7 дней |
| `cnt_30d` | rolling count | Кол-во операций клиента за последние 30 дней |
| `amt_sum_1h` | rolling sum | Сумма операций клиента за последний 1 час |
| `amt_sum_6h` | rolling sum | Сумма операций клиента за последние 6 часов |
| `amt_sum_24h` | rolling sum | Сумма операций клиента за последние 24 часа |
| `amt_sum_7d` | rolling sum | Сумма операций клиента за последние 7 дней |
| `amt_sum_30d` | rolling sum | Сумма операций клиента за последние 30 дней |
| `secs_since_last` | `current_time - prev_time` | Секунды с предыдущей операции клиента |
| `voip_cnt_24h` | rolling count | Кол-во операций с VoIP за 24ч |

### Флаги новизны

| Фича | Формула | Описание |
|------|---------|----------|
| `is_new_mcc_code` | first seen | Первый раз этот MCC у клиента? (1/0) |
| `is_new_channel_indicator_type` | first seen | Первый раз этот канал у клиента? (1/0) |
| `cum_unique_mcc_approx` | cumulative count | Сколько уникальных MCC клиент уже использовал |

### Сессия

| Фича | Формула | Описание |
|------|---------|----------|
| `session_ops_before` | count | Кол-во операций ДО этой в текущей сессии |
| `session_amt_before` | sum | Сумма операций ДО этой в текущей сессии |

### Безопасность

| Фича | Формула | Описание |
|------|---------|----------|
| `security_flags_sum` | `compromised + rdp + voip + dev_tools` | Сумма security-флагов (0-4) |
| `is_high_risk_type` | lookup | Тип события в списке "подозрительных" (1/0) |

---

## 2.3 Derived-фичи (pipeline_v9.py :: add_features)

Создаются функцией `add_features()` из предвычисленных колонок.

### Presence/Absence флаги

| Фича | Формула | Описание |
|------|---------|----------|
| `has_browser_lang` | `browser_language IS NOT NULL` | Есть язык браузера (1/0) |
| `has_accept_lang` | `accept_language IS NOT NULL` | Есть Accept-Language (1/0) |
| `has_device_ver` | `device_system_version IS NOT NULL` | Есть версия ОС (1/0) |
| `has_session` | `session_id IS NOT NULL` | Есть session_id (1/0) |
| `has_pos_data` | `pos_cd IS NOT NULL` | Есть POS condition code (1/0) |
| `is_no_screen` | `screen_size IS NULL` | Нет данных об экране (1/0) |
| `is_no_timezone` | `timezone IS NULL` | Нет часового пояса (1/0) |

### Screen dimensions

| Фича | Формула | Описание |
|------|---------|----------|
| `screen_w` | `screen_size.split('x')[0]` | Ширина экрана в пикселях |
| `screen_h` | `screen_size.split('x')[1]` | Высота экрана в пикселях |

### POS-код

| Фича | Формула | Описание |
|------|---------|----------|
| `is_manual_entry` | `pos_cd == 1` | Ручной ввод карты (самый рисковый POS-тип) |

### Velocity Ratios (отношения скоростей)

| Фича | Формула | Описание |
|------|---------|----------|
| `cnt_ratio_1h_24h` | `cnt_1h / (cnt_24h + 1)` | Доля часовых операций в дневных |
| `cnt_ratio_24h_7d` | `cnt_24h / (cnt_7d + 1)` | Доля дневных операций в недельных |
| `cnt_ratio_7d_30d` | `cnt_7d / (cnt_30d + 1)` | Доля недельных операций в месячных |
| `amt_ratio_1h_24h` | `amt_sum_1h / (|amt_sum_24h| + 1)` | Отношение сумм 1ч/24ч |
| `amt_ratio_24h_7d` | `amt_sum_24h / (|amt_sum_7d| + 1)` | Отношение сумм 24ч/7д |
| `amt_ratio_7d_30d` | `amt_sum_7d / (|amt_sum_30d| + 1)` | Отношение сумм 7д/30д |
| `amt_cur_ratio_24h` | `operaton_amt / (|amt_sum_24h| + 1)` | Текущая сумма vs дневной оборот |
| `amt_cur_ratio_7d` | `operaton_amt / (|amt_sum_7d| + 1)` | Текущая сумма vs недельный оборот |
| `amt_cur_ratio_30d` | `operaton_amt / (|amt_sum_30d| + 1)` | Текущая сумма vs месячный оборот |

**Суть:** Если ratio → 1.0, значит почти все операции сконцентрированы в короткий период (всплеск).

### Composit temporal

| Фича | Формула | Описание |
|------|---------|----------|
| `hour_weekday` | `hour * 7 + weekday` | Комбинация час×день (0-167), уникальный слот недели |
| `day_of_month` | `event_dttm.day()` | День месяца (1-31) |

### Security composite

| Фича | Формула | Описание |
|------|---------|----------|
| `security_risk_score` | `compromised*3 + rdp*2 + voip*2 + dev_tools*1` | Взвешенная сумма рисков (0-8) |

### Speed-derived

| Фича | Формула | Описание |
|------|---------|----------|
| `is_fast_60s` | `secs_since_last < 60` | Менее 60 сек с прошлой операции (1/0) |
| `log_secs_since_last` | `log(secs_since_last + 1)` | Логарифм интервала |

### Amount-derived

| Фича | Формула | Описание |
|------|---------|----------|
| `avg_amt_30d` | `amt_sum_30d / (cnt_30d + 1)` | Средняя сумма за 30 дней |
| `avg_amt_7d` | `amt_sum_7d / (cnt_7d + 1)` | Средняя сумма за 7 дней |
| `amt_deviation_30d` | `operaton_amt / (avg_amt_30d + 1)` | Текущая сумма vs средняя 30д |
| `amt_pct_of_30d` | `operaton_amt / (amt_sum_30d + 1)` | Доля текущей суммы в месячном обороте |

### Activity-derived

| Фича | Формула | Описание |
|------|---------|----------|
| `hourly_activity_ratio` | `cnt_1h / (cnt_24h/24 + 0.01)` | Часовая активность vs средняя за день |
| `is_first_in_session` | `session_ops_before == 0` | Первая операция в сессии (1/0) |
| `log_cnt_30d` | `log(cnt_30d + 1)` | Логарифм кол-ва операций за 30д |

### Interaction features

| Фича | Формула | Описание |
|------|---------|----------|
| `voip_x_logamt` | `phone_voip_call_state * log_amount` | VoIP × сумма (мошенник по телефону + большая сумма) |
| `manual_entry_x_amt` | `is_manual_entry * log_amount` | Ручной ввод × сумма (рисковая комбинация) |

---

## 2.4 Customer Profile фичи (pipeline_v9.py :: add_customer_profiles)

Предвычислены по 85M операциям train периода. Статические для каждого клиента.
Функция `add_customer_profiles()` джойнит профили и вычисляет отклонения.

### Статистики клиента (из customer_profiles.parquet)

| Фича | Описание |
|------|----------|
| `cust_avg_amt` | Средняя сумма операции клиента за всю историю |
| `cust_std_amt` | Стандартное отклонение суммы клиента |
| `cust_med_amt` | Медиана суммы клиента |
| `cust_max_amt` | Максимальная сумма клиента за всю историю |
| `cust_p95_amt` | 95-й перцентиль суммы клиента |
| `cust_n_tx` | Общее кол-во операций клиента |
| `cust_n_unique_mcc` | Кол-во уникальных MCC у клиента |
| `cust_n_unique_channel` | Кол-во уникальных каналов у клиента |
| `cust_avg_hour` | Средний час операций клиента (e.g. 14.3 = обычно после обеда) |
| `cust_hour_std` | Разброс часов операций |
| `cust_pct_night` | Доля ночных операций (22:00-06:00) |
| `cust_pct_high_risk` | Доля операций "высокого риска" |
| `cust_tenure_days` | Срок жизни клиента (дни от первой до последней операции) |
| `cust_avg_gap_sec` | Средний интервал между операциями (секунды) |

### Вычисляемые отклонения (в add_customer_profiles)

| Фича | Формула | Описание |
|------|---------|----------|
| `amt_zscore` | `(amt - cust_avg_amt) / (cust_std_amt + 1)` | Z-score суммы vs история клиента |
| `amt_vs_median` | `amt / (cust_med_amt + 1)` | Отношение суммы к медиане клиента |
| `dormancy_days` | `(event_epoch - cust_last_epoch) / 86400` | Дней с последней операции клиента |
| `hour_deviation` | `|hour - cust_avg_hour|` | Отклонение часа от типичного для клиента |
| `freq_vs_historical` | `cnt_30d / (30d / cust_avg_gap + 0.01)` | Текущая частота vs историческая |

**dormancy_days** — самая "shift-prone" фича: adversarial importance = 98.8%.
В train dormancy ≈ 0 (непрерывные данные), в test ≈ 30-70 дней (gap Jun→Aug).

---

## 2.5 Anomaly-фичи (pipeline_v14.py :: add_anomaly_features)

Вычисляются по deep_customer_profiles (177M операций pretrain+train).
Суть: насколько ТЕКУЩАЯ операция отклоняется от ИСТОРИЧЕСКОГО профиля клиента.

### Amount anomalies (отклонения суммы)

| Фича | Формула | Описание |
|------|---------|----------|
| `anom_amt_zscore` | `(amt - prof_mean) / prof_std` | Z-score суммы vs глубокий профиль |
| `anom_amt_vs_median` | `amt / prof_median` | Отношение к медиане профиля |
| `anom_amt_vs_p95` | `amt / prof_p95` | Отношение к 95-му перцентилю (>1 = аномалия) |
| `anom_amt_vs_p99` | `amt / prof_p99` | Отношение к 99-му перцентилю |
| `anom_amt_above_max` | `amt > prof_max` | Превышает максимум за всю историю? (1/0) |
| `anom_amt_iqr_score` | `(amt - prof_median) / prof_IQR` | IQR-score (устойчив к выбросам) |
| `anom_amt_below_min` | `amt < prof_min` | Ниже минимума за историю? (1/0) |

### Time anomalies (отклонения времени)

| Фича | Формула | Описание |
|------|---------|----------|
| `anom_hour_zscore` | `|hour - prof_mean_hour| / prof_hour_std` | Z-score часа (circular, clip=12) |
| `anom_night_unusual` | `is_night * (1 - prof_pct_night)` | Ночная операция у "дневного" клиента |
| `anom_weekend_unusual` | `is_weekend * (1 - prof_pct_weekend)` | Выходная операция у "будничного" клиента |

### MCC anomalies (отклонения категории)

| Фича | Формула | Описание |
|------|---------|----------|
| `anom_mcc_novel` | `mcc_n_tx IS NULL` | Клиент НИКОГДА не использовал этот MCC (1/0) |
| `anom_mcc_familiarity` | `log(mcc_n_tx + 1)` | Сколько раз клиент использовал этот MCC (log) |
| `anom_amt_vs_mcc_typical` | `(amt - mcc_mean) / prof_std` | Сумма vs типичная для этого MCC у клиента |
| `anom_mcc_share` | `mcc_n_tx / prof_n_tx` | Доля этого MCC в операциях клиента |

### Gap anomalies (отклонения интервала)

| Фича | Формула | Описание |
|------|---------|----------|
| `anom_gap_zscore` | `(secs_since_last - prof_gap_mean) / prof_gap_std` | Z-score интервала vs профиль |
| `anom_gap_vs_median` | `secs_since_last / prof_gap_median` | Интервал vs медианный для клиента |

### Velocity anomalies (отклонения скорости)

| Фича | Формула | Описание |
|------|---------|----------|
| `anom_daily_velocity` | `cnt_24h / prof_tx_per_day` | Дневная активность vs типичная (#8 в importance) |
| `anom_hourly_burst` | `cnt_1h / (prof_tx_per_day / 24)` | Часовой всплеск vs средняя часовая активность (**#1 в importance, 6.41%**) |

---

## 2.6 Profile-фичи (pipeline_v14.py, из deep_customer_profiles)

Статические характеристики клиента, вычисленные по 177M операциям (pretrain + train).
Используются как фичи напрямую, а также как базис для anomaly-вычислений.

| Фича | Описание |
|------|----------|
| `prof_amt_cv` | Коэффициент вариации суммы (std/mean) — насколько разнообразны суммы |
| `prof_hour_entropy` | Энтропия распределения часов (бит) — насколько разнообразно время операций |
| `prof_mcc_entropy` | Энтропия распределения MCC — насколько разнообразны категории покупок |
| `prof_voip_rate` | Доля операций с VoIP-звонком |
| `prof_rdp_rate` | Доля операций с RDP-подключением |
| `prof_tx_per_day` | Среднее кол-во операций в день |
| `prof_n_unique_mcc` | Кол-во уникальных MCC (за всю историю) |
| `prof_n_unique_channel` | Кол-во уникальных каналов |
| `prof_n_unique_os` | Кол-во уникальных ОС |
| `prof_amt_iqr` | Межквартильный размах суммы (P75-P25) |
| `prof_gap_mean` | Средний интервал между операциями (секунды) |
| `prof_gap_std` | Стандартное отклонение интервала (**#3 в importance, 3.16%**) |



---

# 3. ИТОГО: 121 ФИЧА ПО ГРУППАМ

```
ГРУППА                           КОЛ-ВО    GAIN %    ПРИМЕР ТОПОВОЙ
─────────────────────────────────────────────────────────────────────
Velocity (cnt_*, amt_sum_*)         22      ~30%     amt_ratio_7d_30d (2.61%)
Anomaly (anom_*)                    18      ~20%     anom_hourly_burst (6.41%)
Profile (prof_*)                    12      ~15%     prof_gap_std (3.16%)
Customer profile (cust_*)           14      ~15%     dormancy_days (2.22%)
Presence/Device flags               10      ~5%      is_no_screen
Temporal                             6      ~8%      hour, month
Amount-derived                       8      ~4%      amt_deviation_30d
Categorical (mcc, event, channel)   10      ~12%     mcc_code (3.52%)
Interaction                          3      ~1%      voip_x_logamt
Security                             3      ~0.5%    security_risk_score
Session                              3      ~1%      is_first_in_session
Screen                               2      ~0.5%    screen_w
─────────────────────────────────────────────────────────────────────
ИТОГО                              121     100%
```

---

*Файл создан: 2026-03-18*
*Модель: v14-C (LB=0.1010), pipeline_v9.py + pipeline_v14.py*
