# Инструкции для настройки Claude на сервере

## 1. Настрой разрешения (один раз)
cd ~/PyPr/hak
claude config set allowedTools '["Write","Edit","Bash","Read","Glob","Grep"]'

## 2. Запусти Claude с задачей
cd ~/PyPr/hak
claude -p 'Прочитай CLAUDE.md, FINDINGS.md, pipeline_v3f.py и pipeline_v5.py. Создай pipeline_v6.py: 1) Стратегия v3f (labeled+green, НЕ labeled-only), 2) Добавь фичи из v5 (pos_cd_1, is_no_screen, is_no_timezone), 3) Добавь customer-level агрегаты из pretrain (средняя сумма, частота, уникальные MCC за pretrain), 4) Multi-seed averaging и rank blending. НЕ запускай, только создай файл.'

## 3. Или запусти интерактивно
cd ~/PyPr/hak
claude

Потом сам напиши ему задачу.
