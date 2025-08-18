# Docker Management with Makefile

## Обзор

Этот Makefile предоставляет удобные команды для управления Docker-контейнерами проекта Agent Trade. Все команды используют файл конфигурации `settings/prod.env`.

## Быстрый старт

### 1. Показать доступные команды
```bash
make help
```

### 2. Быстрый запуск всего проекта
```bash
make quick-start
```

### 3. Только для разработки (база данных + RabbitMQ)
```bash
make quick-dev
```

## Основные команды

### Разработка
- `make dev` - Запуск среды разработки (только БД и RabbitMQ)
- `make dev-full` - Запуск полной среды разработки (все сервисы)
- `make quick-dev` - Быстрый запуск для разработки

### Продакшн
- `make build` - Сборка всех Docker образов
- `make build-no-cache` - Сборка без кэша
- `make up` - Запуск продакшн окружения
- `make down` - Остановка всех контейнеров
- `make restart` - Перезапуск всех контейнеров
- `make deploy` - Полный деплой (проверки + сборка + запуск)

### Управление сервисами
- `make up-db` - Запуск только базы данных и RabbitMQ
- `make up-backend` - Запуск backend сервисов (API + Worker)
- `make up-frontend` - Запуск frontend сервиса

## Логи и мониторинг

### Просмотр логов
- `make logs` - Логи всех сервисов
- `make logs-api` - Логи API
- `make logs-worker` - Логи Worker
- `make logs-frontend` - Логи Frontend
- `make logs-db` - Логи базы данных

### Мониторинг
- `make status` - Статус всех сервисов
- `make health` - Проверка здоровья сервисов
- `make monitor` - Мониторинг использования ресурсов
- `make ps` - Список запущенных контейнеров

## База данных

### Миграции и управление
- `make db-migrate` - Запуск миграций базы данных
- `make db-backup` - Создание резервной копии БД
- `make db-reset` - Сброс базы данных (⚠️ удаляет все данные!)

### Восстановление
- `make backup` - Полная резервная копия (БД + volumes)
- `make restore BACKUP_FILE=filename` - Восстановление из резервной копии

## Разработка

### Доступ к контейнерам
- `make shell-api` - Shell в API контейнере
- `make shell-worker` - Shell в Worker контейнере
- `make shell-db` - Подключение к базе данных

### Очистка
- `make clean` - Удаление всех контейнеров, сетей и volumes
- `make clean-images` - Удаление всех образов проекта

## Проверки и диагностика

### Pre-flight проверки
- `make preflight` - Проверка окружения (Docker, Docker Compose)
- `make env-check` - Проверка наличия файла конфигурации

### Диагностика
- `make health` - Проверка здоровья всех сервисов
- `make status` - Статус сервисов

## Примеры использования

### Типичный workflow разработки

1. **Начало работы:**
```bash
make quick-dev
```

2. **Запуск backend для тестирования:**
```bash
make up-backend
```

3. **Просмотр логов:**
```bash
make logs-api
```

4. **Запуск миграций:**
```bash
make db-migrate
```

5. **Остановка:**
```bash
make down
```

### Продакшн деплой

1. **Полный деплой:**
```bash
make deploy
```

2. **Проверка статуса:**
```bash
make status
make health
```

3. **Создание резервной копии:**
```bash
make backup
```

### Отладка проблем

1. **Проверка логов:**
```bash
make logs-api
make logs-worker
```

2. **Проверка здоровья сервисов:**
```bash
make health
```

3. **Доступ к контейнеру для отладки:**
```bash
make shell-api
```

4. **Перезапуск проблемного сервиса:**
```bash
make restart
```

## Конфигурация

### Переменные окружения
Makefile использует файл `settings/prod.env` для конфигурации. Основные переменные:

- `DB__USER` - Пользователь базы данных
- `DB__PASSWORD` - Пароль базы данных
- `DB__DB_NAME` - Имя базы данных
- `RABBITMQ__HOST` - Хост RabbitMQ
- `RABBITMQ__USER` - Пользователь RabbitMQ
- `RABBITMQ__PASSWORD` - Пароль RabbitMQ

### Порты сервисов
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **Database**: localhost:5432
- **RabbitMQ**: localhost:5672
- **RabbitMQ Management**: http://localhost:15672

## Troubleshooting

### Проблемы с Docker
```bash
# Проверка Docker
docker --version
docker-compose --version

# Очистка Docker
make clean
docker system prune -a
```

### Проблемы с базой данных
```bash
# Проверка подключения
make shell-db

# Сброс базы данных (⚠️ осторожно!)
make db-reset
```

### Проблемы с RabbitMQ
```bash
# Проверка RabbitMQ
curl http://localhost:15672

# Перезапуск RabbitMQ
docker-compose restart rabbitmq
```

### Проблемы с сетью
```bash
# Проверка портов
netstat -tulpn | grep :8000
netstat -tulpn | grep :5173

# Перезапуск всех сервисов
make restart
```

## Безопасность

### Резервные копии
- Регулярно создавайте резервные копии: `make backup`
- Храните резервные копии в безопасном месте
- Тестируйте восстановление на тестовой среде

### Переменные окружения
- Никогда не коммитьте файлы с паролями в git
- Используйте разные файлы конфигурации для разных окружений
- Регулярно обновляйте пароли

### Мониторинг
- Регулярно проверяйте статус сервисов: `make health`
- Мониторьте использование ресурсов: `make monitor`
- Настройте алерты для критических сервисов

## Производительность

### Оптимизация сборки
```bash
# Сборка без кэша (если есть проблемы)
make build-no-cache

# Очистка неиспользуемых образов
docker image prune -a
```

### Мониторинг ресурсов
```bash
# Мониторинг в реальном времени
make monitor

# Проверка использования диска
docker system df
```

## Интеграция с CI/CD

### Пример GitHub Actions
```yaml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy with Makefile
        run: |
          make deploy
```

### Пример локального скрипта деплоя
```bash
#!/bin/bash
set -e

echo "Starting deployment..."
make preflight
make build
make up
make health

echo "Deployment completed successfully!"
```
