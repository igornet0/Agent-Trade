# Docker Setup для Agent Trade Project

Этот документ содержит инструкции по запуску проекта Agent Trade в Docker для разработки и продакшена.

## 🚀 Быстрый старт

### Разработка (Development)
```bash
# Сборка и запуск всех сервисов для разработки
make dev-build
make dev-start

# Или одной командой
make quick-start
```

### Продакшен (Production)
```bash
# Сборка и запуск всех сервисов для продакшена
make prod-build
make prod-start

# Или одной командой
make deploy
```

## 📋 Доступные команды

### Основные команды разработки
```bash
make dev-build          # Сборка Docker образов для разработки
make dev-start          # Запуск среды разработки
make dev-stop           # Остановка среды разработки
make dev-restart        # Перезапуск среды разработки
make dev-logs           # Просмотр логов разработки
```

### Основные команды продакшена
```bash
make prod-build         # Сборка Docker образов для продакшена
make prod-start         # Запуск продакшен среды
make prod-stop          # Остановка продакшен среды
make prod-restart       # Перезапуск продакшен среды
make prod-logs          # Просмотр логов продакшена
```

### Управление сервисами
```bash
make up-db              # Запуск только базы данных и RabbitMQ
make up-backend         # Запуск backend сервисов (API + Worker)
make up-frontend        # Запуск frontend сервиса
```

### Логи и мониторинг
```bash
make dev-logs-frontend  # Логи frontend
make dev-logs-backend   # Логи backend
make dev-logs-celery    # Логи celery worker
make logs-db            # Логи базы данных
make status             # Статус всех сервисов
make health             # Проверка здоровья сервисов
```

### База данных
```bash
make db-migrate         # Запуск миграций
make db-reset           # Сброс базы данных (ОСТОРОЖНО!)
make create-admin       # Создание админ пользователя
make check-admin        # Проверка админ пользователя
```

### Утилиты
```bash
make clean              # Очистка всех контейнеров и томов
make clean-images       # Удаление всех образов
make backup             # Создание резервной копии
make monitor            # Мониторинг ресурсов
```

## 🏗️ Архитектура проекта

### Development Environment (docker-compose.dev.yml)
- **Frontend**: Vite dev server на порту 5173
- **Backend**: FastAPI на порту 8000
- **Celery Worker**: Обработка фоновых задач
- **PostgreSQL**: База данных на порту 5432
- **Redis**: Кэш на порту 6379
- **RabbitMQ**: Message broker на порту 5672 (Management UI: 15672)
- **Prometheus**: Мониторинг на порту 9090
- **Grafana**: Дашборды на порту 3000
- **AlertManager**: Алерты на порту 9093
- **Node Exporter**: Системные метрики на порту 9100
- **cAdvisor**: Метрики контейнеров на порту 8080

### Production Environment (docker-compose.prod.yml)
- **Frontend**: Nginx с собранным React приложением на порту 80
- **Backend**: FastAPI на порту 8000
- **Celery Worker**: Обработка фоновых задач
- **PostgreSQL**: База данных на порту 5432
- **Redis**: Кэш на порту 6379
- **RabbitMQ**: Message broker на порту 5672 (Management UI: 15672)
- **Nginx**: Reverse proxy на порту 80/443
- **Prometheus**: Мониторинг на порту 9090
- **Grafana**: Дашборды на порту 3000
- **AlertManager**: Алерты на порту 9093
- **Backup Service**: Автоматическое резервное копирование

## 🔧 Конфигурация

### Environment Files
- `settings/dev.env` - Переменные окружения для разработки
- `settings/prod.env` - Переменные окружения для продакшена
- `settings/local.env` - Локальные переменные окружения

### Docker Compose Files
- `docker-compose.dev.yml` - Конфигурация для разработки
- `docker-compose.prod.yml` - Конфигурация для продакшена
- `docker-compose.yml` - Основная конфигурация (обновлена)

## 🚀 Пошаговые инструкции

### 1. Первоначальная настройка

```bash
# Клонирование репозитория
git clone <repository-url>
cd Agent-Trade

# Проверка окружения
make env-check

# Сборка и запуск для разработки
make dev-build
make dev-start

# Создание админ пользователя
make create-admin

# Проверка статуса
make status
```

### 2. Разработка

```bash
# Запуск среды разработки
make dev-start

# Просмотр логов
make dev-logs

# Открытие shell в контейнере
make shell-api
make shell-frontend

# Остановка
make dev-stop
```

### 3. Продакшен

```bash
# Сборка продакшен образов
make prod-build

# Запуск продакшен среды
make prod-start

# Проверка статуса
make status

# Мониторинг
make monitor
```

### 4. Обслуживание

```bash
# Резервное копирование
make backup

# Восстановление (указать файл)
make restore BACKUP_FILE=backup_20250101_120000.sql

# Очистка
make clean
```

## 🌐 Доступные сервисы

### Development
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **RabbitMQ Management**: http://localhost:15672 (agent/agent)
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093

### Production
- **Frontend**: http://localhost (через Nginx)
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **RabbitMQ Management**: http://localhost:15672 (agent/agent)
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093

## 🔍 Troubleshooting

### Проблемы с запуском
```bash
# Проверка здоровья сервисов
make health

# Просмотр логов
make dev-logs

# Проверка статуса контейнеров
make status

# Перезапуск сервисов
make dev-restart
```

### Проблемы с базой данных
```bash
# Проверка подключения к БД
make shell-db

# Сброс базы данных (ОСТОРОЖНО!)
make db-reset

# Создание таблиц
make create-tables-docker
```

### Проблемы с Docker
```bash
# Очистка Docker
make clean

# Удаление образов
make clean-images

# Мониторинг ресурсов
make monitor
```

## 📊 Мониторинг

### Prometheus
- Метрики приложения: http://localhost:9090
- Конфигурация: `settings/prometheus.yml`
- Правила алертов: `settings/alert_rules.yml`

### Grafana
- Дашборды: http://localhost:3000
- Логин: admin/admin
- Дашборды: `settings/grafana-dashboards/`

### AlertManager
- Алерты: http://localhost:9093
- Конфигурация: `settings/alertmanager.yml`

## 🔐 Безопасность

### Production
- Используйте сильные пароли в `settings/prod.env`
- Настройте SSL/TLS сертификаты
- Ограничьте доступ к портам мониторинга
- Регулярно обновляйте образы

### Development
- Используйте тестовые данные
- Не используйте production пароли
- Ограничьте доступ к портам

## 📝 Полезные команды

```bash
# Полная настройка проекта
make complete-setup

# Быстрый старт разработки
make quick-start

# Проверка всех сервисов
make check-services

# Создание админ пользователя интерактивно
make create-admin-interactive

# Просмотр ресурсов
make monitor
```

## 🆘 Поддержка

При возникновении проблем:
1. Проверьте логи: `make dev-logs`
2. Проверьте статус: `make status`
3. Проверьте здоровье: `make health`
4. Очистите окружение: `make clean`
5. Пересоберите образы: `make dev-build-no-cache`
