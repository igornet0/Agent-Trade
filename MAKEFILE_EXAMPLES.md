# Примеры использования Makefile

## 🚀 Быстрый старт

### Первый запуск проекта
```bash
# 1. Проверить доступные команды
make help

# 2. Проверить окружение
make preflight

# 3. Быстрый запуск всего проекта
make quick-start
```

### Только для разработки (без frontend)
```bash
# Запуск только базы данных и RabbitMQ
make quick-dev

# Запуск backend сервисов
make up-backend

# Проверка статуса
make status
```

## 🔧 Разработка

### Типичный день разработчика

```bash
# Утро: запуск среды разработки
make quick-dev
make up-backend

# В течение дня: просмотр логов
make logs-api
make logs-worker

# Вечер: остановка
make down
```

### Отладка проблем

```bash
# Проверка здоровья сервисов
make health

# Просмотр логов конкретного сервиса
make logs-api
make logs-worker

# Доступ к контейнеру для отладки
make shell-api

# Перезапуск проблемного сервиса
make restart
```

### Работа с базой данных

```bash
# Запуск миграций
make db-migrate

# Создание резервной копии
make db-backup

# Подключение к базе данных
make shell-db

# Сброс базы данных (⚠️ осторожно!)
make db-reset
```

## 🏭 Продакшн

### Деплой в продакшн

```bash
# Полный деплой с проверками
make deploy

# Проверка статуса после деплоя
make status
make health

# Создание резервной копии перед обновлением
make backup
```

### Мониторинг продакшн

```bash
# Проверка здоровья сервисов
make health

# Мониторинг ресурсов
make monitor

# Просмотр логов
make logs
```

### Обновление в продакшн

```bash
# Остановка сервисов
make down

# Сборка новых образов
make build

# Запуск обновленных сервисов
make up

# Проверка
make health
```

## 🧪 Тестирование

### Тестирование новой функциональности

```bash
# Запуск тестовой среды
make quick-dev
make up-backend

# Тестирование API
curl http://localhost:8000/health

# Просмотр логов тестов
make logs-api

# Очистка после тестов
make down
```

### Тестирование с новой базой данных

```bash
# Сброс базы данных
make db-reset

# Запуск миграций
make db-migrate

# Запуск сервисов
make up-backend

# Тестирование
make health
```

## 🔍 Диагностика

### Диагностика проблем с сетью

```bash
# Проверка статуса всех сервисов
make status

# Проверка здоровья
make health

# Проверка портов
netstat -tulpn | grep :8000
netstat -tulpn | grep :5173

# Перезапуск всех сервисов
make restart
```

### Диагностика проблем с базой данных

```bash
# Проверка подключения к БД
make shell-db

# Просмотр логов БД
make logs-db

# Проверка миграций
make db-migrate
```

### Диагностика проблем с RabbitMQ

```bash
# Проверка RabbitMQ
curl http://localhost:15672

# Просмотр логов RabbitMQ
docker-compose logs rabbitmq

# Перезапуск RabbitMQ
docker-compose restart rabbitmq
```

## 🛠 Обслуживание

### Регулярное обслуживание

```bash
# Создание резервной копии
make backup

# Очистка неиспользуемых ресурсов
docker system prune -f

# Проверка здоровья
make health
```

### Полная очистка

```bash
# Остановка и удаление всех контейнеров
make clean

# Удаление всех образов
make clean-images

# Очистка Docker системы
docker system prune -a
```

## 📊 Мониторинг

### Мониторинг производительности

```bash
# Мониторинг ресурсов в реальном времени
make monitor

# Проверка использования диска
docker system df

# Статус сервисов
make status
```

### Мониторинг логов

```bash
# Логи всех сервисов
make logs

# Логи конкретного сервиса
make logs-api
make logs-worker
make logs-frontend
```

## 🔄 CI/CD

### Автоматический деплой

```bash
#!/bin/bash
# deploy.sh

set -e

echo "🚀 Starting deployment..."

# Pre-flight checks
make preflight

# Build new images
make build

# Stop old containers
make down

# Start new containers
make up

# Wait for services to be ready
sleep 30

# Health check
make health

echo "✅ Deployment completed successfully!"
```

### Rollback скрипт

```bash
#!/bin/bash
# rollback.sh

set -e

echo "🔄 Starting rollback..."

# Stop current containers
make down

# Restore from backup
make restore BACKUP_FILE=backup_20241201_120000.sql

# Start services
make up

# Health check
make health

echo "✅ Rollback completed successfully!"
```

## 🎯 Специфичные сценарии

### Разработка ML-моделей

```bash
# Запуск только базы данных для работы с данными
make up-db

# Запуск backend для обучения моделей
make up-backend

# Просмотр логов обучения
make logs-worker

# Остановка после обучения
make down
```

### Тестирование API

```bash
# Запуск API сервиса
make up-backend

# Тестирование endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/agents

# Просмотр логов API
make logs-api
```

### Разработка Frontend

```bash
# Запуск только базы данных и backend
make quick-dev
make up-backend

# Запуск frontend в отдельном терминале
make up-frontend

# Просмотр логов frontend
make logs-frontend
```

### Отладка Celery задач

```bash
# Запуск worker
make up-backend

# Просмотр логов worker
make logs-worker

# Доступ к worker контейнеру
make shell-worker

# Перезапуск worker
docker-compose restart worker
```

## 🔧 Продвинутые команды

### Комбинированные команды

```bash
# Пересборка и перезапуск
make build && make restart

# Проверка и деплой
make preflight && make deploy

# Очистка и перезапуск
make clean && make quick-start
```

### Условные команды

```bash
# Запуск с проверкой
make preflight && make up || make clean

# Деплой с резервной копией
make backup && make deploy

# Откат при ошибке
make deploy || (make down && make restore BACKUP_FILE=latest.sql)
```

### Автоматизация

```bash
# Автоматический перезапуск при изменениях
watch -n 30 'make health || make restart'

# Мониторинг с алертами
while true; do
    make health || echo "ALERT: Service health check failed!"
    sleep 60
done
```

## 📝 Полезные алиасы

Добавьте в ваш `.bashrc` или `.zshrc`:

```bash
# Алиасы для Makefile команд
alias mh='make help'
alias mu='make up'
alias md='make down'
alias mr='make restart'
alias ml='make logs'
alias ms='make status'
alias mh='make health'
alias mq='make quick-start'
alias mqd='make quick-dev'
alias mb='make build'
alias mde='make deploy'
alias mdb='make db-migrate'
alias mbb='make db-backup'
```

## 🚨 Troubleshooting

### Частые проблемы

1. **Порт уже занят:**
```bash
# Проверка занятых портов
netstat -tulpn | grep :8000
# Остановка процесса
sudo kill -9 <PID>
```

2. **Проблемы с правами доступа:**
```bash
# Исправление прав на Docker socket
sudo chmod 666 /var/run/docker.sock
```

3. **Недостаточно места на диске:**
```bash
# Очистка Docker
make clean
docker system prune -a
```

4. **Проблемы с сетью Docker:**
```bash
# Перезапуск Docker daemon
sudo systemctl restart docker
```

### Логи ошибок

```bash
# Просмотр всех логов с ошибками
make logs | grep -i error

# Просмотр логов конкретного сервиса с ошибками
make logs-api | grep -i error
```
