# 🚀 Быстрый запуск Agent Trade Project

## Разработка (Development)

```bash
# 1. Сборка и запуск всех сервисов
make dev-build
make dev-start

# 2. Создание админ пользователя
make create-admin

# 3. Проверка статуса
make status
```

**Доступные сервисы:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- RabbitMQ Management: http://localhost:15672 (agent/agent)
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

## Продакшен (Production)

```bash
# 1. Сборка и запуск продакшен среды
make prod-build
make prod-start

# 2. Проверка статуса
make status
```

**Доступные сервисы:**
- Frontend: http://localhost (через Nginx)
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Полезные команды

```bash
# Остановка
make dev-stop          # Остановка разработки
make prod-stop         # Остановка продакшена

# Логи
make dev-logs          # Логи разработки
make prod-logs         # Логи продакшена

# Статус и мониторинг
make status            # Статус сервисов
make health            # Проверка здоровья
make monitor           # Мониторинг ресурсов

# База данных
make db-migrate        # Миграции
make create-admin      # Создание админа
make backup            # Резервная копия

# Очистка
make clean             # Очистка всех контейнеров
make clean-images      # Удаление образов
```

## Админ пользователь
- **Логин:** admin
- **Пароль:** admin123
- **Email:** admin@agent-trade.com

## Подробная документация
См. [DOCKER_README.md](DOCKER_README.md) для полной документации.
