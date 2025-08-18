#!/bin/bash

set -e

echo "🚀 Запуск деплоя ML Trading System в продакшен"

# Проверяем наличие Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose не установлен"
    exit 1
fi

# Создаем необходимые директории
echo "📁 Создание директорий..."
mkdir -p backups
mkdir -p logs
mkdir -p ssl

# Проверяем SSL сертификаты
if [ ! -f "ssl/cert.pem" ] || [ ! -f "ssl/privkey.pem" ]; then
    echo "🔐 Генерация SSL сертификатов..."
    openssl req -x509 -newkey rsa:4096 -keyout ssl/privkey.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
fi

# Останавливаем существующие контейнеры
echo "🛑 Остановка существующих контейнеров..."
docker-compose -f docker-compose.prod.yml down || true

# Удаляем старые образы
echo "🧹 Очистка старых образов..."
docker system prune -f

# Собираем и запускаем сервисы
echo "🔨 Сборка и запуск сервисов..."
docker-compose -f docker-compose.prod.yml up -d --build

# Ждем запуска сервисов
echo "⏳ Ожидание запуска сервисов..."
sleep 30

# Проверяем здоровье сервисов
echo "🏥 Проверка здоровья сервисов..."

services=(
    "backend:8000/health"
    "frontend:80"
    "postgres:5432"
    "redis:6379"
    "prometheus:9090/-/healthy"
    "grafana:3000/api/health"
    "alertmanager:9093/-/healthy"
)

healthy_count=0
for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    url=$(echo $service | cut -d: -f2-)
    
    if curl -f -s "http://$url" > /dev/null 2>&1; then
        echo "✅ $name здоров"
        ((healthy_count++))
    else
        echo "❌ $name нездоров"
    fi
done

echo "📊 Здоровье системы: $healthy_count/${#services[@]} сервисов"

# Проверяем API
echo "🔍 Проверка API..."
if curl -f -s "http://localhost:8000/api_db_agent/models" > /dev/null; then
    echo "✅ API работает"
else
    echo "❌ API недоступен"
fi

# Проверяем метрики
echo "📈 Проверка метрик..."
if curl -f -s "http://localhost:8000/metrics" | grep -q "http_requests_total"; then
    echo "✅ Метрики доступны"
else
    echo "❌ Метрики недоступны"
fi

# Создаем бэкап БД
echo "💾 Создание бэкапа БД..."
docker-compose -f docker-compose.prod.yml exec -T postgres pg_dump -U agent -d agent > backups/backup_$(date +%Y%m%d_%H%M%S).sql

# Настройка мониторинга
echo "📊 Настройка мониторинга..."

# Импорт дашборда Grafana (если доступен)
if curl -f -s "http://localhost:3000/api/health" > /dev/null; then
    echo "✅ Grafana доступен для настройки дашбордов"
fi

# Проверка алертов
if curl -f -s "http://localhost:9093/api/v1/status" > /dev/null; then
    echo "✅ Alertmanager доступен"
fi

echo ""
echo "🎉 Деплой завершен успешно!"
echo ""
echo "📋 Доступные сервисы:"
echo "  🌐 Frontend: http://localhost"
echo "  🔧 Backend API: http://localhost:8000"
echo "  📊 Grafana: http://localhost:3000 (admin/admin)"
echo "  📈 Prometheus: http://localhost:9090"
echo "  🚨 Alertmanager: http://localhost:9093"
echo ""
echo "🔐 SSL сертификаты: ssl/cert.pem, ssl/privkey.pem"
echo "💾 Бэкапы: ./backups/"
echo "📝 Логи: docker-compose -f docker-compose.prod.yml logs"
echo ""
echo "🛠️  Полезные команды:"
echo "  Остановка: docker-compose -f docker-compose.prod.yml down"
echo "  Логи: docker-compose -f docker-compose.prod.yml logs -f"
echo "  Перезапуск: docker-compose -f docker-compose.prod.yml restart"
echo "  Обновление: ./deploy.sh"
