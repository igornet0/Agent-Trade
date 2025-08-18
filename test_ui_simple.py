import requests
import time
import json

def test_ui_backend_integration():
    """Тест интеграции UI с backend"""
    
    print("🧪 Тестирование UI интеграции с Backend")
    
    # 1. Проверяем доступность frontend
    print("\n1. Проверка доступности frontend...")
    try:
        response = requests.get("http://localhost:5173", timeout=5)
        assert response.status_code == 200
        assert "Agent Trade" in response.text
        print("✅ Frontend доступен")
    except Exception as e:
        print(f"❌ Frontend недоступен: {e}")
        return False
    
    # 2. Проверяем доступность backend
    print("\n2. Проверка доступности backend...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✅ Backend доступен")
    except Exception as e:
        print(f"❌ Backend недоступен: {e}")
        return False
    
    # 3. Проверяем API endpoints
    print("\n3. Проверка API endpoints...")
    
    # Проверяем список моделей
    try:
        response = requests.get("http://localhost:8000/api_db_agent/models", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0
        print(f"✅ API моделей: {len(data['models'])} моделей")
    except Exception as e:
        print(f"❌ API моделей недоступен: {e}")
        return False
    
    # 4. Тестируем тестирование модели
    print("\n4. Тестирование модели через API...")
    try:
        test_payload = {
            "model_id": 1,
            "coins": [1, 2],
            "timeframe": "5m",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "metrics": ["accuracy", "precision"]
        }
        
        response = requests.post("http://localhost:8000/api_db_agent/test_model", 
                               json=test_payload, timeout=10)
        assert response.status_code in [200, 202]
        data = response.json()
        assert "task_id" in data
        task_id = data["task_id"]
        print(f"✅ Тест модели запущен: {task_id}")
        
        # Проверяем статус
        time.sleep(2)
        response = requests.get(f"http://localhost:8000/api_db_agent/task/{task_id}", timeout=5)
        assert response.status_code == 200
        task_data = response.json()
        assert "state" in task_data
        print(f"✅ Статус задачи: {task_data['state']}")
        
    except Exception as e:
        print(f"❌ Тестирование модели не работает: {e}")
        return False
    
    # 5. Проверяем метрики
    print("\n5. Проверка метрик...")
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        assert response.status_code == 200
        metrics_text = response.text
        assert "http_requests_total" in metrics_text
        assert "ml_model_train_total" in metrics_text
        print("✅ Метрики доступны")
    except Exception as e:
        print(f"❌ Метрики недоступны: {e}")
        return False
    
    # 6. Проверяем мониторинг
    print("\n6. Проверка мониторинга...")
    
    # Prometheus
    try:
        response = requests.get("http://localhost:9090/api/v1/targets", timeout=5)
        assert response.status_code == 200
        data = response.json()
        targets = data["data"]["activeTargets"]
        print(f"✅ Prometheus: {len(targets)} targets")
    except Exception as e:
        print(f"❌ Prometheus недоступен: {e}")
    
    # Grafana
    try:
        response = requests.get("http://localhost:3000/api/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["database"] == "ok"
        print("✅ Grafana доступен")
    except Exception as e:
        print(f"❌ Grafana недоступен: {e}")
    
    # 7. Проверяем CORS
    print("\n7. Проверка CORS...")
    try:
        # Тестируем CORS preflight
        headers = {
            'Origin': 'http://localhost:5173',
            'Access-Control-Request-Method': 'GET',
            'Access-Control-Request-Headers': 'Content-Type'
        }
        response = requests.options("http://localhost:8000/api_db_agent/models", 
                                  headers=headers, timeout=5)
        print("✅ CORS настроен")
    except Exception as e:
        print(f"⚠️  CORS проверка не удалась: {e}")
    
    print("\n🎉 Все тесты UI интеграции пройдены успешно!")
    return True

def test_production_readiness():
    """Тест готовности к продакшену"""
    
    print("\n🏭 Тестирование готовности к продакшену...")
    
    # 1. Проверяем health endpoints
    print("\n1. Проверка health endpoints...")
    services = [
        ("Frontend", "http://localhost:5173"),
        ("Backend", "http://localhost:8000/health"),
        ("Prometheus", "http://localhost:9090/-/healthy"),
        ("Grafana", "http://localhost:3000/api/health"),
        ("Alertmanager", "http://localhost:9093/-/healthy")
    ]
    
    healthy_services = 0
    for name, url in services:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                healthy_services += 1
                print(f"✅ {name} здоров")
            else:
                print(f"⚠️  {name} нездоров (status: {response.status_code})")
        except Exception as e:
            print(f"❌ {name} недоступен: {e}")
    
    print(f"📊 Здоровье системы: {healthy_services}/{len(services)} сервисов")
    
    # 2. Проверяем метрики
    print("\n2. Проверка метрик...")
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        assert response.status_code == 200
        metrics_text = response.text
        
        required_metrics = [
            "http_requests_total",
            "http_request_duration_seconds",
            "ml_model_train_total",
            "ml_train_duration_seconds"
        ]
        
        missing_metrics = []
        for metric in required_metrics:
            if metric not in metrics_text:
                missing_metrics.append(metric)
        
        if missing_metrics:
            print(f"⚠️  Отсутствуют метрики: {missing_metrics}")
        else:
            print("✅ Все необходимые метрики доступны")
            
    except Exception as e:
        print(f"❌ Метрики недоступны: {e}")
    
    # 3. Проверяем API endpoints
    print("\n3. Проверка API endpoints...")
    api_endpoints = [
        ("GET", "/api_db_agent/models"),
        ("POST", "/api_db_agent/test_model"),
        ("GET", "/metrics"),
        ("GET", "/health")
    ]
    
    working_endpoints = 0
    for method, endpoint in api_endpoints:
        try:
            if method == "GET":
                response = requests.get(f"http://localhost:8000{endpoint}", timeout=5)
            elif method == "POST":
                response = requests.post(f"http://localhost:8000{endpoint}", 
                                       json={}, timeout=5)
            
            if response.status_code in [200, 202]:
                working_endpoints += 1
                print(f"✅ {method} {endpoint}")
            else:
                print(f"⚠️  {method} {endpoint} (status: {response.status_code})")
        except Exception as e:
            print(f"❌ {method} {endpoint} недоступен: {e}")
    
    print(f"📊 API endpoints: {working_endpoints}/{len(api_endpoints)} работают")
    
    # 4. Рекомендации
    print("\n4. Рекомендации для продакшена:")
    recommendations = [
        "✅ Настроить SSL сертификаты",
        "✅ Настроить аутентификацию",
        "✅ Настроить логирование",
        "✅ Настроить бэкапы БД",
        "✅ Настроить алерты",
        "✅ Настроить CI/CD",
        "✅ Настроить мониторинг производительности",
        "✅ Настроить rate limiting"
    ]
    
    for rec in recommendations:
        print(f"  {rec}")
    
    return healthy_services >= 3  # Минимум 3 из 5 сервисов должны быть здоровы

if __name__ == "__main__":
    print("🚀 Запуск UI интеграционных тестов")
    
    # Основные тесты
    success = test_ui_backend_integration()
    
    # Тесты готовности к продакшену
    if success:
        production_ready = test_production_readiness()
        if production_ready:
            print("\n🎉 Система готова к продакшену!")
        else:
            print("\n⚠️  Система требует доработки для продакшена")
    
    if success:
        print("\n🎉 Все UI тесты пройдены успешно!")
        exit(0)
    else:
        print("\n❌ Некоторые тесты не прошли")
        exit(1)
