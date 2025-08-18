import requests
import json
import time

def test_frontend_backend_integration():
    """Тест интеграции frontend с backend"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Тестирование интеграции Frontend-Backend")
    
    # 1. Проверка health endpoint
    print("\n1. Проверка health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✅ Health endpoint работает")
    except Exception as e:
        print(f"❌ Health endpoint не работает: {e}")
        return False
    
    # 2. Проверка метрик
    print("\n2. Проверка метрик...")
    try:
        response = requests.get(f"{base_url}/metrics", timeout=5)
        assert response.status_code == 200
        metrics_text = response.text
        assert "http_requests_total" in metrics_text
        assert "ml_model_train_total" in metrics_text
        print("✅ Метрики доступны")
    except Exception as e:
        print(f"❌ Метрики недоступны: {e}")
        return False
    
    # 3. Проверка списка моделей
    print("\n3. Проверка списка моделей...")
    try:
        response = requests.get(f"{base_url}/api_db_agent/models", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0
        print(f"✅ Список моделей: {len(data['models'])} моделей")
        
        # Проверяем структуру модели
        model = data["models"][0]
        required_fields = ["id", "name", "type", "status", "timeframe", "version"]
        for field in required_fields:
            assert field in model, f"Отсутствует поле {field}"
        print("✅ Структура модели корректна")
        
    except Exception as e:
        print(f"❌ Список моделей недоступен: {e}")
        return False
    
    # 4. Тестирование модели
    print("\n4. Тестирование модели...")
    try:
        test_payload = {
            "model_id": 1,
            "coins": [1, 2],
            "timeframe": "5m",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "metrics": ["accuracy", "precision"]
        }
        
        response = requests.post(f"{base_url}/api_db_agent/test_model", 
                               json=test_payload, timeout=10)
        assert response.status_code in [200, 202]
        data = response.json()
        assert "task_id" in data
        task_id = data["task_id"]
        print(f"✅ Тест модели запущен, task_id: {task_id}")
        
        # Проверяем статус задачи
        time.sleep(2)
        response = requests.get(f"{base_url}/api_db_agent/task/{task_id}", timeout=5)
        assert response.status_code == 200
        task_data = response.json()
        assert "state" in task_data
        assert "meta" in task_data
        print(f"✅ Статус задачи: {task_data['state']}")
        
        # Проверяем результаты теста
        if "test_results" in task_data.get("meta", {}):
            results = task_data["meta"]["test_results"]
            assert "accuracy" in results
            assert "charts" in results
            print("✅ Результаты теста получены")
        
    except Exception as e:
        print(f"❌ Тестирование модели не работает: {e}")
        return False
    
    # 5. Проверка мониторинга
    print("\n5. Проверка мониторинга...")
    
    # Prometheus
    try:
        response = requests.get("http://localhost:9090/api/v1/targets", timeout=5)
        assert response.status_code == 200
        data = response.json()
        targets = data["data"]["activeTargets"]
        print(f"✅ Prometheus targets: {len(targets)} активных")
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
    
    # Alertmanager
    try:
        response = requests.get("http://localhost:9093/api/v1/status", timeout=5)
        assert response.status_code == 200
        print("✅ Alertmanager доступен")
    except Exception as e:
        print(f"❌ Alertmanager недоступен: {e}")
    
    print("\n🎉 Все тесты интеграции пройдены успешно!")
    return True

if __name__ == "__main__":
    success = test_frontend_backend_integration()
    exit(0 if success else 1)
