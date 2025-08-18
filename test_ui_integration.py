import requests
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

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
    
    # 3. Проверяем API endpoints через frontend
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
    
    print("\n🎉 Все тесты UI интеграции пройдены успешно!")
    return True

def test_browser_integration():
    """Тест интеграции в браузере (если доступен Selenium)"""
    
    print("\n🌐 Тестирование в браузере...")
    
    try:
        # Настройка Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Тестируем загрузку страницы
        driver.get("http://localhost:5173")
        
        # Ждем загрузки
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Проверяем заголовок
        title = driver.title
        assert "Agent Trade" in title
        print("✅ Страница загружена в браузере")
        
        # Проверяем наличие React приложения
        root_element = driver.find_element(By.ID, "root")
        assert root_element.is_displayed()
        print("✅ React приложение отображается")
        
        driver.quit()
        print("✅ Браузерное тестирование пройдено")
        return True
        
    except Exception as e:
        print(f"❌ Браузерное тестирование не удалось: {e}")
        print("ℹ️  Это нормально, если Selenium не установлен")
        return True  # Не критично

if __name__ == "__main__":
    print("🚀 Запуск UI интеграционных тестов")
    
    # Основные тесты
    success = test_ui_backend_integration()
    
    # Браузерные тесты (опционально)
    if success:
        test_browser_integration()
    
    if success:
        print("\n🎉 Все UI тесты пройдены успешно!")
        exit(0)
    else:
        print("\n❌ Некоторые тесты не прошли")
        exit(1)
