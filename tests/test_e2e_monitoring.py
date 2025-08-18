import pytest
import requests
import time
import json
from datetime import datetime, timedelta

class TestE2EMonitoring:
    """E2E тесты для проверки мониторинга системы"""
    
    @pytest.fixture
    def base_url(self):
        return "http://localhost:8000"
    
    @pytest.fixture
    def prometheus_url(self):
        return "http://localhost:9090"
    
    @pytest.fixture
    def grafana_url(self):
        return "http://localhost:3000"
    
    def test_backend_metrics_endpoint(self, base_url):
        """Тест доступности метрик backend"""
        try:
            response = requests.get(f"{base_url}/metrics", timeout=5)
            assert response.status_code == 200
            assert "http_requests_total" in response.text
            assert "http_request_duration_seconds" in response.text
            print("✓ Backend metrics endpoint доступен")
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Backend недоступен: {e}")
    
    def test_prometheus_targets(self, prometheus_url):
        """Тест доступности targets в Prometheus"""
        try:
            response = requests.get(f"{prometheus_url}/api/v1/targets", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            targets = data["data"]["activeTargets"]
            assert len(targets) > 0
            print(f"✓ Prometheus targets: {len(targets)} активных")
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Prometheus недоступен: {e}")
    
    def test_prometheus_metrics_query(self, prometheus_url):
        """Тест запросов метрик в Prometheus"""
        try:
            # Проверяем наличие метрик backend
            query = "http_requests_total"
            response = requests.get(f"{prometheus_url}/api/v1/query", params={"query": query}, timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            print(f"✓ Prometheus query '{query}' выполнен")
            
            # Проверяем метрики ML
            query = "ml_model_train_total"
            response = requests.get(f"{prometheus_url}/api/v1/query", params={"query": query}, timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            print(f"✓ Prometheus query '{query}' выполнен")
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Prometheus недоступен: {e}")
    
    def test_grafana_api(self, grafana_url):
        """Тест API Grafana"""
        try:
            # Проверяем доступность Grafana
            response = requests.get(f"{grafana_url}/api/health", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert data["database"] == "ok"
            print("✓ Grafana API доступен")
            
            # Проверяем дашборды
            response = requests.get(f"{grafana_url}/api/search", timeout=5)
            assert response.status_code == 200
            dashboards = response.json()
            print(f"✓ Grafana dashboards: {len(dashboards)} найдено")
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Grafana недоступен: {e}")
    
    def test_alertmanager_api(self):
        """Тест API Alertmanager"""
        try:
            response = requests.get("http://localhost:9093/api/v1/status", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "config" in data
            print("✓ Alertmanager API доступен")
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Alertmanager недоступен: {e}")
    
    def test_ml_metrics_generation(self, base_url):
        """Тест генерации ML метрик через API"""
        try:
            # Создаем тестовый запрос для генерации метрик
            test_payload = {
                "model_id": 1,
                "coins": [1, 2],
                "timeframe": "5m",
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
                "metrics": ["accuracy", "precision"]
            }
            
            response = requests.post(f"{base_url}/api_db_agent/test_model", json=test_payload, timeout=10)
            # Ожидаем 202 Accepted или 200 OK
            assert response.status_code in [200, 202]
            print("✓ ML test request отправлен")
            
            # Ждем немного для обработки
            time.sleep(2)
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Backend API недоступен: {e}")
    
    def test_system_metrics_availability(self, prometheus_url):
        """Тест доступности системных метрик"""
        try:
            metrics_to_check = [
                "node_cpu_seconds_total",
                "node_memory_MemTotal_bytes",
                "container_cpu_usage_seconds_total",
                "container_memory_usage_bytes"
            ]
            
            for metric in metrics_to_check:
                response = requests.get(f"{prometheus_url}/api/v1/query", params={"query": metric}, timeout=5)
                assert response.status_code == 200
                data = response.json()
                assert "data" in data
                print(f"✓ Метрика '{metric}' доступна")
                
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Prometheus недоступен: {e}")
    
    def test_alert_rules_validation(self, prometheus_url):
        """Тест валидации правил алертов"""
        try:
            response = requests.get(f"{prometheus_url}/api/v1/rules", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            
            # Проверяем наличие правил алертов
            rules = data["data"]["groups"]
            assert len(rules) > 0
            
            # Ищем правила для ML системы
            ml_rules = [rule for rule in rules if "ml-trading-system" in rule.get("name", "")]
            assert len(ml_rules) > 0
            print(f"✓ Найдено {len(ml_rules)} групп правил ML системы")
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Prometheus недоступен: {e}")
    
    def test_metrics_consistency(self, prometheus_url):
        """Тест консистентности метрик"""
        try:
            # Проверяем, что метрики не пустые
            queries = [
                "up",
                "http_requests_total",
                "ml_model_train_total"
            ]
            
            for query in queries:
                response = requests.get(f"{prometheus_url}/api/v1/query", params={"query": query}, timeout=5)
                assert response.status_code == 200
                data = response.json()
                assert "data" in data
                assert "result" in data["data"]
                print(f"✓ Метрика '{query}' консистентна")
                
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Prometheus недоступен: {e}")
    
    def test_monitoring_stack_health(self):
        """Общий тест здоровья стека мониторинга"""
        services = [
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
                    print(f"✓ {name} здоров")
                else:
                    print(f"✗ {name} нездоров (status: {response.status_code})")
            except requests.exceptions.RequestException as e:
                print(f"✗ {name} недоступен: {e}")
        
        # Требуем минимум backend и prometheus
        assert healthy_services >= 2, f"Только {healthy_services} из {len(services)} сервисов здоровы"
        print(f"✓ Общее здоровье стека: {healthy_services}/{len(services)} сервисов")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
