import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
import json

from backend.app.create_app import create_app
from core.database.engine import get_db
from core.database.models.main_models import Coin, Timeseries, DataTimeseries
from core.database.models.ML_models import Agent
from core.database.models.ML_models import StatisticAgent


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_coins(db_session):
    """Создание тестовых монет"""
    coins = []
    for i in range(3):
        coin = Coin(
            name=f"TEST_COIN_{i}",
            price_now=100.0 + i * 10,
            max_price_now=110.0 + i * 10,
            min_price_now=90.0 + i * 10,
            open_price_now=100.0 + i * 10,
            volume_now=1000000.0 + i * 100000
        )
        db_session.add(coin)
        db_session.flush()
        coins.append(coin)
    
    db_session.commit()
    return coins


@pytest.fixture
def sample_timeseries(db_session, sample_coins):
    """Создание тестовых временных рядов"""
    timeseries = []
    for coin in sample_coins:
        ts = Timeseries(
            coin_id=coin.id,
            timestamp="5m",
            path_dataset=f"{coin.name}_5m_20250101"
        )
        db_session.add(ts)
        db_session.flush()
        timeseries.append(ts)
    
    db_session.commit()
    return timeseries


@pytest.fixture
def sample_data(db_session, sample_timeseries):
    """Создание тестовых данных OHLCV"""
    data = []
    base_time = datetime(2025, 1, 1, 0, 0, 0)
    
    for ts in sample_timeseries:
        for i in range(100):  # 100 записей по 5 минут
            dt = base_time + timedelta(minutes=i * 5)
            record = DataTimeseries(
                timeseries_id=ts.id,
                datetime=dt,
                open=100.0 + i * 0.1,
                max=101.0 + i * 0.1,
                min=99.0 + i * 0.1,
                close=100.5 + i * 0.1,
                volume=1000000.0 + i * 1000
            )
            db_session.add(record)
            data.append(record)
    
    db_session.commit()
    return data


@pytest.fixture
def sample_agent(db_session):
    """Создание тестового агента"""
    agent = Agent(
        name="Test Agent",
        type="AgentPredTime",
        timeframe="5m",
        path_model="models/test_agent.pth",
        status="open",
        version="1.0.0"
    )
    db_session.add(agent)
    db_session.flush()
    db_session.commit()
    return agent


@pytest.fixture
def admin_headers():
    """Заголовки для администратора"""
    return {
        "Authorization": "Bearer admin_test_token",
        "Content-Type": "application/json"
    }


class TestDataManagementEndpoints:
    """Тесты для endpoints управления данными"""
    
    def test_get_data_stats(self, client, sample_data, sample_coins, admin_headers):
        """Тест получения статистики данных"""
        coin_ids = [coin.id for coin in sample_coins]
        
        response = client.get(
            "/api_db_agent/data/stats",
            params={
                "coins": ",".join(map(str, coin_ids)),
                "timeframe": "5m",
                "start_date": "2025-01-01",
                "end_date": "2025-01-02"
            },
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_records" in data
        assert "coins_count" in data
        assert "completeness" in data
        assert "missing_values" in data
        assert "duplicates" in data
        assert data["total_records"] > 0
        assert data["coins_count"] == len(sample_coins)
    
    def test_export_data_csv(self, client, sample_data, sample_coins, admin_headers):
        """Тест экспорта данных в CSV"""
        coin_ids = [coin.id for coin in sample_coins]
        
        response = client.get(
            "/api_db_agent/data/export",
            params={
                "coins": ",".join(map(str, coin_ids)),
                "timeframe": "5m",
                "start_date": "2025-01-01",
                "end_date": "2025-01-02",
                "format": "csv"
            },
            headers=admin_headers
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv"
        assert "Content-Disposition" in response.headers
    
    def test_export_data_json(self, client, sample_data, sample_coins, admin_headers):
        """Тест экспорта данных в JSON"""
        coin_ids = [coin.id for coin in sample_coins]
        
        response = client.get(
            "/api_db_agent/data/export",
            params={
                "coins": ",".join(map(str, coin_ids)),
                "timeframe": "5m",
                "start_date": "2025-01-01",
                "end_date": "2025-01-02",
                "format": "json"
            },
            headers=admin_headers
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "datetime" in data[0]
        assert "coin" in data[0]
        assert "open" in data[0]
    
    def test_import_data_csv(self, client, admin_headers):
        """Тест импорта данных из CSV"""
        csv_content = """datetime,coin,open,high,low,close,volume
2025-01-01T00:00:00,BTC,50000,50100,49900,50050,1000000
2025-01-01T00:05:00,BTC,50050,50200,50000,50150,1100000"""
        
        files = {"file": ("test_data.csv", csv_content, "text/csv")}
        data = {"timeframe": "5m"}
        
        response = client.post(
            "/api_db_agent/data/import",
            files=files,
            data=data,
            headers={"Authorization": "Bearer admin_test_token"}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "imported_records" in result
        assert "skipped_records" in result
        assert "errors" in result


class TestModelTestingEndpoints:
    """Тесты для endpoints тестирования моделей"""
    
    def test_test_model(self, client, sample_agent, sample_coins, admin_headers):
        """Тест запуска тестирования модели"""
        coin_ids = [coin.id for coin in sample_coins]
        
        payload = {
            "model_id": sample_agent.id,
            "coins": coin_ids,
            "timeframe": "5m",
            "start_date": "2025-01-01",
            "end_date": "2025-01-02",
            "metrics": ["accuracy", "precision", "recall"]
        }
        
        response = client.post(
            "/api_db_agent/test_model",
            json=payload,
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "task_id" in data
        assert data["status"] == "started"
    
    def test_get_model_metrics(self, client, sample_agent, admin_headers, db_session):
        """Тест получения метрик модели"""
        # Создаем тестовую статистику
        stats = StatisticAgent(
            agent_id=sample_agent.id,
            type="test",
            loss=0.15,
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1score=0.85
        )
        db_session.add(stats)
        db_session.commit()
        
        response = client.get(
            f"/api_db_agent/models/{sample_agent.id}/metrics",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_id" in data
        assert "metrics" in data
        assert data["model_id"] == sample_agent.id
        assert "accuracy" in data["metrics"]
        assert data["metrics"]["accuracy"] == 0.85


class TestModelsListEndpoint:
    """Тесты для endpoint списка моделей"""
    
    def test_list_models(self, client, sample_agent, admin_headers):
        """Тест получения списка моделей"""
        response = client.get(
            "/api_db_agent/models",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "models" in data
        assert "count" in data
        assert data["status"] == "success"
        assert data["count"] > 0
        
        # Проверяем, что наша тестовая модель в списке
        model_names = [model["name"] for model in data["models"]]
        assert sample_agent.name in model_names
    
    def test_list_models_with_filters(self, client, sample_agent, admin_headers):
        """Тест получения списка моделей с фильтрами"""
        response = client.get(
            "/api_db_agent/models",
            params={
                "type": "AgentPredTime",
                "status": "open",
                "limit": 10
            },
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert len(data["models"]) <= 10
        
        # Проверяем фильтры
        for model in data["models"]:
            assert model["type"] == "AgentPredTime"
            assert model["status"] == "open"


class TestTaskStatusEndpoint:
    """Тесты для endpoint статуса задач"""
    
    def test_get_task_status(self, client, admin_headers):
        """Тест получения статуса задачи"""
        # Создаем тестовую задачу
        task_id = "test_task_123"
        
        response = client.get(
            f"/api_db_agent/task_status/{task_id}",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "task_id" in data
        assert "state" in data
        assert "ready" in data
        assert data["task_id"] == task_id


class TestNewsBackgroundEndpoints:
    """Тесты для endpoints новостного фона"""
    
    def test_recalc_news_background(self, client, sample_coins, admin_headers):
        """Тест пересчета новостного фона"""
        coin_ids = [coin.id for coin in sample_coins]
        
        response = client.post(
            "/api_db_agent/news/recalc_background",
            params={
                "coins": ",".join(map(str, coin_ids)),
                "window_hours": 24,
                "decay_factor": 0.95,
                "force_recalculate": False
            },
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "task_id" in data
        assert data["status"] == "started"
    
    def test_get_news_background(self, client, sample_coins, admin_headers):
        """Тест получения новостного фона"""
        coin_id = sample_coins[0].id
        
        response = client.get(
            f"/api_db_agent/news/background/{coin_id}",
            params={
                "start_time": "2025-01-01T00:00:00",
                "end_time": "2025-01-02T00:00:00",
                "limit": 100
            },
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "data" in data


class TestModelPromotionEndpoint:
    """Тесты для endpoint продвижения моделей"""
    
    def test_promote_agent(self, client, sample_agent, admin_headers):
        """Тест продвижения агента"""
        response = client.post(
            f"/api_db_agent/agents/{sample_agent.id}/promote",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "agent_id" in data
        assert data["agent_id"] == sample_agent.id


class TestErrorHandling:
    """Тесты обработки ошибок"""
    
    def test_invalid_model_id(self, client, admin_headers):
        """Тест с несуществующим ID модели"""
        response = client.get(
            "/api_db_agent/models/999999/metrics",
            headers=admin_headers
        )
        
        assert response.status_code == 404
    
    def test_invalid_date_format(self, client, sample_coins, admin_headers):
        """Тест с неверным форматом даты"""
        coin_ids = [coin.id for coin in sample_coins]
        
        response = client.get(
            "/api_db_agent/data/stats",
            params={
                "coins": ",".join(map(str, coin_ids)),
                "start_date": "invalid-date",
                "end_date": "2025-01-02"
            },
            headers=admin_headers
        )
        
        assert response.status_code == 400
    
    def test_missing_required_params(self, client, admin_headers):
        """Тест с отсутствующими обязательными параметрами"""
        response = client.post(
            "/api_db_agent/test_model",
            json={},
            headers=admin_headers
        )
        
        assert response.status_code == 400
    
    def test_unauthorized_access(self, client):
        """Тест неавторизованного доступа"""
        response = client.get("/api_db_agent/models")
        assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
