import pytest
from datetime import datetime, timedelta
from src.core.database.models.main_models import Coin, Timeseries, DataTimeseries
from src.core.database.models.ML_models import Agent
from src.core.database.models.Strategy_models import StatisticAgent


def test_create_sample_data():
    """Тест создания тестовых данных"""
    # Создаем тестовую монету
    coin = Coin(
        name="TEST_BTC",
        price_now=50000.0,
        max_price_now=51000.0,
        min_price_now=49000.0,
        open_price_now=50000.0,
        volume_now=1000000.0
    )
    
    assert coin.name == "TEST_BTC"
    assert coin.price_now == 50000.0
    
    # Создаем временной ряд
    timeseries = Timeseries(
        coin_id=1,  # Предполагаем, что coin.id = 1
        timestamp="5m",
        path_dataset="TEST_BTC_5m_20250101"
    )
    
    assert timeseries.timestamp == "5m"
    assert "TEST_BTC" in timeseries.path_dataset
    
    # Создаем данные OHLCV
    data_point = DataTimeseries(
        timeseries_id=1,  # Предполагаем, что timeseries.id = 1
        datetime=datetime(2025, 1, 1, 0, 0, 0),
        open=50000.0,
        max=50100.0,
        min=49900.0,
        close=50050.0,
        volume=1000000.0
    )
    
    assert data_point.open == 50000.0
    assert data_point.close == 50050.0
    assert data_point.volume == 1000000.0


def test_create_agent():
    """Тест создания агента"""
    agent = Agent(
        name="Test PredTime Agent",
        type="AgentPredTime",
        timeframe="5m",
        path_model="models/test_pred_time.pth",
        status="open",
        version="1.0.0"
    )
    
    assert agent.name == "Test PredTime Agent"
    assert agent.type == "AgentPredTime"
    assert agent.timeframe == "5m"
    assert agent.status == "open"


def test_create_statistics():
    """Тест создания статистики"""
    stats = StatisticAgent(
        agent_id=1,
        type="test",
        loss=0.15,
        accuracy=0.85,
        precision=0.82,
        recall=0.88,
        f1score=0.85
    )
    
    assert stats.agent_id == 1
    assert stats.type == "test"
    assert stats.loss == 0.15
    assert stats.accuracy == 0.85
    assert stats.f1score == 0.85


def test_data_validation():
    """Тест валидации данных"""
    # Проверяем, что цены не могут быть отрицательными
    with pytest.raises(ValueError):
        Coin(
            name="TEST_COIN",
            price_now=-100.0,  # Отрицательная цена
            max_price_now=110.0,
            min_price_now=90.0,
            open_price_now=100.0,
            volume_now=1000000.0
        )


def test_timeframe_validation():
    """Тест валидации таймфреймов"""
    valid_timeframes = ["5m", "15m", "30m", "1h", "4h", "1d"]
    
    for tf in valid_timeframes:
        timeseries = Timeseries(
            coin_id=1,
            timestamp=tf,
            path_dataset=f"TEST_{tf}_20250101"
        )
        assert timeseries.timestamp in valid_timeframes


def test_agent_types():
    """Тест типов агентов"""
    agent_types = [
        "AgentNews",
        "AgentPredTime", 
        "AgentTradeTime",
        "AgentRisk",
        "AgentTradeAggregator"
    ]
    
    for agent_type in agent_types:
        agent = Agent(
            name=f"Test {agent_type}",
            type=agent_type,
            timeframe="5m",
            path_model=f"models/test_{agent_type.lower()}.pth",
            status="open",
            version="1.0.0"
        )
        assert agent.type in agent_types


def test_data_consistency():
    """Тест консистентности данных OHLCV"""
    # High должен быть >= Open, Close
    # Low должен быть <= Open, Close
    
    data_point = DataTimeseries(
        timeseries_id=1,
        datetime=datetime(2025, 1, 1, 0, 0, 0),
        open=100.0,
        max=110.0,  # High > Open
        min=90.0,   # Low < Open
        close=105.0,
        volume=1000000.0
    )
    
    assert data_point.max >= data_point.open
    assert data_point.max >= data_point.close
    assert data_point.min <= data_point.open
    assert data_point.min <= data_point.close
    assert data_point.volume >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
