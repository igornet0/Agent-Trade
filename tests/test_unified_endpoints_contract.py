import pytest
import importlib.util
from pathlib import Path


def _import_agent_schema():
    root = Path(__file__).resolve().parents[1]
    agent_schema_path = root / 'src' / 'backend' / 'app' / 'configuration' / 'schemas' / 'agent.py'
    spec = importlib.util.spec_from_file_location("agent_schema", str(agent_schema_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@pytest.mark.anyio
async def test_train_request_has_all_optional_configs():
    m = _import_agent_schema()
    req = m.TrainRequest(
        name="agg",
        type=m.AgentType.TRADE_AGGREGATOR,
        timeframe="5m",
        coins=[],
        features=[],
        train_data=m.TrainData(epochs=1, batch_size=1, learning_rate=0.001, weight_decay=0.0),
        news_config=m.NewsTrainConfig(),
        pred_time_config=m.PredTimeTrainConfig(),
        trade_time_config=m.TradeTimeTrainConfig(),
        risk_config=m.RiskTrainConfig(),
        trade_aggregator_config=m.TradeAggregatorConfig(),
    )
    assert req.trade_aggregator_config is not None

