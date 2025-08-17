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
async def test_train_request_minimal_payload():
    m = _import_agent_schema()
    req = m.TrainRequest(
        name="test_pred_time",
        type=m.AgentType.PREDTIME,
        timeframe="5m",
        coins=[1, 2],
        features=[],
        train_data=m.TrainData(epochs=1, batch_size=32, learning_rate=0.001, weight_decay=0.0, extra_config={"warmup": 10}),
    )
    assert req.name == "test_pred_time"
    assert req.type == m.AgentType.PREDTIME
    assert req.coins == [1, 2]


@pytest.mark.anyio
async def test_train_request_with_pred_config():
    m = _import_agent_schema()
    cfg = m.PredTimeTrainConfig(model_name="Informer", seq_len=128, pred_len=24, indicators=["SMA", "MACD"], use_news_background=True)
    req = m.TrainRequest(
        name="test_pred_cfg",
        type=m.AgentType.PREDTIME,
        timeframe="5m",
        coins=[],
        features=[],
        train_data=m.TrainData(epochs=1, batch_size=16, learning_rate=0.0005, weight_decay=0.0),
        pred_time_config=cfg,
    )
    assert req.pred_time_config is not None
    assert req.pred_time_config.model_name == "Informer"


@pytest.mark.anyio
async def test_train_request_with_news_config_defaults():
    m = _import_agent_schema()
    cfg = m.NewsTrainConfig()
    assert isinstance(cfg.sources, list)
    assert cfg.nlp_model in ("bert", "finbert", "finbert") or isinstance(cfg.nlp_model, str)
    # Ensure TrainData.extra_config exists and accepts dict
    td = m.TrainData(epochs=1, batch_size=1, learning_rate=0.001, weight_decay=0.0, extra_config={"note": "ok"})
    assert td.extra_config == {"note": "ok"}

