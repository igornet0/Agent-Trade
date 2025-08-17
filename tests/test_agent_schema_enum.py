import pytest
import importlib.util
from pathlib import Path


@pytest.mark.anyio
async def test_agent_type_enum_extended_values():
    # Load schema module directly to avoid heavy package imports during test collection
    root = Path(__file__).resolve().parents[1]
    agent_schema_path = root / 'src' / 'backend' / 'app' / 'configuration' / 'schemas' / 'agent.py'
    spec = importlib.util.spec_from_file_location("agent_schema", str(agent_schema_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    AgentType = module.AgentType

    assert AgentType.PREDTIME.value == 'AgentPredTime'
    assert AgentType.TRADETIME.value == 'AgentTradeTime'
    assert AgentType.NEWS.value == 'AgentNews'
    assert AgentType.RISK.value == 'AgentRisk'
    assert AgentType.TRADE_AGGREGATOR.value == 'AgentTradeAggregator'


