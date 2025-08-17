import pytest
import importlib.util
from pathlib import Path

pytest.importorskip('pydantic_settings', reason='env lacks pydantic_settings; skipping model import test')


def _import_models():
    root = Path(__file__).resolve().parents[1]
    models_path = root / 'src' / 'core' / 'database' / 'models' / 'Strategy_models.py'
    spec = importlib.util.spec_from_file_location("strategy_models", str(models_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@pytest.mark.anyio
async def test_agent_train_has_unified_fields():
    m = _import_models()
    at = m.AgentTrain
    # Ensure attributes exist on SQLAlchemy model
    assert hasattr(at, 'extra_config')
    assert hasattr(at, 'metrics')
    assert hasattr(at, 'artifact_path')

