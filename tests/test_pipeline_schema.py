import pytest
import importlib.util
from pathlib import Path


def _import_pipeline_schema():
    root = Path(__file__).resolve().parents[1]
    schema_path = root / 'src' / 'backend' / 'app' / 'configuration' / 'schemas' / 'pipeline.py'
    spec = importlib.util.spec_from_file_location("pipeline_schema", str(schema_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@pytest.mark.anyio
async def test_pipeline_config_minimal():
    m = _import_pipeline_schema()
    cfg = m.PipelineConfig(nodes=[m.PipelineNode(id='n1', type='DataSource', config={'source': 'ohlcv'})])
    assert isinstance(cfg.nodes, list)
    assert cfg.nodes[0].id == 'n1'


@pytest.mark.anyio
async def test_pipeline_config_with_edges_and_window():
    m = _import_pipeline_schema()
    node_a = m.PipelineNode(id='a', type='DataSource', config={})
    node_b = m.PipelineNode(id='b', type='Pred_time', config={'seq_len': 96})
    edge = m.PipelineEdge(source='a', target='b')
    cfg = m.PipelineConfig(nodes=[node_a, node_b], edges=[edge], timeframe='5m')
    assert cfg.timeframe == '5m'
    assert len(cfg.edges) == 1


