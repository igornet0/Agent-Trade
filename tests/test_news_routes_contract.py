import pytest
import importlib

# Skip this test if heavy deps are missing in the test environment
pytest.importorskip('pydantic_settings', reason='env lacks pydantic_settings; skipping celery task import test')
pytest.importorskip('passlib', reason='env lacks passlib; skipping celery task import test')


@pytest.mark.anyio
async def test_train_news_task_placeholder_runs():
    m = importlib.import_module('backend.celery_app.tasks')
    task = getattr(m, 'train_news_task')
    res = task.run(coins=[1, 2], config={})
    assert isinstance(res, dict)
    assert res.get('status') in ('success', 'error')


