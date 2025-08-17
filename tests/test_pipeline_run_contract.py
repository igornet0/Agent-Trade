import pytest
from httpx import AsyncClient


passlib = pytest.importorskip("passlib", reason="passlib not installed; skipping pipeline run contract test")


@pytest.mark.anyio
async def test_pipeline_run_returns_task_id():
    from backend.app import create_app
    app = create_app(create_custom_static_urls=False)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # register user and auth
        payload = {"login": "adm2", "email": "adm2@example.com", "password": "secret"}
        r = await ac.post("/auth/register/", json=payload)
        assert r.status_code == 200
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        cfg = {"nodes": [{"id": "n1", "type": "DataSource", "config": {}}], "edges": []}
        r = await ac.post("/pipeline/run", json=cfg, headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert "task_id" in data and isinstance(data["task_id"], str)


