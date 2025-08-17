import pytest
from httpx import AsyncClient

from backend.app import create_app


@pytest.mark.anyio
async def test_train_agent_endpoint_contract():
    app = create_app(create_custom_static_urls=False)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # admin user (simplify: register + manually set role in DB if needed - skipping full admin path)
        await ac.post("/auth/register/", json={"login": "admin1", "email": "admin1@example.com", "password": "secret"})
        r = await ac.post("/auth/login_user/", data={"username": "admin1", "password": "secret"})
        token = r.json()["access_token"]
        # Try to call get train agents (should require admin) — here we just assert 403 for non-admin
        r = await ac.get("/api_db_agent/train_agents/", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code in (401, 403)


