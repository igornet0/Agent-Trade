import pytest
from httpx import AsyncClient
from backend.app import create_app


@pytest.mark.anyio
async def test_strategy_crud_minimal():
    app = create_app(create_custom_static_urls=False)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Register/login
        await ac.post("/auth/register/", json={"login": "s1", "email": "s1@example.com", "password": "secret"})
        r = await ac.post("/auth/login_user/", data={"username": "s1", "password": "secret"})
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create empty strategy (no coins/agents)
        payload = {"name": "my-str", "type": "test", "risk": 0.1, "reward": 0.2, "coins": [], "agents": []}
        r = await ac.post("/strategy/create", json=payload, headers=headers)
        assert r.status_code == 200
        sid = r.json()["id"]

        # Get
        r = await ac.get(f"/strategy/{sid}", headers=headers)
        assert r.status_code == 200

        # List
        r = await ac.get("/strategy/list", headers=headers)
        assert r.status_code == 200
        assert any(s["id"] == sid for s in r.json())

        # Delete
        r = await ac.delete(f"/strategy/{sid}", headers=headers)
        assert r.status_code == 200


