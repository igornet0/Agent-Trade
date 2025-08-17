import pytest
from httpx import AsyncClient
from backend.app import create_app


@pytest.mark.anyio
async def test_auth_register_and_login():
    app = create_app(create_custom_static_urls=False)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Register
        payload = {"login": "user1", "email": "user1@example.com", "password": "secret"}
        r = await ac.post("/auth/register/", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "access_token" in data and "refresh_token" in data

        # Login
        form = {"username": payload["login"], "password": payload["password"]}
        r = await ac.post("/auth/login_user/", data=form)
        assert r.status_code == 200
        data = r.json()
        assert "access_token" in data and "refresh_token" in data


