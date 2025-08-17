import pytest
from httpx import AsyncClient
from backend.app import create_app


async def _register_and_login(ac: AsyncClient):
    payload = {"login": "bill1", "email": "bill1@example.com", "password": "secret"}
    await ac.post("/auth/register/", json=payload)
    form = {"username": payload["login"], "password": payload["password"]}
    r = await ac.post("/auth/login_user/", data=form)
    return r.json()["access_token"]


@pytest.mark.anyio
async def test_billing_idempotency():
    app = create_app(create_custom_static_urls=False)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        token = await _register_and_login(ac)
        headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": "k1"}
        r1 = await ac.post("/billing/deposit", params={"amount": 10}, headers=headers)
        r2 = await ac.post("/billing/deposit", params={"amount": 10}, headers=headers)
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["operation_id"] == r2.json()["operation_id"]


