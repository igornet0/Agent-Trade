import pytest
from httpx import AsyncClient

from backend.app import create_app
from core.database import db_helper
from core.database.orm import orm_add_coin, orm_get_coin_by_name


async def _register_and_login(ac: AsyncClient, login: str):
    await ac.post("/auth/register/", json={"login": login, "email": f"{login}@example.com", "password": "secret"})
    r = await ac.post("/auth/login_user/", data={"username": login, "password": "secret"})
    return r.json()["access_token"]


@pytest.mark.anyio
async def test_order_create_and_list():
    app = create_app(create_custom_static_urls=False)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        token = await _register_and_login(ac, "ord1")
        headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": "dep-ord1"}
        # Ensure coin exists in DB
        async with db_helper.get_session() as session:
            await orm_add_coin(session, name="TEST", price_now=1.0)
            coin = await orm_get_coin_by_name(session, "TEST")

        # Deposit funds
        r = await ac.post("/billing/deposit", params={"amount": 1000}, headers=headers)
        assert r.status_code == 200

        # Create BUY order
        order = {"coin_id": coin.id, "type": "buy", "amount": 10, "price": 5}
        r = await ac.post("/api_db_order/create_order/", json=order, headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["id"]

        # List orders
        r = await ac.get("/api_db_order/get_orders/", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert any(o["id"] == data["id"] for o in r.json())


