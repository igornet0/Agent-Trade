import pytest
from httpx import AsyncClient
from backend.app.configuration.schemas.agent import AgentType

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


@pytest.mark.anyio
async def test_agent_type_enum_extended():
    # Ensure new types exist and are strings
    assert AgentType.PREDTIME.value == 'AgentPredTime'
    assert AgentType.TRADETIME.value == 'AgentTradeTime'
    assert AgentType.NEWS.value == 'AgentNews'
    # Newly added
    assert AgentType.RISK.value == 'AgentRisk'
    assert AgentType.TRADE_AGGREGATOR.value == 'AgentTradeAggregator'


@pytest.mark.anyio
async def test_agents_filter_by_type_query_param_contract():
    app = create_app(create_custom_static_urls=False)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        await ac.post("/auth/register/", json={"login": "admin2", "email": "admin2@example.com", "password": "secret"})
        r = await ac.post("/auth/login_user/", data={"username": "admin2", "password": "secret"})
        token = r.json()["access_token"]
        # Без прав администратора ожидаем 403
        r = await ac.get("/api_db_agent/agents/", params={"type": AgentType.PREDTIME.value}, headers={"Authorization": f"Bearer {token}"})
        assert r.status_code in (401, 403)


@pytest.mark.anyio
async def test_unified_train_contract_mismatch_type():
    app = create_app(create_custom_static_urls=False)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        await ac.post("/auth/register/", json={"login": "admin3", "email": "admin3@example.com", "password": "secret"})
        r = await ac.post("/auth/login_user/", data={"username": "admin3", "password": "secret"})
        token = r.json()["access_token"]
        # Пытаемся вызвать unified train без прав админа и с несовпадающим типом
        payload = {
            "name": "test_agent",
            "type": AgentType.TRADETIME.value,
            "timeframe": "5m",
            "coins": [],
            "features": [],
            "train_data": {"epochs": 1, "batch_size": 1, "learning_rate": 0.001, "weight_decay": 0.0}
        }
        r = await ac.post(f"/api_db_agent/{AgentType.PREDTIME.value}/train", json=payload, headers={"Authorization": f"Bearer {token}"})
        # либо 401/403 (нет прав админа), либо 400 (рассинхрон типов) — контрактная проверка
        assert r.status_code in (400, 401, 403)


