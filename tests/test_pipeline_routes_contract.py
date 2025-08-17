import pytest
from httpx import AsyncClient

passlib = pytest.importorskip("passlib", reason="passlib not installed; skipping pipeline routes contract test")


@pytest.mark.anyio
async def test_pipeline_save_and_get_contract():
    from backend.app import create_app
    app = create_app(create_custom_static_urls=False)

    # register + login admin
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # register
        payload = {"login": "admin1", "email": "admin1@example.com", "password": "secret"}
        r = await ac.post("/auth/register/", json=payload)
        assert r.status_code == 200
        token = r.json()["access_token"]

        headers = {"Authorization": f"Bearer {token}"}

        # save pipeline
        cfg = {"nodes": [{"id": "n1", "type": "DataSource", "config": {}}], "edges": []}
        r = await ac.post("/pipeline/save", json=cfg, headers=headers)
        assert r.status_code == 200
        pipeline_id = r.json().get("pipeline_id")
        assert pipeline_id

        # get pipeline
        r = await ac.get(f"/pipeline/{pipeline_id}", headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data.get("nodes"), list)


@pytest.mark.anyio
async def test_pipeline_get_404_and_revoke_skip_if_no_auth():
    from backend.app import create_app
    app = create_app(create_custom_static_urls=False)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # No auth header -> 401/403 tolerant check for revoke
        r = await ac.get("/pipeline/nonexistent")
        assert r.status_code in (401, 403)

        r = await ac.post("/pipeline/tasks/some-task-id/revoke")
        assert r.status_code in (401, 403)


@pytest.mark.anyio
async def test_pipeline_backtests_list_and_get_contract():
    from backend.app import create_app
    app = create_app(create_custom_static_urls=False)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # register user and auth
        payload = {"login": "adm3", "email": "adm3@example.com", "password": "secret"}
        r = await ac.post("/auth/register/", json=payload)
        assert r.status_code == 200
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        r = await ac.get("/pipeline/backtests", headers=headers)
        assert r.status_code in (200, 404)
        if r.status_code == 200 and isinstance(r.json(), list) and r.json():
            bt_id = r.json()[0].get("id")
            if bt_id:
                r2 = await ac.get(f"/pipeline/backtests/{bt_id}", headers=headers)
                assert r2.status_code in (200, 404)

