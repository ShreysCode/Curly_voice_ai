import pytest

from httpx import ASGITransport, AsyncClient
from asgi_lifespan import LifespanManager

from app.main import app


@pytest.mark.asyncio
async def test_health():

    async with LifespanManager(app):

        transport = ASGITransport(
            app=app
        )

        async with AsyncClient(
            transport=transport,
            base_url="http://test"
        ) as client:

            response = await client.get(
                "/health"
            )

        assert response.status_code == 200

        data = response.json()

        assert data["service"] == "curly-ai"


@pytest.mark.asyncio
async def test_create_session():

    async with LifespanManager(app):

        transport = ASGITransport(
            app=app
        )

        async with AsyncClient(
            transport=transport,
            base_url="http://test"
        ) as client:

            response = await client.post(
                "/v1/session"
            )

        assert response.status_code == 200

        data = response.json()

        assert "session_id" in data
        assert data["active"] is True


@pytest.mark.asyncio
async def test_delete_session():

    async with LifespanManager(app):

        transport = ASGITransport(
            app=app
        )

        async with AsyncClient(
            transport=transport,
            base_url="http://test"
        ) as client:

            create_response = await client.post(
                "/v1/session"
            )

            assert create_response.status_code == 200

            session_id = (
                create_response.json()["session_id"]
            )

            delete_response = await client.delete(
                f"/v1/session/{session_id}"
            )

        assert delete_response.status_code == 200

        data = delete_response.json()

        assert data["active"] is False