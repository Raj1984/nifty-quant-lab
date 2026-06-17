"""
Tests for api/app.py — FastAPI routes.
Uses httpx.AsyncClient with ASGITransport so routes are exercised
without a real DB — database dependency is overridden.
"""

from __future__ import annotations

import pytest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi import FastAPI


# ─────────────────────────────────────────────────────────────
# APP FIXTURE — bypass lifespan (no DB/scheduler on import)
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def app():
    """Return the FastAPI app with lifespan disabled for unit testing."""
    with patch("nifty_quant_lab.api.app.create_all_tables", new=AsyncMock()):
        with patch("nifty_quant_lab.api.app.scheduler"):
            with patch("nifty_quant_lab.api.app.setup_logging"):
                from nifty_quant_lab.api import app as api_module
                # Re-import to get fresh app without triggered lifespan
                import importlib
                importlib.reload(api_module)
                return api_module.app


@pytest.fixture
def mock_db():
    """Async mock DB session."""
    db = AsyncMock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
async def client(app, mock_db):
    """Async test client with DB dependency overridden."""
    from nifty_quant_lab.database.connection import get_db

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c

    app.dependency_overrides.clear()


# ─────────────────────────────────────────────────────────────
# HEALTH ENDPOINTS
# ─────────────────────────────────────────────────────────────

class TestHealthEndpoints:

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "NIFTY" in data["service"]

    @pytest.mark.asyncio
    async def test_health_db_ok(self, client):
        with patch("nifty_quant_lab.api.app.check_connection", new=AsyncMock(return_value=True)):
            resp = await client.get("/health/db")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_db_unavailable(self, client):
        with patch("nifty_quant_lab.api.app.check_connection", new=AsyncMock(return_value=False)):
            resp = await client.get("/health/db")
        assert resp.status_code == 503


# ─────────────────────────────────────────────────────────────
# MARKET DATA ENDPOINTS
# ─────────────────────────────────────────────────────────────

class TestMarketEndpoints:

    def _mock_symbol_result(self, mock_db, symbol_id: int = 1, name: str = "NIFTY 50"):
        mock_db.execute = AsyncMock(return_value=MagicMock(
            first=MagicMock(return_value=(symbol_id, name)),
            scalar_one_or_none=MagicMock(return_value=symbol_id),
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        ))

    @pytest.mark.asyncio
    async def test_nifty_endpoint_404_on_missing_symbol(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=MagicMock(
            first=MagicMock(return_value=None)
        ))
        resp = await client.get("/api/nifty")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_historical_endpoint_requires_symbol(self, client):
        resp = await client.get("/api/historical")
        assert resp.status_code == 422  # missing required query param

    @pytest.mark.asyncio
    async def test_historical_days_validation(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=MagicMock(
            first=MagicMock(return_value=None)
        ))
        # days=0 should fail validation
        resp = await client.get("/api/historical?symbol=RELIANCE&days=0")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_indicators_endpoint_requires_symbol(self, client):
        resp = await client.get("/api/indicators")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_indicators_404_on_missing_symbol(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=None)
        ))
        resp = await client.get("/api/indicators?symbol=UNKNOWN")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_sr_endpoint_requires_symbol(self, client):
        resp = await client.get("/api/sr")
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────
# SCANNER ENDPOINT
# ─────────────────────────────────────────────────────────────

class TestScannerEndpoints:

    @pytest.mark.asyncio
    async def test_scanner_invalid_signal_returns_400(self, client, mock_db):
        resp = await client.get("/api/scanner?signal=INVALID")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_scanner_returns_empty_results(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=MagicMock(
            all=MagicMock(return_value=[])
        ))
        resp = await client.get("/api/scanner")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["count"] == 0

    @pytest.mark.asyncio
    async def test_scan_run_endpoint_returns_ok(self, client):
        import asyncio as asyncio_mod
        with patch.object(asyncio_mod, "create_task", return_value=MagicMock()):
            resp = await client.post("/api/scan/run")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_min_score_filter_accepted(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=MagicMock(
            all=MagicMock(return_value=[])
        ))
        resp = await client.get("/api/scanner?min_score=70")
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────
# RESPONSE MODEL
# ─────────────────────────────────────────────────────────────

class TestApiResponse:

    def test_api_response_has_timestamp(self):
        from nifty_quant_lab.api.app import ApiResponse
        r = ApiResponse(status="ok", data={})
        assert r.timestamp != ""

    def test_api_response_status_field(self):
        from nifty_quant_lab.api.app import ApiResponse
        r = ApiResponse(status="error", data=None, message="failed")
        assert r.status == "error"
        assert r.message == "failed"
