"""
Tests for database/connection.py
Covers: session context managers, check_connection, create/drop tables.
No real MySQL connection — SQLAlchemy engine is mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


class TestDatabaseConnection:

    @pytest.mark.asyncio
    async def test_check_connection_returns_true_on_success(self):
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)
        mock_conn.execute = AsyncMock()

        with patch("nifty_quant_lab.database.connection.async_engine") as mock_engine:
            mock_engine.connect.return_value = mock_conn
            from nifty_quant_lab.database.connection import check_connection
            result = await check_connection()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_connection_returns_false_on_error(self):
        with patch("nifty_quant_lab.database.connection.async_engine") as mock_engine:
            mock_engine.connect.side_effect = Exception("connection refused")
            from nifty_quant_lab.database.connection import check_connection
            result = await check_connection()
        assert result is False

    @pytest.mark.asyncio
    async def test_create_all_tables_calls_run_sync(self):
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)
        mock_conn.run_sync = AsyncMock()

        with patch("nifty_quant_lab.database.connection.async_engine") as mock_engine:
            mock_engine.begin.return_value = mock_conn
            from nifty_quant_lab.database.connection import create_all_tables
            await create_all_tables()
        mock_conn.run_sync.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_async_session_commits_on_success(self):
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("nifty_quant_lab.database.connection.AsyncSessionLocal",
                   return_value=mock_session):
            from nifty_quant_lab.database.connection import get_async_session
            async with get_async_session() as session:
                pass  # no exception
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_async_session_rollback_on_exception(self):
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("nifty_quant_lab.database.connection.AsyncSessionLocal",
                   return_value=mock_session):
            from nifty_quant_lab.database.connection import get_async_session
            with pytest.raises(ValueError):
                async with get_async_session():
                    raise ValueError("boom")
        mock_session.rollback.assert_awaited_once()
        mock_session.commit.assert_not_awaited()

    def test_sync_session_commits(self):
        mock_session = MagicMock()
        with patch("nifty_quant_lab.database.connection.SyncSessionLocal",
                   return_value=mock_session):
            from nifty_quant_lab.database.connection import get_sync_session
            with get_sync_session() as session:
                pass
        mock_session.commit.assert_called_once()

    def test_sync_session_rollback_on_exception(self):
        mock_session = MagicMock()
        with patch("nifty_quant_lab.database.connection.SyncSessionLocal",
                   return_value=mock_session):
            from nifty_quant_lab.database.connection import get_sync_session
            with pytest.raises(RuntimeError):
                with get_sync_session():
                    raise RuntimeError("fail")
        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()
