"""
Tests for signals/scanner_service.py
Covers: save_scan_session, get_latest_results, purge_old_results.
All DB interactions mocked.
"""

from __future__ import annotations

import pytest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from nifty_quant_lab.database.models import SignalType
from nifty_quant_lab.signals.scanner import ScanResult, ScanSession


def _make_session(n_buy: int = 3, n_watch: int = 2) -> ScanSession:
    session = ScanSession(scan_date=date.today(), total_scanned=n_buy + n_watch)
    for i in range(n_buy):
        session.results.append(ScanResult(
            symbol=f"BUY{i}", signal=SignalType.BUY,
            score=70 + i, close_price=1000.0 + i,
            scan_date=date.today(),
        ))
    for i in range(n_watch):
        session.results.append(ScanResult(
            symbol=f"WATCH{i}", signal=SignalType.WATCHLIST,
            score=50 + i, close_price=500.0 + i,
            scan_date=date.today(),
        ))
    return session


class TestScannerPersistenceService:

    @pytest.fixture
    def service(self):
        from nifty_quant_lab.signals.scanner_service import ScannerPersistenceService
        return ScannerPersistenceService()

    @pytest.mark.asyncio
    async def test_save_empty_session_returns_zero(self, service):
        session = ScanSession(scan_date=date.today(), total_scanned=0)
        result = await service.save_scan_session(session)
        assert result == 0

    @pytest.mark.asyncio
    async def test_save_session_resolves_symbol_ids(self, service):
        session = _make_session(n_buy=2, n_watch=1)

        mock_db = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)
        mock_db.execute = AsyncMock()

        with patch.object(service, "_resolve_symbol_ids",
                          new=AsyncMock(return_value={"BUY0": 1, "BUY1": 2, "WATCH0": 3})):
            with patch("nifty_quant_lab.signals.scanner_service.get_async_session",
                       return_value=mock_db):
                with patch("nifty_quant_lab.signals.scanner_service.mysql_upsert") as mock_up:
                    mock_stmt = MagicMock()
                    mock_stmt.on_duplicate_key_update.return_value = mock_stmt
                    mock_stmt.inserted = MagicMock()
                    mock_up.return_value.values.return_value = mock_stmt
                    count = await service.save_scan_session(session)

        assert count == 3

    @pytest.mark.asyncio
    async def test_save_skips_unknown_symbols(self, service):
        session = _make_session(n_buy=2)

        mock_db = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)
        mock_db.execute = AsyncMock()

        # Only BUY0 is known; BUY1 is missing
        with patch.object(service, "_resolve_symbol_ids",
                          new=AsyncMock(return_value={"BUY0": 1})):
            with patch("nifty_quant_lab.signals.scanner_service.get_async_session",
                       return_value=mock_db):
                with patch("nifty_quant_lab.signals.scanner_service.mysql_upsert") as mock_up:
                    mock_stmt = MagicMock()
                    mock_stmt.on_duplicate_key_update.return_value = mock_stmt
                    mock_stmt.inserted = MagicMock()
                    mock_up.return_value.values.return_value = mock_stmt
                    count = await service.save_scan_session(session)

        assert count == 1  # Only the known symbol saved

    @pytest.mark.asyncio
    async def test_get_latest_results_returns_list(self, service):
        mock_row = MagicMock()
        mock_row.signal = SignalType.BUY
        mock_row.score = 80.0
        mock_row.close_price = 1500.0
        mock_row.scan_date = date.today()
        mock_row.rsi = 62.0
        mock_row.notes = "EMA✓ RSI✓"
        mock_row.ema20_above_ema50 = True
        mock_row.rsi_above_55 = True
        mock_row.macd_bullish_cross = False
        mock_row.price_above_supertrend = True
        mock_row.volume_above_avg = False
        mock_row.week52_breakout = False

        mock_db = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)
        mock_result = MagicMock()
        mock_result.all.return_value = [(mock_row, "RELIANCE", "Reliance Industries", "Energy")]
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("nifty_quant_lab.signals.scanner_service.get_async_session",
                   return_value=mock_db):
            rows = await service.get_latest_results()

        assert len(rows) == 1
        assert rows[0]["symbol"] == "RELIANCE"
        assert rows[0]["signal"] == "BUY"
        assert rows[0]["score"] == 80.0

    @pytest.mark.asyncio
    async def test_purge_old_results_returns_count(self, service):
        mock_db = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)
        mock_result = MagicMock()
        mock_result.rowcount = 42
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("nifty_quant_lab.signals.scanner_service.get_async_session",
                   return_value=mock_db):
            deleted = await service.purge_old_results(keep_days=30)

        assert deleted == 42

    @pytest.mark.asyncio
    async def test_resolve_symbol_ids_returns_mapping(self, service):
        mock_db = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)
        mock_result = MagicMock()
        mock_result.all.return_value = [("NIFTY50", 1), ("RELIANCE", 2)]
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("nifty_quant_lab.signals.scanner_service.get_async_session",
                   return_value=mock_db):
            mapping = await service._resolve_symbol_ids(["NIFTY50", "RELIANCE"])

        assert mapping == {"NIFTY50": 1, "RELIANCE": 2}
