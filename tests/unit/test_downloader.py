"""
Tests for data/downloader.py
Covers: symbol registry sync, historical/intraday download orchestration,
        batch logic, upsert calls. DB and provider are fully mocked.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


def _make_df(n: int = 100, symbol: str = "TEST") -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 18000 + np.cumsum(np.random.randn(n) * 50)
    return pd.DataFrame({
        "open": close * 0.99, "high": close * 1.01,
        "low": close * 0.98, "close": close,
        "volume": np.ones(n, dtype=int) * 5_000_000,
        "symbol": symbol,
    }, index=idx)


@pytest.fixture
def mock_provider():
    from nifty_quant_lab.data.base_provider import DataFetchResult
    provider = MagicMock()
    df = _make_df()
    provider.fetch_multiple = AsyncMock(return_value={
        "NIFTY50": DataFetchResult.ok(df, "yfinance"),
        "RELIANCE": DataFetchResult.ok(df, "yfinance"),
    })
    provider.fetch_historical = AsyncMock(
        return_value=DataFetchResult.ok(df, "yfinance")
    )
    provider.fetch_intraday = AsyncMock(
        return_value=DataFetchResult.ok(df, "yfinance")
    )
    return provider


@pytest.fixture
def downloader(mock_provider):
    from nifty_quant_lab.data.downloader import NSEDataDownloader
    dl = NSEDataDownloader(provider=mock_provider)
    return dl


class TestSymbolRegistry:

    @pytest.mark.asyncio
    async def test_sync_executes_for_all_manifest_entries(self, downloader):
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock()

        with patch("nifty_quant_lab.data.downloader.get_async_session",
                   return_value=mock_session):
            with patch("nifty_quant_lab.data.downloader.mysql_upsert") as mock_upsert:
                mock_stmt = MagicMock()
                mock_stmt.on_duplicate_key_update.return_value = mock_stmt
                mock_stmt.inserted = MagicMock()
                mock_upsert.return_value.values.return_value = mock_stmt
                await downloader.sync_symbol_registry()

        # Should have been called for indices + NIFTY50 stocks
        from nifty_quant_lab.data.downloader import INDEX_MANIFEST
        from nifty_quant_lab.config.settings import NIFTY50_SYMBOLS
        expected_calls = len(INDEX_MANIFEST) + len(NIFTY50_SYMBOLS)
        assert mock_upsert.call_count == expected_calls


class TestHistoricalDownload:

    @pytest.mark.asyncio
    async def test_download_calls_provider_fetch_multiple(self, downloader, mock_provider):
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: 1))

        with patch("nifty_quant_lab.data.downloader.get_async_session",
                   return_value=mock_session):
            with patch.object(downloader, "_get_symbol_id", return_value=1):
                with patch.object(downloader, "_upsert_historical",
                                  new=AsyncMock(return_value=100)):
                    results = await downloader.download_historical(
                        symbols=["NIFTY50", "RELIANCE"], years=1
                    )

        assert mock_provider.fetch_multiple.await_count >= 1

    @pytest.mark.asyncio
    async def test_download_handles_fetch_failure_gracefully(self, downloader):
        from nifty_quant_lab.data.base_provider import DataFetchResult
        downloader.provider.fetch_multiple = AsyncMock(return_value={
            "NIFTY50": DataFetchResult.err("timeout", "yfinance"),
        })

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("nifty_quant_lab.data.downloader.get_async_session",
                   return_value=mock_session):
            results = await downloader.download_historical(
                symbols=["NIFTY50"], years=1
            )

        assert results.get("NIFTY50") is False

    @pytest.mark.asyncio
    async def test_upsert_historical_builds_correct_records(self, downloader):
        df = _make_df(50)
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()

        with patch("nifty_quant_lab.data.downloader.mysql_upsert") as mock_upsert:
            mock_stmt = MagicMock()
            mock_stmt.on_duplicate_key_update.return_value = mock_stmt
            mock_stmt.inserted = MagicMock()
            mock_upsert.return_value.values.return_value = mock_stmt

            count = await downloader._upsert_historical(mock_session, 1, df, force=False)

        assert count == 50
        # Verify records were built with correct keys
        records_arg = mock_upsert.return_value.values.call_args[0][0]
        assert len(records_arg) == 50
        assert "symbol_id" in records_arg[0]
        assert "date" in records_arg[0]
        assert "close" in records_arg[0]

    @pytest.mark.asyncio
    async def test_upsert_empty_df_returns_zero(self, downloader):
        mock_session = AsyncMock()
        count = await downloader._upsert_historical(mock_session, 1, pd.DataFrame(), False)
        assert count == 0


class TestIntradayDownload:

    @pytest.mark.asyncio
    async def test_intraday_calls_fetch_per_symbol_interval(self, downloader, mock_provider):
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("nifty_quant_lab.data.downloader.get_async_session",
                   return_value=mock_session):
            with patch.object(downloader, "_get_symbol_id", return_value=1):
                with patch.object(downloader, "_upsert_intraday", new=AsyncMock(return_value=30)):
                    results = await downloader.download_intraday(
                        symbols=["NIFTY50"],
                        intervals=["5m"],
                        days={"5m": 7},
                    )

        assert mock_provider.fetch_intraday.await_count == 1
        assert results["NIFTY50"]["5m"] is True

    @pytest.mark.asyncio
    async def test_update_today_delegates_to_download_historical(self, downloader):
        with patch.object(downloader, "download_historical",
                           new=AsyncMock(return_value={})) as mock_dl:
            await downloader.update_today()
        mock_dl.assert_awaited_once()
        # Verify force_refresh=True
        call_kwargs = mock_dl.call_args[1]
        assert call_kwargs.get("force_refresh") is True


class TestSymbolIdCache:

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_db_query(self, downloader):
        downloader._symbol_cache["NIFTY50"] = 42
        mock_session = AsyncMock()
        sid = await downloader._get_symbol_id("NIFTY50", mock_session)
        assert sid == 42
        mock_session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_queries_db(self, downloader):
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 7
        mock_session.execute = AsyncMock(return_value=mock_result)

        sid = await downloader._get_symbol_id("BANKNIFTY", mock_session)
        assert sid == 7
        assert downloader._symbol_cache["BANKNIFTY"] == 7
