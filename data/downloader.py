"""
NIFTY Quant Lab - NSE Data Downloader
========================================
Orchestrates historical and intraday data downloads.
Persists to MySQL via SQLAlchemy + MySQL INSERT ON DUPLICATE KEY UPDATE.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import select

from nifty_quant_lab.config.settings import settings, NIFTY50_SYMBOLS, SECTORAL_INDICES
from nifty_quant_lab.data.providers.yfinance_provider import YFinanceProvider
from nifty_quant_lab.database.connection import get_async_session
from nifty_quant_lab.database.models import AssetType, HistoricalPrice, IntradayPrice, Symbol
from nifty_quant_lab.database.upsert import mysql_upsert
from nifty_quant_lab.utils.logger import get_logger

logger = get_logger("downloader")

INDEX_MANIFEST: List[Tuple[str, str, AssetType, bool, bool]] = [
    ("NIFTY50",   "NIFTY 50 Index",         AssetType.INDEX, False, True),
    ("BANKNIFTY", "BANK NIFTY Index",        AssetType.INDEX, False, True),
    ("FINNIFTY",  "NIFTY Financial Services", AssetType.INDEX, False, True),
    ("MIDCAP50",  "NIFTY MIDCAP 50 Index",   AssetType.INDEX, False, False),
    ("INDIA_VIX", "India VIX",               AssetType.INDEX, False, False),
    ("NIFTY_IT",          "NIFTY IT",          AssetType.INDEX, False, False),
    ("NIFTY_PHARMA",      "NIFTY Pharma",      AssetType.INDEX, False, False),
    ("NIFTY_AUTO",        "NIFTY Auto",        AssetType.INDEX, False, False),
    ("NIFTY_FMCG",        "NIFTY FMCG",        AssetType.INDEX, False, False),
    ("NIFTY_METAL",       "NIFTY Metal",       AssetType.INDEX, False, False),
    ("NIFTY_REALTY",      "NIFTY Realty",      AssetType.INDEX, False, False),
    ("NIFTY_ENERGY",      "NIFTY Energy",      AssetType.INDEX, False, False),
    ("NIFTY_PSU_BANK",    "NIFTY PSU Bank",    AssetType.INDEX, False, False),
    ("NIFTY_FIN_SERVICE", "NIFTY Fin Service", AssetType.INDEX, False, False),
]


class NSEDataDownloader:

    def __init__(self, provider: Optional[YFinanceProvider] = None):
        self.provider = provider or YFinanceProvider(max_workers=8)
        self._symbol_cache: Dict[str, int] = {}

    # ─────────────────────────────────────────────────────────
    # SYMBOL REGISTRY
    # ─────────────────────────────────────────────────────────

    async def sync_symbol_registry(self) -> None:
        logger.info("Syncing symbol registry...")
        async with get_async_session() as session:
            for symbol, name, asset_type, nifty50, fo in INDEX_MANIFEST:
                stmt = mysql_upsert(Symbol).values(
                    symbol=symbol, name=name, asset_type=asset_type,
                    exchange="NSE", nifty50=nifty50, is_fo_eligible=fo,
                )
                stmt = stmt.on_duplicate_key_update(
                    name=stmt.inserted.name,
                    is_active=True,
                )
                await session.execute(stmt)

            for symbol in NIFTY50_SYMBOLS:
                stmt = mysql_upsert(Symbol).values(
                    symbol=symbol, name=symbol,
                    asset_type=AssetType.EQUITY,
                    exchange="NSE", nifty50=True, is_fo_eligible=True,
                )
                stmt = stmt.on_duplicate_key_update(
                    nifty50=True, is_active=True,
                )
                await session.execute(stmt)

        logger.info(f"Symbol registry synced: {len(INDEX_MANIFEST)} indices + {len(NIFTY50_SYMBOLS)} equities")

    async def _get_symbol_id(self, symbol: str, session) -> Optional[int]:
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]
        result = await session.execute(
            select(Symbol.id).where(Symbol.symbol == symbol, Symbol.exchange == "NSE")
        )
        row = result.scalar_one_or_none()
        if row:
            self._symbol_cache[symbol] = row
        return row

    # ─────────────────────────────────────────────────────────
    # HISTORICAL DOWNLOAD
    # ─────────────────────────────────────────────────────────

    async def download_historical(
        self,
        symbols: Optional[List[str]] = None,
        years: int = 10,
        force_refresh: bool = False,
    ) -> Dict[str, bool]:
        if symbols is None:
            symbols = [m[0] for m in INDEX_MANIFEST] + list(NIFTY50_SYMBOLS)

        end = date.today()
        start = date(end.year - years, end.month, end.day)
        logger.info(f"Historical download: {len(symbols)} symbols | {start} → {end}")

        batch_size = 20
        results: Dict[str, bool] = {}

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            fetch_results = await self.provider.fetch_multiple(batch, start, end)

            async with get_async_session() as session:
                for symbol, result in fetch_results.items():
                    if not result.success or result.data is None:
                        results[symbol] = False
                        continue
                    symbol_id = await self._get_symbol_id(symbol, session)
                    if symbol_id is None:
                        results[symbol] = False
                        continue
                    inserted = await self._upsert_historical(
                        session, symbol_id, result.data, force_refresh
                    )
                    results[symbol] = True
                    logger.debug(f"✓ {symbol}: {inserted} rows")

            await asyncio.sleep(1)

        success = sum(1 for v in results.values() if v)
        logger.info(f"Historical download complete: {success}/{len(symbols)} OK")
        return results

    async def _upsert_historical(
        self, session, symbol_id: int, df: pd.DataFrame, force: bool
    ) -> int:
        records = []
        for ts, row in df.iterrows():
            records.append({
                "symbol_id": symbol_id,
                "date": ts.date() if hasattr(ts, "date") else ts,
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": int(row.get("volume", 0)),
                "adjusted_close": float(row.get("adjusted_close", row.get("close", 0))),
                "vwap": float(row["vwap"]) if row.get("vwap") else None,
            })
        if not records:
            return 0

        stmt = mysql_upsert(HistoricalPrice).values(records)
        if force:
            stmt = stmt.on_duplicate_key_update(
                open=stmt.inserted.open,
                high=stmt.inserted.high,
                low=stmt.inserted.low,
                close=stmt.inserted.close,
                volume=stmt.inserted.volume,
                adjusted_close=stmt.inserted.adjusted_close,
            )
        else:
            # INSERT IGNORE equivalent: update with same value (no-op)
            stmt = stmt.on_duplicate_key_update(symbol_id=stmt.inserted.symbol_id)

        await session.execute(stmt)
        return len(records)

    # ─────────────────────────────────────────────────────────
    # INTRADAY DOWNLOAD
    # ─────────────────────────────────────────────────────────

    async def download_intraday(
        self,
        symbols: Optional[List[str]] = None,
        intervals: Optional[List[str]] = None,
        days: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Dict[str, bool]]:
        if symbols is None:
            symbols = ["NIFTY50", "BANKNIFTY", "FINNIFTY"] + list(NIFTY50_SYMBOLS[:20])
        if intervals is None:
            intervals = ["5m", "15m", "30m", "1h"]
        if days is None:
            days = {"1m": 7, "5m": 30, "15m": 60, "30m": 365, "1h": 365}

        results: Dict[str, Dict[str, bool]] = {s: {} for s in symbols}

        for interval in intervals:
            n_days = days.get(interval, 30)
            for symbol in symbols:
                result = await self.provider.fetch_intraday(symbol, interval, days=n_days)
                if not result.success or result.data is None:
                    results[symbol][interval] = False
                    continue
                async with get_async_session() as session:
                    symbol_id = await self._get_symbol_id(symbol, session)
                    if symbol_id is None:
                        results[symbol][interval] = False
                        continue
                    await self._upsert_intraday(session, symbol_id, result.data, interval)
                    results[symbol][interval] = True
                await asyncio.sleep(0.2)

        return results

    async def _upsert_intraday(
        self, session, symbol_id: int, df: pd.DataFrame, interval: str
    ) -> int:
        records = []
        for ts, row in df.iterrows():
            if hasattr(ts, "tz_localize") and ts.tz is None:
                ts = ts.tz_localize("Asia/Kolkata")
            records.append({
                "symbol_id": symbol_id,
                "timestamp": ts,
                "interval": interval,
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": int(row.get("volume", 0)),
            })
        if not records:
            return 0

        stmt = mysql_upsert(IntradayPrice).values(records)
        stmt = stmt.on_duplicate_key_update(
            open=stmt.inserted.open,
            high=stmt.inserted.high,
            low=stmt.inserted.low,
            close=stmt.inserted.close,
            volume=stmt.inserted.volume,
        )
        await session.execute(stmt)
        return len(records)

    # ─────────────────────────────────────────────────────────
    # INCREMENTAL UPDATE
    # ─────────────────────────────────────────────────────────

    async def update_today(self, symbols: Optional[List[str]] = None) -> None:
        if symbols is None:
            symbols = [m[0] for m in INDEX_MANIFEST] + list(NIFTY50_SYMBOLS)
        await self.download_historical(symbols=symbols, years=1, force_refresh=True)

    async def run_full_setup(self) -> None:
        logger.info("=" * 60)
        logger.info("NIFTY Quant Lab — Full Data Setup")
        logger.info("=" * 60)
        await self.sync_symbol_registry()
        await self.download_historical(years=10)
        await self.download_intraday()
        logger.info("Full setup complete.")
