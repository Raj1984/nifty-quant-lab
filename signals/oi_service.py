"""
NIFTY Quant Lab — OI Persistence Service
==========================================
Saves option chain snapshots, PCR data, and futures data to MySQL.
Runs every 5 minutes during market hours via the scheduler.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Dict, List, Optional

from sqlalchemy import select

from nifty_quant_lab.analytics.futures_analytics import FuturesAnalysisResult
from nifty_quant_lab.analytics.oi_analytics import OIAnalysisResult
from nifty_quant_lab.data.providers.nse_scraper import OptionChainSnapshot
from nifty_quant_lab.database.connection import get_async_session
from nifty_quant_lab.database.models import (
    FuturesData, OIData, PCRData, Symbol,
)
from nifty_quant_lab.database.upsert import mysql_upsert
from nifty_quant_lab.utils.logger import get_logger

logger = get_logger("oi_service")


class OIPersistenceService:
    """Saves OI/PCR/Futures snapshots to MySQL."""

    async def save_option_chain(
        self,
        snapshot: OptionChainSnapshot,
        analysis: Optional[OIAnalysisResult] = None,
    ) -> int:
        """Persist strike-level OI data from a snapshot."""
        if not snapshot or not snapshot.rows:
            return 0

        symbol_id = await self._get_symbol_id(snapshot.symbol)
        if not symbol_id:
            logger.warning(f"Symbol {snapshot.symbol} not found — run sync_symbol_registry first")
            return 0

        # Build OI records for each strike
        records = []
        for row in snapshot.rows:
            ts = snapshot.timestamp
            expiry_date = self._parse_expiry_date(row.expiry)

            for opt_type, oi, oi_change, volume, price in [
                ("CE", row.ce_oi, row.ce_oi_change, row.ce_volume, row.ce_ltp),
                ("PE", row.pe_oi, row.pe_oi_change, row.pe_volume, row.pe_ltp),
            ]:
                if oi == 0 and volume == 0:
                    continue
                oi_change_pct = round(oi_change / oi * 100, 2) if oi > 0 else None
                records.append({
                    "symbol_id": symbol_id,
                    "timestamp": ts,
                    "expiry_date": expiry_date,
                    "strike_price": row.strike,
                    "option_type": opt_type,
                    "open_interest": oi,
                    "oi_change": oi_change,
                    "oi_change_pct": oi_change_pct,
                    "volume": volume,
                    "price": price,
                })

        if not records:
            return 0

        async with get_async_session() as db:
            stmt = mysql_upsert(OIData).values(records)
            stmt = stmt.on_duplicate_key_update(
                open_interest=stmt.inserted.open_interest,
                oi_change=stmt.inserted.oi_change,
                oi_change_pct=stmt.inserted.oi_change_pct,
                volume=stmt.inserted.volume,
                price=stmt.inserted.price,
            )
            await db.execute(stmt)

        logger.info(f"✓ OI saved: {snapshot.symbol} | {len(records)} strike records")
        return len(records)

    async def save_pcr(self, snapshot: OptionChainSnapshot) -> bool:
        """Persist PCR summary for a snapshot."""
        symbol_id = await self._get_symbol_id(snapshot.symbol)
        if not symbol_id:
            return False

        expiry_date = self._parse_expiry_date(snapshot.expiry)

        async with get_async_session() as db:
            stmt = mysql_upsert(PCRData).values({
                "symbol_id": symbol_id,
                "timestamp": snapshot.timestamp,
                "expiry_date": expiry_date,
                "pcr_oi": snapshot.pcr_oi,
                "pcr_volume": snapshot.pcr_volume,
                "total_ce_oi": snapshot.total_ce_oi,
                "total_pe_oi": snapshot.total_pe_oi,
                "total_ce_volume": snapshot.total_ce_volume,
                "total_pe_volume": snapshot.total_pe_volume,
                "max_pain": snapshot.max_pain,
            })
            stmt = stmt.on_duplicate_key_update(
                pcr_oi=stmt.inserted.pcr_oi,
                pcr_volume=stmt.inserted.pcr_volume,
                total_ce_oi=stmt.inserted.total_ce_oi,
                total_pe_oi=stmt.inserted.total_pe_oi,
                max_pain=stmt.inserted.max_pain,
            )
            await db.execute(stmt)

        logger.info(f"✓ PCR saved: {snapshot.symbol} PCR={snapshot.pcr_oi:.2f}")
        return True

    async def save_futures(self, result: FuturesAnalysisResult) -> bool:
        """Persist futures basis and OI data."""
        if not result.success or not result.near_month:
            return False

        symbol_id = await self._get_symbol_id(result.symbol)
        if not symbol_id:
            return False

        nm = result.near_month
        expiry_date = nm.expiry_date or date.today()

        async with get_async_session() as db:
            stmt = mysql_upsert(FuturesData).values({
                "symbol_id": symbol_id,
                "timestamp": result.timestamp,
                "expiry_date": expiry_date,
                "spot_price": nm.spot_price,
                "futures_price": nm.futures_price,
                "basis": nm.basis,
                "basis_pct": nm.basis_pct,
                "open_interest": nm.open_interest,
                "oi_change": nm.oi_change,
                "volume": nm.volume,
                "rollover_pct": result.rollover.rollover_pct if result.rollover else None,
            })
            stmt = stmt.on_duplicate_key_update(
                futures_price=stmt.inserted.futures_price,
                basis=stmt.inserted.basis,
                basis_pct=stmt.inserted.basis_pct,
                open_interest=stmt.inserted.open_interest,
                oi_change=stmt.inserted.oi_change,
            )
            await db.execute(stmt)

        logger.info(f"✓ Futures saved: {result.symbol} basis={nm.basis:+.1f}")
        return True

    async def get_pcr_history(
        self,
        symbol: str,
        hours: int = 6,
    ) -> List[Dict]:
        """Fetch PCR time series for trend analysis."""
        from datetime import timedelta
        from sqlalchemy import desc
        since = datetime.now() - timedelta(hours=hours)

        symbol_id = await self._get_symbol_id(symbol)
        if not symbol_id:
            return []

        async with get_async_session() as db:
            result = await db.execute(
                select(PCRData)
                .where(
                    PCRData.symbol_id == symbol_id,
                    PCRData.timestamp >= since,
                )
                .order_by(PCRData.timestamp.asc())
            )
            rows = result.scalars().all()

        return [
            {
                "timestamp": r.timestamp.strftime("%H:%M"),
                "pcr_oi": r.pcr_oi,
                "pcr_volume": r.pcr_volume,
                "total_ce_oi": r.total_ce_oi,
                "total_pe_oi": r.total_pe_oi,
                "max_pain": r.max_pain,
            }
            for r in rows
        ]

    async def get_latest_oi_snapshot(
        self,
        symbol: str,
        expiry: Optional[str] = None,
    ) -> List[Dict]:
        """Get the most recent OI data per strike for a symbol."""
        symbol_id = await self._get_symbol_id(symbol)
        if not symbol_id:
            return []

        from sqlalchemy import func, desc
        async with get_async_session() as db:
            # Get latest timestamp
            latest_ts = await db.execute(
                select(func.max(OIData.timestamp))
                .where(OIData.symbol_id == symbol_id)
            )
            max_ts = latest_ts.scalar_one_or_none()
            if not max_ts:
                return []

            result = await db.execute(
                select(OIData)
                .where(
                    OIData.symbol_id == symbol_id,
                    OIData.timestamp == max_ts,
                )
                .order_by(OIData.strike_price.asc())
            )
            rows = result.scalars().all()

        return [
            {
                "strike": r.strike_price,
                "option_type": r.option_type,
                "oi": r.open_interest,
                "oi_change": r.oi_change,
                "oi_change_pct": r.oi_change_pct,
                "volume": r.volume,
                "price": r.price,
                "timestamp": r.timestamp.strftime("%H:%M"),
            }
            for r in rows
        ]

    async def _get_symbol_id(self, symbol: str) -> Optional[int]:
        """Resolve symbol string to DB id."""
        nse_sym = {
            "NIFTY": "NIFTY50",
            "BANKNIFTY": "BANKNIFTY",
            "FINNIFTY": "FINNIFTY",
        }.get(symbol.upper(), symbol.upper())

        async with get_async_session() as db:
            result = await db.execute(
                select(Symbol.id).where(
                    Symbol.symbol == nse_sym,
                    Symbol.exchange == "NSE",
                )
            )
            return result.scalar_one_or_none()

    @staticmethod
    def _parse_expiry_date(expiry_str: str) -> date:
        for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(expiry_str, fmt).date()
            except (ValueError, TypeError):
                continue
        return date.today()
