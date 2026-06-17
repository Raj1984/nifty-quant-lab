"""
NIFTY Quant Lab - Scanner Persistence Service
===============================================
Persists SwingScanner results to PostgreSQL.
Bridge between the in-memory ScanSession and the database layer.

Inspired by qlib's workflow R (experiment tracking) — every scan run
is recorded with full metadata, enabling historical comparison.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional

from sqlalchemy import select, delete
from nifty_quant_lab.database.upsert import mysql_upsert

from nifty_quant_lab.database.connection import get_async_session
from nifty_quant_lab.database.models import ScannerResult, Symbol, SignalType
from nifty_quant_lab.signals.scanner import ScanResult, ScanSession
from nifty_quant_lab.utils.logger import get_logger

logger = get_logger("scanner_service")


class ScannerPersistenceService:
    """
    Saves and retrieves scanner sessions from PostgreSQL.

    Design: upsert by (symbol_id, scan_date) so re-running the scanner
    on the same day just refreshes the record — no duplicates.
    """

    async def save_scan_session(self, session: ScanSession) -> int:
        """
        Persist all results from a ScanSession to DB.

        Returns:
            Number of rows upserted.
        """
        if not session.results:
            logger.warning("ScanSession has no results — nothing to save")
            return 0

        # Pre-load symbol ID map
        symbol_ids = await self._resolve_symbol_ids([r.symbol for r in session.results])

        records = []
        for result in session.results:
            sid = symbol_ids.get(result.symbol)
            if sid is None:
                logger.warning(f"Symbol {result.symbol} not found in DB — skipping")
                continue
            records.append({
                "symbol_id": sid,
                "scan_date": result.scan_date,
                "signal": result.signal,
                "score": result.score,
                "ema20_above_ema50": result.ema20_above_ema50,
                "rsi_above_55": result.rsi_above_55,
                "macd_bullish_cross": result.macd_bullish_cross,
                "price_above_supertrend": result.price_above_supertrend,
                "volume_above_avg": result.volume_above_avg,
                "week52_breakout": result.week52_breakout,
                "close_price": result.close_price,
                "ema_20": result.ema_20,
                "ema_50": result.ema_50,
                "rsi": result.rsi,
                "volume": result.volume,
                "volume_avg_20": result.volume_avg_20,
                "week52_high": result.week52_high,
                "notes": result.notes,
            })

        if not records:
            return 0

        async with get_async_session() as db:
            stmt = mysql_upsert(ScannerResult).values(records)
            stmt = stmt.on_duplicate_key_update(
                signal=stmt.inserted.signal,
                score=stmt.inserted.score,
                ema20_above_ema50=stmt.inserted.ema20_above_ema50,
                rsi_above_55=stmt.inserted.rsi_above_55,
                macd_bullish_cross=stmt.inserted.macd_bullish_cross,
                price_above_supertrend=stmt.inserted.price_above_supertrend,
                volume_above_avg=stmt.inserted.volume_above_avg,
                week52_breakout=stmt.inserted.week52_breakout,
                close_price=stmt.inserted.close_price,
                rsi=stmt.inserted.rsi,
                notes=stmt.inserted.notes,
            )
            await db.execute(stmt)

        logger.info(
            f"✓ Scan session saved: {len(records)} results | "
            f"{sum(1 for r in records if r['signal'] == SignalType.BUY)} BUY | "
            f"date={session.scan_date}"
        )
        return len(records)

    async def get_latest_results(
        self,
        signal: Optional[SignalType] = None,
        min_score: float = 0,
        limit: int = 50,
    ) -> List[Dict]:
        """Fetch the most recent scan results from DB."""
        async with get_async_session() as db:
            query = (
                select(ScannerResult, Symbol.symbol, Symbol.name, Symbol.sector)
                .join(Symbol, ScannerResult.symbol_id == Symbol.id)
                .order_by(ScannerResult.score.desc())
            )
            if signal:
                query = query.where(ScannerResult.signal == signal)
            if min_score > 0:
                query = query.where(ScannerResult.score >= min_score)
            query = query.limit(limit)

            result = await db.execute(query)
            rows = result.all()

        return [
            {
                "symbol": sym,
                "name": name,
                "sector": sector,
                "signal": r.signal.value,
                "score": r.score,
                "close_price": r.close_price,
                "scan_date": str(r.scan_date),
                "rsi": r.rsi,
                "conditions_met": sum([
                    r.ema20_above_ema50, r.rsi_above_55, r.macd_bullish_cross,
                    r.price_above_supertrend, r.volume_above_avg, r.week52_breakout,
                ]),
                "notes": r.notes,
            }
            for r, sym, name, sector in rows
        ]

    async def get_scan_history(self, symbol: str, days: int = 30) -> List[Dict]:
        """Historical scan results for one symbol."""
        from datetime import timedelta
        since = date.today() - timedelta(days=days)

        async with get_async_session() as db:
            sym_res = await db.execute(
                select(Symbol.id).where(Symbol.symbol == symbol.upper(), Symbol.exchange == "NSE")
            )
            symbol_id = sym_res.scalar_one_or_none()
            if not symbol_id:
                return []

            result = await db.execute(
                select(ScannerResult)
                .where(
                    ScannerResult.symbol_id == symbol_id,
                    ScannerResult.scan_date >= since,
                )
                .order_by(ScannerResult.scan_date.desc())
            )
            rows = result.scalars().all()

        return [
            {
                "scan_date": str(r.scan_date),
                "signal": r.signal.value,
                "score": r.score,
                "close_price": r.close_price,
                "conditions_met": sum([
                    r.ema20_above_ema50, r.rsi_above_55, r.macd_bullish_cross,
                    r.price_above_supertrend, r.volume_above_avg, r.week52_breakout,
                ]),
            }
            for r in rows
        ]

    async def _resolve_symbol_ids(self, symbols: List[str]) -> Dict[str, int]:
        """Batch resolve symbol strings to DB IDs."""
        async with get_async_session() as db:
            result = await db.execute(
                select(Symbol.symbol, Symbol.id)
                .where(Symbol.symbol.in_(symbols), Symbol.exchange == "NSE")
            )
            return {row[0]: row[1] for row in result.all()}

    async def purge_old_results(self, keep_days: int = 90) -> int:
        """Delete scan results older than keep_days."""
        from datetime import timedelta
        cutoff = date.today() - timedelta(days=keep_days)
        async with get_async_session() as db:
            result = await db.execute(
                delete(ScannerResult).where(ScannerResult.scan_date < cutoff)
            )
            deleted = result.rowcount
        logger.info(f"Purged {deleted} old scanner results (before {cutoff})")
        return deleted
