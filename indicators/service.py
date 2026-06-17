"""
NIFTY Quant Lab - Indicator Persistence Service
=================================================
Computes all indicators and persists them to PostgreSQL.
Acts as the bridge between IndicatorEngine (pure computation)
and the database layer.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import select
from nifty_quant_lab.database.upsert import mysql_upsert

from nifty_quant_lab.database.connection import get_async_session
from nifty_quant_lab.database.models import HistoricalPrice, Symbol, TechnicalIndicator
from nifty_quant_lab.indicators.engine import IndicatorEngine
from nifty_quant_lab.utils.logger import get_logger

logger = get_logger("indicator_service")


class IndicatorService:
    """
    Loads OHLCV from DB → Computes indicators → Saves back to DB.
    Processes symbols one at a time to keep memory footprint low.
    """

    def __init__(self):
        self.engine = IndicatorEngine()

    async def compute_and_save(
        self,
        symbol: str,
        lookback_days: int = 400,
        force: bool = False,
    ) -> bool:
        """
        Compute and persist indicators for one symbol.

        Args:
            symbol: NSE symbol string
            lookback_days: Window for indicator computation
            force: Overwrite existing records
        """
        async with get_async_session() as session:
            # Get symbol ID
            result = await session.execute(
                select(Symbol.id).where(Symbol.symbol == symbol, Symbol.exchange == "NSE")
            )
            symbol_id = result.scalar_one_or_none()
            if symbol_id is None:
                logger.warning(f"Symbol {symbol} not found in DB")
                return False

            # Load historical price data
            since = date.today() - timedelta(days=lookback_days)
            result = await session.execute(
                select(HistoricalPrice)
                .where(
                    HistoricalPrice.symbol_id == symbol_id,
                    HistoricalPrice.date >= since,
                )
                .order_by(HistoricalPrice.date.asc())
            )
            rows = result.scalars().all()

        if len(rows) < 60:
            logger.warning(f"{symbol}: Only {len(rows)} bars — skipping indicators")
            return False

        # Build DataFrame
        df = pd.DataFrame([{
            "date": r.date,
            "open": float(r.open),
            "high": float(r.high),
            "low": float(r.low),
            "close": float(r.close),
            "volume": int(r.volume),
        } for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # Compute indicators
        ind_result = self.engine.compute(df, symbol)
        if not ind_result.success:
            logger.error(f"{symbol}: {ind_result.error}")
            return False

        # Build records for DB
        records = []
        for ts, row in ind_result.df.iterrows():
            rec = {
                "symbol_id": symbol_id,
                "date": ts.date() if hasattr(ts, "date") else ts,
                "interval": "1d",
            }
            # Map all indicator columns
            col_map = {
                "ema_9": "ema_9",
                "ema_20": "ema_20",
                "ema_50": "ema_50",
                "ema_200": "ema_200",
                "sma_20": "sma_20",
                "sma_50": "sma_50",
                "sma_200": "sma_200",
                "rsi_14": "rsi_14",
                "rsi_9": "rsi_9",
                "stoch_rsi": "stoch_rsi",
                "stoch_rsi_k": "stoch_rsi_k",
                "stoch_rsi_d": "stoch_rsi_d",
                "macd_line": "macd_line",
                "macd_signal": "macd_signal",
                "macd_histogram": "macd_histogram",
                "bb_upper": "bb_upper",
                "bb_middle": "bb_middle",
                "bb_lower": "bb_lower",
                "bb_width": "bb_width",
                "bb_pct_b": "bb_pct_b",
                "atr_14": "atr_14",
                "adx_14": "adx_14",
                "adx_di_plus": "adx_di_plus",
                "adx_di_minus": "adx_di_minus",
                "supertrend": "supertrend",
                "supertrend_direction": "supertrend_direction",
                "vwap": "vwap",
                "obv": "obv",
                "volume_sma_20": "volume_sma_20",
                "volume_ratio": "volume_ratio",
                "cci_20": "cci_20",
                "ichimoku_tenkan": "ichimoku_tenkan",
                "ichimoku_kijun": "ichimoku_kijun",
                "ichimoku_senkou_a": "ichimoku_senkou_a",
                "ichimoku_senkou_b": "ichimoku_senkou_b",
                "ichimoku_chikou": "ichimoku_chikou",
                "pivot_classic": "pivot_classic",
                "pivot_r1": "pivot_r1",
                "pivot_r2": "pivot_r2",
                "pivot_r3": "pivot_r3",
                "pivot_s1": "pivot_s1",
                "pivot_s2": "pivot_s2",
                "pivot_s3": "pivot_s3",
            }
            for df_col, db_col in col_map.items():
                val = row.get(df_col)
                if val is not None and pd.notna(val):
                    rec[db_col] = float(val) if db_col != "supertrend_direction" else int(val)
                else:
                    rec[db_col] = None
            records.append(rec)

        # Bulk upsert
        async with get_async_session() as session:
            stmt = mysql_upsert(TechnicalIndicator).values(records)
            if force:
                # MySQL ON DUPLICATE KEY UPDATE — update all indicator cols
                update_cols = {
                    col: stmt.inserted[col]
                    for col in records[0].keys()
                    if col not in ("symbol_id", "date", "interval")
                }
                stmt = stmt.on_duplicate_key_update(**update_cols)
            else:
                # No-op insert (skip duplicates): update with same value
                stmt = stmt.on_duplicate_key_update(symbol_id=stmt.inserted.symbol_id)

            await session.execute(stmt)

        logger.info(f"✓ {symbol}: {len(records)} indicator rows saved")
        return True

    async def compute_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        force: bool = False,
    ) -> Dict[str, bool]:
        """Compute and save indicators for all symbols."""
        from nifty_quant_lab.config.settings import NIFTY50_SYMBOLS

        if symbols is None:
            index_symbols = ["NIFTY50", "BANKNIFTY", "FINNIFTY", "MIDCAP50"]
            symbols = index_symbols + list(NIFTY50_SYMBOLS)

        results: Dict[str, bool] = {}
        for symbol in symbols:
            results[symbol] = await self.compute_and_save(symbol, force=force)

        success = sum(1 for v in results.values() if v)
        logger.info(f"Indicator compute complete: {success}/{len(symbols)} OK")
        return results
