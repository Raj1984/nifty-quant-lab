"""
NIFTY Quant Lab - FastAPI Application
========================================
REST API server for all quant lab data and analytics.

Phase 1 endpoints:
  GET /api/nifty        — NIFTY50 quote + indicators
  GET /api/banknifty    — BANKNIFTY quote + indicators
  GET /api/scanner      — Latest scan results
  GET /api/historical   — Historical OHLCV data
  GET /api/indicators   — Technical indicators for symbol
  GET /api/sr           — Support/Resistance levels
  POST /api/scan/run    — Trigger scanner run

Health:
  GET /health           — Server health check
  GET /health/db        — Database connectivity check
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from nifty_quant_lab.config.settings import settings, NIFTY50_SYMBOLS
from nifty_quant_lab.config.scheduler import scheduler
from nifty_quant_lab.database.connection import (
    create_all_tables,
    get_db,
    check_connection,
)
from nifty_quant_lab.database.models import (
    HistoricalPrice,
    ScannerResult,
    SignalType,
    Symbol,
    TechnicalIndicator,
)
from nifty_quant_lab.utils.logger import get_logger, setup_logging

logger = get_logger("api")


# ─────────────────────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    setup_logging()
    logger.info("NIFTY Quant Lab API starting...")
    await create_all_tables()
    scheduler.setup()
    scheduler.start()
    logger.info("✓ API ready — scheduler running")
    yield
    scheduler.stop()
    logger.info("NIFTY Quant Lab API shut down.")


# ─────────────────────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="NIFTY Quant Lab API",
        description=(
            "Institutional-grade Indian market analytics platform. "
            "Phase 1: Data, Indicators, S/R, Scanner."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health_router, tags=["Health"])
    app.include_router(market_router, prefix="/api", tags=["Market Data"])
    app.include_router(scanner_router, prefix="/api", tags=["Scanner"])
    app.include_router(indicator_router, prefix="/api", tags=["Indicators"])
    app.include_router(sr_router, prefix="/api", tags=["Support/Resistance"])
    # Phase 2 — OI/PCR/Futures
    from nifty_quant_lab.api.oi_routes import oi_router
    app.include_router(oi_router, tags=["OI / PCR / Futures"])

    return app


# ─────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class OHLCVResponse(BaseModel):
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    change: Optional[float] = None
    change_pct: Optional[float] = None


class IndicatorResponse(BaseModel):
    symbol: str
    date: str
    ema_9: Optional[float]
    ema_20: Optional[float]
    ema_50: Optional[float]
    ema_200: Optional[float]
    rsi_14: Optional[float]
    macd_line: Optional[float]
    macd_signal: Optional[float]
    macd_histogram: Optional[float]
    bb_upper: Optional[float]
    bb_lower: Optional[float]
    atr_14: Optional[float]
    adx_14: Optional[float]
    supertrend: Optional[float]
    supertrend_direction: Optional[int]
    vwap: Optional[float]


class ScanResultResponse(BaseModel):
    symbol: str
    signal: str
    score: float
    close_price: float
    scan_date: str
    conditions_met: int
    ema20_above_ema50: bool
    rsi_above_55: bool
    macd_bullish_cross: bool
    price_above_supertrend: bool
    volume_above_avg: bool
    week52_breakout: bool
    rsi: Optional[float]


class ApiResponse(BaseModel):
    status: str
    data: Any
    message: str = ""
    timestamp: str = ""

    def __init__(self, **data):
        if "timestamp" not in data or not data["timestamp"]:
            data["timestamp"] = datetime.now().isoformat()
        super().__init__(**data)


# ─────────────────────────────────────────────────────────────
# HEALTH ROUTER
# ─────────────────────────────────────────────────────────────

from fastapi import APIRouter

health_router = APIRouter()


@health_router.get("/health")
async def health_check():
    return {"status": "ok", "service": "NIFTY Quant Lab", "version": "1.0.0"}


@health_router.get("/health/db")
async def db_health():
    ok = await check_connection()
    if ok:
        return {"status": "ok", "database": "connected"}
    raise HTTPException(status_code=503, detail="Database unavailable")


# ─────────────────────────────────────────────────────────────
# MARKET DATA ROUTER
# ─────────────────────────────────────────────────────────────

market_router = APIRouter()


@market_router.get("/nifty", response_model=ApiResponse)
async def get_nifty(
    days: int = Query(default=30, ge=1, le=3650, description="Number of days"),
    db: AsyncSession = Depends(get_db),
):
    """NIFTY50 historical quote."""
    return await _get_index_data("NIFTY50", days, db)


@market_router.get("/banknifty", response_model=ApiResponse)
async def get_banknifty(
    days: int = Query(default=30, ge=1, le=3650),
    db: AsyncSession = Depends(get_db),
):
    """BANKNIFTY historical quote."""
    return await _get_index_data("BANKNIFTY", days, db)


@market_router.get("/historical", response_model=ApiResponse)
async def get_historical(
    symbol: str = Query(..., description="NSE symbol e.g. RELIANCE"),
    days: int = Query(default=365, ge=1, le=3650),
    db: AsyncSession = Depends(get_db),
):
    """Historical OHLCV for any NSE symbol."""
    return await _get_index_data(symbol.upper(), days, db)


async def _get_index_data(symbol: str, days: int, db: AsyncSession) -> ApiResponse:
    """Shared logic for market data endpoints."""
    # Resolve symbol → ID
    result = await db.execute(
        select(Symbol.id, Symbol.name)
        .where(Symbol.symbol == symbol, Symbol.exchange == "NSE")
    )
    row = result.first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    symbol_id, symbol_name = row

    since = date.today() - timedelta(days=days)
    prices = await db.execute(
        select(HistoricalPrice)
        .where(
            HistoricalPrice.symbol_id == symbol_id,
            HistoricalPrice.date >= since,
        )
        .order_by(HistoricalPrice.date.asc())
    )
    rows = prices.scalars().all()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No data for '{symbol}'")

    data = []
    for i, r in enumerate(rows):
        close = float(r.close)
        prev_close = float(rows[i - 1].close) if i > 0 else close
        change = close - prev_close
        change_pct = change / prev_close * 100 if prev_close else 0
        data.append(OHLCVResponse(
            symbol=symbol,
            date=str(r.date),
            open=float(r.open),
            high=float(r.high),
            low=float(r.low),
            close=close,
            volume=int(r.volume),
            change=round(change, 2),
            change_pct=round(change_pct, 2),
        ))

    return ApiResponse(
        status="ok",
        data={
            "symbol": symbol,
            "name": symbol_name,
            "bars": len(data),
            "from": str(rows[0].date),
            "to": str(rows[-1].date),
            "latest": data[-1].model_dump() if data else None,
            "history": [d.model_dump() for d in data],
        },
    )


# ─────────────────────────────────────────────────────────────
# INDICATOR ROUTER
# ─────────────────────────────────────────────────────────────

indicator_router = APIRouter()


@indicator_router.get("/indicators", response_model=ApiResponse)
async def get_indicators(
    symbol: str = Query(..., description="NSE symbol"),
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """Technical indicators for a symbol."""
    result = await db.execute(
        select(Symbol.id).where(Symbol.symbol == symbol.upper(), Symbol.exchange == "NSE")
    )
    symbol_id = result.scalar_one_or_none()
    if not symbol_id:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    since = date.today() - timedelta(days=days)
    ind_rows = await db.execute(
        select(TechnicalIndicator)
        .where(
            TechnicalIndicator.symbol_id == symbol_id,
            TechnicalIndicator.date >= since,
            TechnicalIndicator.interval == "1d",
        )
        .order_by(TechnicalIndicator.date.desc())
        .limit(days)
    )
    rows = ind_rows.scalars().all()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No indicators for '{symbol}' — run indicator computation first",
        )

    data = [
        IndicatorResponse(
            symbol=symbol.upper(),
            date=str(r.date),
            ema_9=r.ema_9,
            ema_20=r.ema_20,
            ema_50=r.ema_50,
            ema_200=r.ema_200,
            rsi_14=r.rsi_14,
            macd_line=r.macd_line,
            macd_signal=r.macd_signal,
            macd_histogram=r.macd_histogram,
            bb_upper=r.bb_upper,
            bb_lower=r.bb_lower,
            atr_14=r.atr_14,
            adx_14=r.adx_14,
            supertrend=r.supertrend,
            supertrend_direction=r.supertrend_direction,
            vwap=r.vwap,
        )
        for r in rows
    ]

    latest = data[0].model_dump() if data else {}
    return ApiResponse(
        status="ok",
        data={
            "symbol": symbol.upper(),
            "count": len(data),
            "latest": latest,
            "history": [d.model_dump() for d in reversed(data)],
        },
    )


# ─────────────────────────────────────────────────────────────
# SCANNER ROUTER
# ─────────────────────────────────────────────────────────────

scanner_router = APIRouter()


@scanner_router.get("/scanner", response_model=ApiResponse)
async def get_scan_results(
    signal: Optional[str] = Query(None, description="Filter by signal: BUY/SELL/WATCHLIST"),
    min_score: float = Query(default=0, ge=0, le=100),
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Latest scanner results."""
    query = (
        select(ScannerResult, Symbol.symbol)
        .join(Symbol, ScannerResult.symbol_id == Symbol.id)
        .order_by(desc(ScannerResult.score))
    )

    if signal:
        try:
            sig = SignalType(signal.upper())
            query = query.where(ScannerResult.signal == sig)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid signal: {signal}")

    if min_score > 0:
        query = query.where(ScannerResult.score >= min_score)

    query = query.limit(limit)

    result = await db.execute(query)
    rows = result.all()

    data = [
        ScanResultResponse(
            symbol=sym,
            signal=r.signal.value,
            score=r.score,
            close_price=r.close_price,
            scan_date=str(r.scan_date),
            conditions_met=sum([
                r.ema20_above_ema50, r.rsi_above_55, r.macd_bullish_cross,
                r.price_above_supertrend, r.volume_above_avg, r.week52_breakout,
            ]),
            ema20_above_ema50=r.ema20_above_ema50,
            rsi_above_55=r.rsi_above_55,
            macd_bullish_cross=r.macd_bullish_cross,
            price_above_supertrend=r.price_above_supertrend,
            volume_above_avg=r.volume_above_avg,
            week52_breakout=r.week52_breakout,
            rsi=r.rsi,
        )
        for r, sym in rows
    ]

    return ApiResponse(
        status="ok",
        data={
            "count": len(data),
            "results": [d.model_dump() for d in data],
        },
    )


@scanner_router.post("/scan/run", response_model=ApiResponse)
async def trigger_scan():
    """Trigger an on-demand scanner run (async background task)."""
    import asyncio
    from nifty_quant_lab.signals.scanner import SwingScanner

    async def _run():
        scanner = SwingScanner()
        return await scanner.scan_universe()

    asyncio.create_task(_run())
    return ApiResponse(
        status="ok",
        data={"message": "Scanner started in background. Check /api/scanner for results."},
    )


# ─────────────────────────────────────────────────────────────
# S/R ROUTER
# ─────────────────────────────────────────────────────────────

sr_router = APIRouter()


@sr_router.get("/sr", response_model=ApiResponse)
async def get_support_resistance(
    symbol: str = Query(..., description="NSE symbol"),
    lookback: int = Query(default=252, ge=30, le=1500, description="Bars of data to analyse"),
    db: AsyncSession = Depends(get_db),
):
    """Support & Resistance levels for a symbol — computed live."""
    from nifty_quant_lab.analytics.support_resistance import SupportResistanceEngine

    # Load price data
    result = await db.execute(
        select(Symbol.id).where(Symbol.symbol == symbol.upper(), Symbol.exchange == "NSE")
    )
    symbol_id = result.scalar_one_or_none()
    if not symbol_id:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    since = date.today() - timedelta(days=lookback + 50)
    prices = await db.execute(
        select(HistoricalPrice)
        .where(
            HistoricalPrice.symbol_id == symbol_id,
            HistoricalPrice.date >= since,
        )
        .order_by(HistoricalPrice.date.asc())
    )
    rows = prices.scalars().all()

    if len(rows) < 30:
        raise HTTPException(status_code=422, detail=f"Insufficient data for '{symbol}'")

    import pandas as pd
    df = pd.DataFrame([{
        "date": r.date,
        "open": float(r.open),
        "high": float(r.high),
        "low": float(r.low),
        "close": float(r.close),
        "volume": int(r.volume),
    } for r in rows[-lookback:]])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    engine = SupportResistanceEngine()
    analysis = engine.analyze(df, symbol=symbol.upper())

    if not analysis.success:
        raise HTTPException(status_code=422, detail=analysis.error)

    return ApiResponse(
        status="ok",
        data={
            "symbol": symbol.upper(),
            "current_price": analysis.current_price,
            "nearest_support": analysis.nearest_support,
            "nearest_resistance": analysis.nearest_resistance,
            "risk_reward_estimate": analysis.risk_reward_estimate,
            "strong_supports": [
                {"price": l.price, "method": l.method, "strength": l.strength, "notes": l.notes}
                for l in analysis.strong_supports
            ],
            "weak_supports": [
                {"price": l.price, "method": l.method, "strength": l.strength}
                for l in analysis.weak_supports
            ],
            "resistances": [
                {"price": l.price, "method": l.method, "strength": l.strength, "notes": l.notes}
                for l in analysis.resistances
            ],
            "breakout_levels": [
                {"price": l.price, "method": l.method, "strength": l.strength}
                for l in analysis.breakout_levels
            ],
            "total_levels": len(analysis.levels),
        },
    )


# ─────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "nifty_quant_lab.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
