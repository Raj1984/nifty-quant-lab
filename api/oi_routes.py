"""
NIFTY Quant Lab — OI / PCR / Futures API Routes
==================================================
Phase 2 FastAPI router. Registered in api/app.py.

Endpoints:
  GET /api/oi/chain/{symbol}      — Live option chain with OI analysis
  GET /api/oi/pcr/{symbol}        — PCR + max pain
  GET /api/oi/pcr/history/{symbol} — PCR time series (last N hours)
  GET /api/oi/walls/{symbol}      — OI walls (S/R from option chain)
  GET /api/oi/futures/{symbol}    — Futures basis + rollover
  POST /api/oi/refresh/{symbol}   — Trigger live fetch from NSE
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from nifty_quant_lab.analytics.futures_analytics import FuturesAnalyticsEngine
from nifty_quant_lab.analytics.oi_analytics import OIAnalyticsEngine
from nifty_quant_lab.data.providers.nse_scraper import NSEOptionChainScraper
from nifty_quant_lab.signals.oi_service import OIPersistenceService
from nifty_quant_lab.utils.logger import get_logger

logger = get_logger("oi_routes")

oi_router = APIRouter(prefix="/api/oi", tags=["OI / PCR / Futures"])

# Shared instances (created once per worker)
_scraper = NSEOptionChainScraper()
_oi_engine = OIAnalyticsEngine()
_fut_engine = FuturesAnalyticsEngine()
_oi_svc = OIPersistenceService()

VALID_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}


def _validate_symbol(symbol: str) -> str:
    s = symbol.upper()
    if s not in VALID_SYMBOLS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid symbol '{symbol}'. Valid: {sorted(VALID_SYMBOLS)}"
        )
    return s


# ─────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class PCRResponse(BaseModel):
    symbol: str
    expiry: str
    spot_price: float
    pcr_oi: float
    pcr_volume: float
    signal: str
    signal_strength: str
    interpretation: str
    total_ce_oi: int
    total_pe_oi: int
    max_pain: Optional[float]
    max_pain_gap_pct: Optional[float]
    atm_strike: Optional[float]
    timestamp: str


class OIWallResponse(BaseModel):
    strike: float
    option_type: str
    oi: int
    oi_change: int
    oi_change_pct: float
    wall_type: str
    distance_from_spot_pct: float
    strength: str


class OptionRowResponse(BaseModel):
    strike: float
    expiry: str
    ce_oi: int
    ce_oi_change: int
    ce_volume: int
    ce_iv: float
    ce_ltp: float
    pe_oi: int
    pe_oi_change: int
    pe_volume: int
    pe_iv: float
    pe_ltp: float
    pcr_oi: Optional[float]


class FuturesResponse(BaseModel):
    symbol: str
    expiry: str
    spot_price: float
    futures_price: float
    basis: float
    basis_pct: float
    market_bias: str
    annualised_coc: Optional[float]
    days_to_expiry: Optional[int]
    open_interest: int
    oi_change: int
    rollover_pct: Optional[float]
    rollover_status: Optional[str]
    roll_cost: Optional[float]
    timestamp: str


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@oi_router.get("/chain/{symbol}")
async def get_option_chain(
    symbol: str,
    expiry: Optional[str] = Query(None, description="Expiry date e.g. 26-Jun-2025"),
    atm_range: int = Query(default=20, ge=5, le=60, description="Strikes above+below ATM"),
):
    """
    Fetch live option chain from NSE + full OI analysis.
    Returns ATM ± N strikes with PCR, walls, and IV skew.
    """
    sym = _validate_symbol(symbol)

    result = await _scraper.fetch_option_chain(sym, expiry)
    if not result.success:
        raise HTTPException(status_code=503, detail=f"NSE fetch failed: {result.error}")

    snapshot = result.data
    analysis = _oi_engine.analyze(snapshot)

    # Filter to ATM range
    atm_rows = snapshot.atm_rows(atm_range) if snapshot.atm_strike else snapshot.rows
    rows = [
        OptionRowResponse(
            strike=r.strike, expiry=r.expiry,
            ce_oi=r.ce_oi, ce_oi_change=r.ce_oi_change,
            ce_volume=r.ce_volume, ce_iv=r.ce_iv, ce_ltp=r.ce_ltp,
            pe_oi=r.pe_oi, pe_oi_change=r.pe_oi_change,
            pe_volume=r.pe_volume, pe_iv=r.pe_iv, pe_ltp=r.pe_ltp,
            pcr_oi=r.pcr_oi,
        )
        for r in sorted(atm_rows, key=lambda x: x.strike)
    ]

    # Save to DB in background
    import asyncio
    asyncio.create_task(_oi_svc.save_option_chain(snapshot, analysis))
    asyncio.create_task(_oi_svc.save_pcr(snapshot))

    return {
        "status": "ok",
        "symbol": sym,
        "expiry": snapshot.expiry,
        "spot_price": snapshot.spot_price,
        "atm_strike": snapshot.atm_strike,
        "timestamp": snapshot.timestamp.isoformat(),
        "pcr": {
            "oi": snapshot.pcr_oi,
            "volume": snapshot.pcr_volume,
            "signal": analysis.pcr_analysis.signal.value if analysis.pcr_analysis else None,
            "interpretation": analysis.pcr_analysis.interpretation if analysis.pcr_analysis else None,
        },
        "max_pain": snapshot.max_pain,
        "iv_skew": analysis.iv_skew,
        "key_levels": analysis.key_levels,
        "nearest_ce_wall": {
            "strike": analysis.nearest_ce_wall.strike,
            "oi": analysis.nearest_ce_wall.oi,
            "strength": analysis.nearest_ce_wall.strength,
        } if analysis.nearest_ce_wall else None,
        "nearest_pe_wall": {
            "strike": analysis.nearest_pe_wall.strike,
            "oi": analysis.nearest_pe_wall.oi,
            "strength": analysis.nearest_pe_wall.strength,
        } if analysis.nearest_pe_wall else None,
        "total_strikes": len(rows),
        "rows": [r.model_dump() for r in rows],
    }


@oi_router.get("/pcr/{symbol}", response_model=PCRResponse)
async def get_pcr(
    symbol: str,
    expiry: Optional[str] = Query(None),
):
    """PCR + max pain for a symbol. Faster than full chain — no strike rows returned."""
    sym = _validate_symbol(symbol)

    result = await _scraper.fetch_option_chain(sym, expiry)
    if not result.success:
        raise HTTPException(status_code=503, detail=f"NSE fetch failed: {result.error}")

    snapshot = result.data
    analysis = _oi_engine.analyze(snapshot)
    pcr = analysis.pcr_analysis

    import asyncio
    asyncio.create_task(_oi_svc.save_pcr(snapshot))

    return PCRResponse(
        symbol=sym,
        expiry=snapshot.expiry,
        spot_price=snapshot.spot_price,
        pcr_oi=pcr.pcr_oi,
        pcr_volume=pcr.pcr_volume,
        signal=pcr.signal.value,
        signal_strength=pcr.signal_strength,
        interpretation=pcr.interpretation,
        total_ce_oi=pcr.total_ce_oi,
        total_pe_oi=pcr.total_pe_oi,
        max_pain=pcr.max_pain,
        max_pain_gap_pct=pcr.max_pain_gap_pct,
        atm_strike=snapshot.atm_strike,
        timestamp=snapshot.timestamp.isoformat(),
    )


@oi_router.get("/pcr/history/{symbol}")
async def get_pcr_history(
    symbol: str,
    hours: int = Query(default=6, ge=1, le=24),
):
    """PCR time series from DB — shows intraday PCR trend."""
    sym = _validate_symbol(symbol)
    history = await _oi_svc.get_pcr_history(sym, hours=hours)
    return {
        "status": "ok",
        "symbol": sym,
        "hours": hours,
        "count": len(history),
        "data": history,
    }


@oi_router.get("/walls/{symbol}")
async def get_oi_walls(
    symbol: str,
    expiry: Optional[str] = Query(None),
    min_strength: str = Query(default="WEAK", description="WEAK / MODERATE / STRONG"),
):
    """OI walls — significant CE/PE OI concentrations acting as S/R."""
    sym = _validate_symbol(symbol)

    result = await _scraper.fetch_option_chain(sym, expiry)
    if not result.success:
        raise HTTPException(status_code=503, detail=f"NSE fetch failed: {result.error}")

    analysis = _oi_engine.analyze(result.data)
    strength_order = {"WEAK": 0, "MODERATE": 1, "STRONG": 2}
    min_val = strength_order.get(min_strength.upper(), 0)

    walls = [
        OIWallResponse(
            strike=w.strike, option_type=w.option_type,
            oi=w.oi, oi_change=w.oi_change, oi_change_pct=w.oi_change_pct,
            wall_type=w.wall_type,
            distance_from_spot_pct=w.distance_from_spot_pct,
            strength=w.strength,
        )
        for w in analysis.oi_walls
        if strength_order.get(w.strength, 0) >= min_val
    ]

    return {
        "status": "ok",
        "symbol": sym,
        "spot_price": result.data.spot_price,
        "timestamp": result.data.timestamp.isoformat(),
        "total_walls": len(walls),
        "ce_walls": [w.model_dump() for w in walls if w.option_type == "CE"],
        "pe_walls": [w.model_dump() for w in walls if w.option_type == "PE"],
    }


@oi_router.get("/futures/{symbol}", response_model=FuturesResponse)
async def get_futures(symbol: str):
    """
    Futures basis and rollover analysis.
    Fetches spot from yfinance, futures from NSE scraper.
    """
    sym = _validate_symbol(symbol)

    # Get spot price
    from nifty_quant_lab.data.providers.yfinance_provider import YFinanceProvider
    from datetime import date, timedelta
    provider = YFinanceProvider()
    end = date.today()
    start = end - timedelta(days=5)
    spot_result = await provider.fetch_historical(sym, start, end)
    spot = float(spot_result.data["close"].iloc[-1]) if spot_result.success and spot_result.data is not None else 0.0

    # Fetch futures from NSE derivative quote via the nse package wrapper
    try:
        raw = await _scraper.session.quote_derivative(sym)
        # Extract futures contracts
        futures_data = [
            item for item in raw.get("stocks", [])
            if item.get("metadata", {}).get("instrumentType") == "Index Futures"
        ]
        # Flatten metadata into flat dicts
        flat = [
            {
                "expiryDate": item["metadata"].get("expiryDate", ""),
                "lastPrice": item["metadata"].get("lastPrice", spot),
                "openInterest": item.get("marketDeptOrderBook", {}).get("tradeInfo", {}).get("totalBuyQuantity", 0),
                "changeinOpenInterest": 0,
                "totalTradedVolume": item.get("marketDeptOrderBook", {}).get("tradeInfo", {}).get("tradedVolume", 0),
            }
            for item in futures_data
        ]
        result = _fut_engine.analyze_from_quotes(sym, spot, flat)
    except Exception as e:
        logger.error(f"Futures fetch failed for {sym}: {e}")
        raise HTTPException(status_code=503, detail=f"Futures data unavailable: {e}")

    if not result.success or not result.near_month:
        raise HTTPException(status_code=503, detail="Futures data unavailable")

    nm = result.near_month
    import asyncio
    asyncio.create_task(_oi_svc.save_futures(result))

    return FuturesResponse(
        symbol=sym,
        expiry=nm.expiry,
        spot_price=nm.spot_price,
        futures_price=nm.futures_price,
        basis=nm.basis,
        basis_pct=nm.basis_pct,
        market_bias=nm.market_bias,
        annualised_coc=nm.annualised_cost_of_carry,
        days_to_expiry=nm.days_to_expiry,
        open_interest=nm.open_interest,
        oi_change=nm.oi_change,
        rollover_pct=result.rollover.rollover_pct if result.rollover else None,
        rollover_status=result.rollover.rollover_status if result.rollover else None,
        roll_cost=result.rollover.roll_cost if result.rollover else None,
        timestamp=result.timestamp.isoformat(),
    )


@oi_router.post("/refresh/{symbol}")
async def refresh_oi(symbol: str):
    """
    Trigger an immediate OI fetch for a symbol and persist to DB.
    Useful for the Admin panel manual refresh button.
    """
    sym = _validate_symbol(symbol)
    result = await _scraper.fetch_option_chain(sym)
    if not result.success:
        raise HTTPException(status_code=503, detail=result.error)

    snapshot = result.data
    analysis = _oi_engine.analyze(snapshot)

    oi_rows = await _oi_svc.save_option_chain(snapshot, analysis)
    await _oi_svc.save_pcr(snapshot)

    return {
        "status": "ok",
        "symbol": sym,
        "timestamp": snapshot.timestamp.isoformat(),
        "pcr_oi": snapshot.pcr_oi,
        "max_pain": snapshot.max_pain,
        "rows_saved": oi_rows,
        "signal": analysis.pcr_analysis.signal.value if analysis.pcr_analysis else None,
    }
