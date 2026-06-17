"""
NIFTY Quant Lab - Swing Scanner
==================================
Scans entire NIFTY universe for swing trade opportunities.

Buy conditions (6 criteria):
  1. EMA20 > EMA50  (bullish alignment)
  2. RSI > 55  (momentum confirmation)
  3. MACD bullish crossover  (entry trigger)
  4. Price above Supertrend  (trend filter)
  5. Volume > 20-day average (institutional confirmation)
  6. 52-week breakout  (momentum expansion)

Signals:
  BUY       ≥ 4 conditions met, score ≥ 70
  WATCHLIST ≥ 3 conditions met, score ≥ 50
  SELL      < 2 conditions, or bearish reversal detected

Scoring: Each condition contributes 0–20 points.
Weighted sum = final score (0–100).

Architecture: qlib Alpha generation pipeline pattern.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from nifty_quant_lab.config.settings import NIFTY50_SYMBOLS, settings
from nifty_quant_lab.data.providers.yfinance_provider import YFinanceProvider
from nifty_quant_lab.database.models import SignalType
from nifty_quant_lab.indicators.engine import IndicatorEngine
from nifty_quant_lab.analytics.volatility import VolatilityEngine
from nifty_quant_lab.utils.logger import get_logger

logger = get_logger("scanner")


# ─────────────────────────────────────────────────────────────
# CONDITION WEIGHTS (must sum to 100)
# ─────────────────────────────────────────────────────────────
CONDITION_WEIGHTS = {
    "ema20_above_ema50": 15,        # Trend alignment
    "rsi_above_55": 20,             # Momentum
    "macd_bullish_cross": 20,       # Entry trigger
    "price_above_supertrend": 20,   # Trend filter
    "volume_above_avg": 15,         # Volume confirmation
    "week52_breakout": 10,          # Breakout
}

assert sum(CONDITION_WEIGHTS.values()) == 100, "Weights must sum to 100"

BUY_THRESHOLD = 65
WATCHLIST_THRESHOLD = 45


@dataclass
class ScanResult:
    """Structured scan result for one symbol."""
    symbol: str
    signal: SignalType
    score: float
    close_price: float
    scan_date: date = field(default_factory=date.today)

    # Conditions
    ema20_above_ema50: bool = False
    rsi_above_55: bool = False
    macd_bullish_cross: bool = False
    price_above_supertrend: bool = False
    volume_above_avg: bool = False
    week52_breakout: bool = False

    # Snapshot values
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    rsi: Optional[float] = None
    volume: Optional[int] = None
    volume_avg_20: Optional[float] = None
    week52_high: Optional[float] = None
    atr: Optional[float] = None
    supertrend: Optional[float] = None

    # Stop loss / target
    suggested_sl: Optional[float] = None
    suggested_target: Optional[float] = None
    risk_reward: Optional[float] = None

    # Volatility context (gs-quant inspired — see analytics/volatility.py)
    # Non-gating: does not block a BUY signal, but adjusts score slightly
    # and is surfaced in the dashboard so the trader sees the vol backdrop.
    vol_regime: Optional[str] = None       # LOW / NORMAL / HIGH / EXTREME
    vol_trend: Optional[str] = None        # EXPANDING / CONTRACTING / STABLE
    realized_vol: Optional[float] = None   # annualized %, primary window

    notes: str = ""

    @property
    def conditions_met(self) -> int:
        return sum([
            self.ema20_above_ema50,
            self.rsi_above_55,
            self.macd_bullish_cross,
            self.price_above_supertrend,
            self.volume_above_avg,
            self.week52_breakout,
        ])

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "signal": self.signal.value,
            "score": round(self.score, 1),
            "close": self.close_price,
            "conditions_met": self.conditions_met,
            "ema_cross": self.ema20_above_ema50,
            "rsi_ok": self.rsi_above_55,
            "macd_cross": self.macd_bullish_cross,
            "supertrend": self.price_above_supertrend,
            "volume_ok": self.volume_above_avg,
            "breakout": self.week52_breakout,
            "rsi": self.rsi,
            "ema_20": self.ema_20,
            "ema_50": self.ema_50,
            "suggested_sl": self.suggested_sl,
            "suggested_target": self.suggested_target,
            "risk_reward": self.risk_reward,
            "vol_regime": self.vol_regime,
            "vol_trend": self.vol_trend,
            "realized_vol": self.realized_vol,
            "notes": self.notes,
        }


@dataclass
class ScanSession:
    """Full scanner run results."""
    scan_date: date
    total_scanned: int
    results: List[ScanResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def buy_signals(self) -> List[ScanResult]:
        return [r for r in self.results if r.signal == SignalType.BUY]

    @property
    def watchlist_signals(self) -> List[ScanResult]:
        return [r for r in self.results if r.signal == SignalType.WATCHLIST]

    @property
    def sell_signals(self) -> List[ScanResult]:
        return [r for r in self.results if r.signal == SignalType.SELL]

    def top_buys(self, n: int = 10) -> List[ScanResult]:
        return sorted(self.buy_signals, key=lambda x: x.score, reverse=True)[:n]

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.results]).sort_values(
            "score", ascending=False
        ).reset_index(drop=True)


class SwingScanner:
    """
    NSE swing trade scanner.

    Uses the IndicatorEngine for computation, YFinanceProvider for data,
    and produces fully structured ScanResult objects.

    Scan flow (qlib alpha generation pattern):
        symbols → fetch OHLCV → compute indicators →
        evaluate conditions → score → classify → output
    """

    def __init__(
        self,
        provider: Optional[YFinanceProvider] = None,
        indicator_engine: Optional[IndicatorEngine] = None,
    ):
        self.provider = provider or YFinanceProvider(max_workers=8)
        self.engine = indicator_engine or IndicatorEngine()
        self.vol_engine = VolatilityEngine()

    async def scan_universe(
        self,
        symbols: Optional[List[str]] = None,
        lookback_days: int = 365,
        batch_size: int = 20,
    ) -> ScanSession:
        """
        Scan the full NIFTY universe.

        Args:
            symbols: Override default NIFTY50 list
            lookback_days: Historical data window
            batch_size: Parallel fetch batch size
        """
        import time
        t0 = time.time()

        if symbols is None:
            symbols = list(NIFTY50_SYMBOLS)

        scan_date = date.today()
        end = scan_date
        start = end - timedelta(days=lookback_days)

        logger.info(f"Scanner: {len(symbols)} symbols | {start} → {end}")
        all_results: List[ScanResult] = []

        # Batch fetch
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            fetch_map = await self.provider.fetch_multiple(batch, start, end)

            for symbol, fetch_result in fetch_map.items():
                if not fetch_result.success or fetch_result.data is None or fetch_result.data.empty:
                    logger.warning(f"✗ {symbol}: {fetch_result.error}")
                    continue

                scan_result = self._evaluate_symbol(symbol, fetch_result.data, scan_date)
                if scan_result is not None:
                    all_results.append(scan_result)

            await asyncio.sleep(1)

        session = ScanSession(
            scan_date=scan_date,
            total_scanned=len(all_results),
            results=all_results,
            elapsed_seconds=time.time() - t0,
        )

        logger.info(
            f"Scan complete: {session.total_scanned} scanned | "
            f"{len(session.buy_signals)} BUY | "
            f"{len(session.watchlist_signals)} WATCHLIST | "
            f"⏱ {session.elapsed_seconds:.1f}s"
        )
        return session

    def _evaluate_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        scan_date: date,
    ) -> Optional[ScanResult]:
        """
        Evaluate all conditions for a single symbol.
        Returns ScanResult or None on failure.
        """
        try:
            ind_result = self.engine.compute(df, symbol, min_bars=60)
            if not ind_result.success:
                return None

            idf = ind_result.df
            latest = idf.iloc[-1]
            prev = idf.iloc[-2]

            close = float(latest["close"])
            ema_20 = float(latest.get("ema_20", np.nan))
            ema_50 = float(latest.get("ema_50", np.nan))
            rsi = float(latest.get("rsi_14", np.nan))
            macd_hist_now = float(latest.get("macd_histogram", np.nan))
            macd_hist_prev = float(prev.get("macd_histogram", np.nan))
            st_direction = float(latest.get("supertrend_direction", 0))
            volume = float(latest.get("volume", 0))
            vol_avg = float(latest.get("volume_sma_20", 1))
            atr = float(latest.get("atr_14", 0))

            # 52-week high
            week52_high = float(idf["high"].tail(252).max())

            # ── Evaluate 6 conditions
            c1 = bool(ema_20 > ema_50) if not (np.isnan(ema_20) or np.isnan(ema_50)) else False
            c2 = bool(rsi > 55) if not np.isnan(rsi) else False
            c3 = bool(macd_hist_now > 0 and macd_hist_prev <= 0) if not (
                np.isnan(macd_hist_now) or np.isnan(macd_hist_prev)
            ) else False
            c4 = bool(st_direction == 1)
            c5 = bool(volume > vol_avg * 1.2) if vol_avg > 0 else False
            c6 = bool(close >= week52_high * 0.98)  # Within 2% of 52w high

            # ── Score
            score = 0.0
            if c1:
                score += CONDITION_WEIGHTS["ema20_above_ema50"]
            if c2:
                score += CONDITION_WEIGHTS["rsi_above_55"]
            if c3:
                score += CONDITION_WEIGHTS["macd_bullish_cross"]
            if c4:
                score += CONDITION_WEIGHTS["price_above_supertrend"]
            if c5:
                score += CONDITION_WEIGHTS["volume_above_avg"]
            if c6:
                score += CONDITION_WEIGHTS["week52_breakout"]

            # Partial RSI score (higher RSI = more points, capped at weight)
            if not np.isnan(rsi) and rsi > 45:
                bonus = min((rsi - 45) / 30, 1.0) * 5  # Up to 5 bonus pts
                score = min(score + bonus, 100.0)

            # ── Volatility regime context (gs-quant inspired, non-gating)
            # EXTREME regime trims a few points — signal quality degrades in
            # panic conditions. LOW+EXPANDING gets a small boost — compression
            # before expansion is exactly the setup swing traders want to catch
            # early. Neither case blocks or forces a signal on its own.
            vol_result = self.vol_engine.quick_regime(idf, symbol)
            if vol_result is not None:
                if vol_result.regime == "EXTREME":
                    score = max(score - 8, 0.0)
                elif vol_result.regime == "LOW" and vol_result.trend == "EXPANDING":
                    score = min(score + 5, 100.0)

            # ── Signal classification
            conditions_met = sum([c1, c2, c3, c4, c5, c6])
            if score >= BUY_THRESHOLD and conditions_met >= 4:
                signal = SignalType.BUY
            elif score >= WATCHLIST_THRESHOLD and conditions_met >= 3:
                signal = SignalType.WATCHLIST
            elif conditions_met <= 1:
                signal = SignalType.SELL
            else:
                signal = SignalType.HOLD

            # ── SL / Target (ATR-based, 1:2 R:R minimum)
            suggested_sl = round(close - 1.5 * atr, 2) if atr > 0 else None
            suggested_target = round(close + 3.0 * atr, 2) if atr > 0 else None
            rr = None
            if suggested_sl and suggested_target and close > suggested_sl:
                rr = round((suggested_target - close) / (close - suggested_sl), 2)

            notes_parts = []
            if c1: notes_parts.append("EMA trend✓")
            if c2: notes_parts.append(f"RSI={rsi:.0f}✓")
            if c3: notes_parts.append("MACD cross✓")
            if c4: notes_parts.append("Supertrend✓")
            if c5: notes_parts.append("Vol spike✓")
            if c6: notes_parts.append("52W high✓")
            if vol_result is not None:
                notes_parts.append(f"VolRegime={vol_result.regime}/{vol_result.trend}")

            return ScanResult(
                symbol=symbol,
                signal=signal,
                score=round(score, 1),
                close_price=round(close, 2),
                scan_date=scan_date,
                vol_regime=vol_result.regime if vol_result else None,
                vol_trend=vol_result.trend if vol_result else None,
                realized_vol=vol_result.current_vol if vol_result else None,
                ema20_above_ema50=c1,
                rsi_above_55=c2,
                macd_bullish_cross=c3,
                price_above_supertrend=c4,
                volume_above_avg=c5,
                week52_breakout=c6,
                ema_20=round(ema_20, 2) if not np.isnan(ema_20) else None,
                ema_50=round(ema_50, 2) if not np.isnan(ema_50) else None,
                rsi=round(rsi, 1) if not np.isnan(rsi) else None,
                volume=int(volume),
                volume_avg_20=round(vol_avg, 0) if vol_avg else None,
                week52_high=round(week52_high, 2),
                atr=round(atr, 2),
                suggested_sl=suggested_sl,
                suggested_target=suggested_target,
                risk_reward=rr,
                notes=" | ".join(notes_parts),
            )

        except Exception as e:
            logger.error(f"✗ {symbol} scan eval failed: {e}", exc_info=True)
            return None

    async def scan_single(self, symbol: str, lookback_days: int = 365) -> Optional[ScanResult]:
        """Quick scan for a single symbol — used by Telegram /scan command."""
        end = date.today()
        start = end - timedelta(days=lookback_days)
        result = await self.provider.fetch_historical(symbol, start, end)
        if not result.success or result.data is None:
            return None
        return self._evaluate_symbol(symbol, result.data, end)

    def format_signal_summary(self, session: ScanSession) -> str:
        """Format scan results into Telegram-ready text."""
        buys = session.top_buys(10)
        lines = [
            f"📊 **NIFTY Swing Scanner** — {session.scan_date}",
            f"Scanned: {session.total_scanned} | ✅ BUY: {len(session.buy_signals)} | "
            f"👁 WATCHLIST: {len(session.watchlist_signals)}",
            "",
            "🔥 **Top BUY Signals:**",
        ]
        for r in buys:
            lines.append(
                f"  • **{r.symbol}** — Score: {r.score:.0f} | "
                f"₹{r.close_price} | Conditions: {r.conditions_met}/6"
            )
            if r.suggested_sl and r.suggested_target:
                lines.append(
                    f"    SL: ₹{r.suggested_sl} | Target: ₹{r.suggested_target} | R:R {r.risk_reward}"
                )
        if not buys:
            lines.append("  No strong BUY setups today.")

        watchlist = session.watchlist_signals[:5]
        if watchlist:
            lines.extend(["", "👁 **Watchlist:**"])
            for r in watchlist:
                lines.append(f"  • {r.symbol} — Score: {r.score:.0f}")

        lines.append(f"\n⏱ Scan completed in {session.elapsed_seconds:.1f}s")
        return "\n".join(lines)
