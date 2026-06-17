"""
NIFTY Quant Lab — Futures Analytics
======================================
Tracks NSE futures basis, cost of carry, and rollover data.

Basis = Futures Price - Spot Price
Cost of Carry = (Basis / Spot) × (365 / days_to_expiry) × 100

Signals:
  Basis > 0  → Premium  (bullish — longs willing to pay more)
  Basis < 0  → Discount (bearish — shorts dominating)
  Basis = 0  → Flat

Data sources:
  Spot:    yfinance (^NSEI, ^NSEBANK)
  Futures: NSE scraper via quote-derivative endpoint
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from nifty_quant_lab.utils.logger import get_logger

logger = get_logger("futures_analytics")


@dataclass
class FuturesQuote:
    """Single futures contract quote."""
    symbol: str
    expiry: str
    expiry_date: Optional[date]
    spot_price: float
    futures_price: float
    open_interest: int
    oi_change: int
    volume: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def basis(self) -> float:
        return round(self.futures_price - self.spot_price, 2)

    @property
    def basis_pct(self) -> float:
        if self.spot_price == 0:
            return 0.0
        return round(self.basis / self.spot_price * 100, 4)

    @property
    def days_to_expiry(self) -> Optional[int]:
        if self.expiry_date:
            return max((self.expiry_date - date.today()).days, 0)
        return None

    @property
    def annualised_cost_of_carry(self) -> Optional[float]:
        """Annualised cost of carry in %."""
        dte = self.days_to_expiry
        if dte and dte > 0 and self.spot_price > 0:
            return round(self.basis_pct * 365 / dte, 2)
        return None

    @property
    def market_bias(self) -> str:
        if self.basis_pct > 0.15:
            return "BULLISH"
        elif self.basis_pct < -0.15:
            return "BEARISH"
        return "NEUTRAL"


@dataclass
class RolloverData:
    """Rollover statistics as current expiry approaches expiry."""
    symbol: str
    current_expiry: str
    next_expiry: str
    rollover_pct: float           # % of OI shifted to next expiry
    current_oi: int
    next_oi: int
    total_oi: int
    roll_cost: Optional[float]    # cost of rolling in points
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def rollover_status(self) -> str:
        if self.rollover_pct >= 70:
            return "HIGH"
        elif self.rollover_pct >= 40:
            return "MODERATE"
        return "LOW"


@dataclass
class FuturesAnalysisResult:
    """Complete futures analysis for a symbol."""
    symbol: str
    timestamp: datetime
    near_month: Optional[FuturesQuote] = None
    next_month: Optional[FuturesQuote] = None
    rollover: Optional[RolloverData] = None
    success: bool = True
    error: Optional[str] = None

    @property
    def basis(self) -> Optional[float]:
        return self.near_month.basis if self.near_month else None

    @property
    def basis_pct(self) -> Optional[float]:
        return self.near_month.basis_pct if self.near_month else None

    @property
    def market_bias(self) -> str:
        return self.near_month.market_bias if self.near_month else "UNKNOWN"


class FuturesAnalyticsEngine:
    """
    Computes basis, cost of carry, and rollover analytics.

    Data flows in from the NSE scraper or Zerodha KiteConnect.
    This engine is purely computational — no I/O.
    """

    def analyze_from_quotes(
        self,
        symbol: str,
        spot: float,
        futures_data: List[Dict],
    ) -> FuturesAnalysisResult:
        """
        Build FuturesAnalysisResult from raw quote dicts.

        Args:
            symbol: e.g. "NIFTY"
            spot: Current spot price
            futures_data: List of dicts from NSE API, each with:
                {expiry, lastPrice, openInterest, changeinOpenInterest, totalTradedVolume}
        """
        if not futures_data:
            return FuturesAnalysisResult(
                symbol=symbol, timestamp=datetime.now(),
                success=False, error="No futures data"
            )

        try:
            quotes: List[FuturesQuote] = []
            for fd in futures_data:
                expiry_str = fd.get("expiryDate", "")
                expiry_date = self._parse_expiry(expiry_str)
                quote = FuturesQuote(
                    symbol=symbol,
                    expiry=expiry_str,
                    expiry_date=expiry_date,
                    spot_price=spot,
                    futures_price=float(fd.get("lastPrice", spot)),
                    open_interest=int(fd.get("openInterest", 0)),
                    oi_change=int(fd.get("changeinOpenInterest", 0)),
                    volume=int(fd.get("totalTradedVolume", 0)),
                )
                quotes.append(quote)

            # Sort by expiry (nearest first)
            quotes.sort(key=lambda q: q.expiry_date or date.max)

            result = FuturesAnalysisResult(symbol=symbol, timestamp=datetime.now())
            if quotes:
                result.near_month = quotes[0]
            if len(quotes) >= 2:
                result.next_month = quotes[1]
                result.rollover = self._compute_rollover(quotes[0], quotes[1])

            logger.info(
                f"✓ Futures {symbol}: basis={result.basis:+.1f} "
                f"({result.basis_pct:+.3f}%) [{result.market_bias}]"
                if result.basis is not None else f"✓ Futures {symbol}: no data"
            )
            return result

        except Exception as e:
            logger.error(f"Futures analysis failed for {symbol}: {e}", exc_info=True)
            return FuturesAnalysisResult(
                symbol=symbol, timestamp=datetime.now(),
                success=False, error=str(e)
            )

    def _compute_rollover(
        self,
        near: FuturesQuote,
        next_: FuturesQuote,
    ) -> RolloverData:
        """Compute rollover % = next OI / (near + next OI)."""
        total = near.open_interest + next_.open_interest
        rollover_pct = round(next_.open_interest / total * 100, 1) if total > 0 else 0.0

        roll_cost = round(next_.futures_price - near.futures_price, 2) if (
            near.futures_price > 0 and next_.futures_price > 0
        ) else None

        return RolloverData(
            symbol=near.symbol,
            current_expiry=near.expiry,
            next_expiry=next_.expiry,
            rollover_pct=rollover_pct,
            current_oi=near.open_interest,
            next_oi=next_.open_interest,
            total_oi=total,
            roll_cost=roll_cost,
        )

    @staticmethod
    def _parse_expiry(expiry_str: str) -> Optional[date]:
        """Parse NSE expiry date string e.g. '26-Jun-2025'."""
        for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(expiry_str, fmt).date()
            except (ValueError, TypeError):
                continue
        return None

    def format_telegram_alert(self, result: FuturesAnalysisResult) -> str:
        """Format futures analysis as Telegram message."""
        if not result.success or not result.near_month:
            return f"❌ Futures data unavailable for {result.symbol}"

        nm = result.near_month
        bias_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(nm.market_bias, "⚪")
        lines = [
            f"{bias_emoji} **Futures — {result.symbol}** [{nm.expiry}]",
            f"Spot: ₹{nm.spot_price:,.2f} | Futures: ₹{nm.futures_price:,.2f}",
            f"Basis: {nm.basis:+.2f} ({nm.basis_pct:+.3f}%) → {nm.market_bias}",
        ]
        if nm.annualised_cost_of_carry is not None:
            lines.append(f"Cost of Carry: {nm.annualised_cost_of_carry:.2f}% p.a.")
        if nm.days_to_expiry is not None:
            lines.append(f"Days to Expiry: {nm.days_to_expiry}")
        if result.rollover:
            r = result.rollover
            lines.append(
                f"📦 Rollover: {r.rollover_pct:.1f}% [{r.rollover_status}]"
                + (f" | Roll cost: {r.roll_cost:+.1f} pts" if r.roll_cost else "")
            )
        lines.append(f"\n🕐 {result.timestamp.strftime('%H:%M IST')}")
        return "\n".join(lines)
