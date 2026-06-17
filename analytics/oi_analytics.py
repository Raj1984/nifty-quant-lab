"""
NIFTY Quant Lab — OI Analytics Engine
========================================
Computes institutional-grade OI and PCR analytics from raw option chain data.

Analyses:
  1. PCR interpretation  (Bullish / Bearish / Neutral thresholds)
  2. OI signal classification  (Long Buildup / Short Buildup / Short Covering / Long Unwinding)
  3. Max Pain vs Spot divergence
  4. Strike-level OI concentration (support / resistance walls)
  5. IV Skew analysis
  6. OI change momentum (buildup velocity)

PCR thresholds (NSE empirical):
  PCR > 1.3  → Oversold / Bullish contrarian
  PCR 0.8–1.3 → Neutral
  PCR < 0.8  → Overbought / Bearish contrarian
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from nifty_quant_lab.data.providers.nse_scraper import OptionChainSnapshot, OptionRow
from nifty_quant_lab.database.models import OISignal, PCRSignal
from nifty_quant_lab.utils.logger import get_logger

logger = get_logger("oi_analytics")

# ── PCR thresholds
PCR_BULLISH_THRESHOLD  = 1.3   # PCR above → contrarian bullish
PCR_BEARISH_THRESHOLD  = 0.8   # PCR below → contrarian bearish
PCR_EXTREME_BULL       = 1.5   # extreme fear → strong bullish signal
PCR_EXTREME_BEAR       = 0.6   # extreme greed → strong bearish signal

# ── OI signal thresholds
OI_CHANGE_SIGNIFICANT  = 10.0  # % OI change to be considered significant
PRICE_CHANGE_UP        = 0.3   # % price change to classify as "price up"
PRICE_CHANGE_DOWN      = -0.3  # % price change to classify as "price down"

# ── OI wall thresholds (strikes with unusually high OI)
OI_WALL_PERCENTILE     = 80    # top 20% of strikes by OI = walls


@dataclass
class PCRAnalysis:
    """Full PCR analysis output."""
    symbol: str
    timestamp: datetime
    pcr_oi: float
    pcr_volume: float
    signal: PCRSignal
    signal_strength: str          # WEAK / MODERATE / STRONG / EXTREME
    interpretation: str
    total_ce_oi: int
    total_pe_oi: int
    total_ce_volume: int
    total_pe_volume: int
    max_pain: Optional[float]
    spot_price: float
    max_pain_gap_pct: Optional[float] = None   # (spot - max_pain) / max_pain * 100
    pcr_trend: Optional[str] = None            # RISING / FALLING / FLAT (needs history)


@dataclass
class OIWall:
    """Significant OI concentration at a strike — acts as S/R."""
    strike: float
    option_type: str        # CE or PE
    oi: int
    oi_change: int
    oi_change_pct: float
    wall_type: str          # RESISTANCE (CE wall) or SUPPORT (PE wall)
    distance_from_spot_pct: float
    strength: str           # STRONG / MODERATE / WEAK


@dataclass
class StrikeOISignal:
    """OI signal for a specific strike."""
    strike: float
    option_type: str
    signal: OISignal
    price_change_pct: float
    oi_change_pct: float
    reasoning: str


@dataclass
class OIAnalysisResult:
    """Complete OI analysis for one symbol snapshot."""
    symbol: str
    expiry: str
    timestamp: datetime
    spot_price: float

    pcr_analysis: Optional[PCRAnalysis] = None
    oi_walls: List[OIWall] = field(default_factory=list)
    strike_signals: List[StrikeOISignal] = field(default_factory=list)
    nearest_ce_wall: Optional[OIWall] = None   # resistance
    nearest_pe_wall: Optional[OIWall] = None   # support
    iv_skew: Optional[float] = None            # put_iv - call_iv at ATM
    atm_strike: Optional[float] = None
    key_levels: Dict[str, float] = field(default_factory=dict)

    success: bool = True
    error: Optional[str] = None


class OIAnalyticsEngine:
    """
    Computes all OI analytics from an OptionChainSnapshot.

    Usage:
        engine = OIAnalyticsEngine()
        result = engine.analyze(snapshot)
    """

    def analyze(self, snapshot: OptionChainSnapshot) -> OIAnalysisResult:
        """Full OI analysis pipeline."""
        if not snapshot or not snapshot.rows:
            return OIAnalysisResult(
                symbol=snapshot.symbol if snapshot else "UNKNOWN",
                expiry="", timestamp=datetime.now(), spot_price=0,
                success=False, error="Empty snapshot",
            )

        try:
            result = OIAnalysisResult(
                symbol=snapshot.symbol,
                expiry=snapshot.expiry,
                timestamp=snapshot.timestamp,
                spot_price=snapshot.spot_price,
                atm_strike=snapshot.atm_strike,
            )

            result.pcr_analysis = self._compute_pcr(snapshot)
            result.oi_walls = self._find_oi_walls(snapshot)
            result.strike_signals = self._classify_strike_signals(snapshot)
            result.iv_skew = self._compute_iv_skew(snapshot)
            result.key_levels = self._build_key_levels(snapshot, result.oi_walls)

            # Nearest CE wall above spot = resistance
            ce_walls_above = [
                w for w in result.oi_walls
                if w.option_type == "CE" and w.strike > snapshot.spot_price
            ]
            if ce_walls_above:
                result.nearest_ce_wall = min(ce_walls_above, key=lambda w: w.strike)

            # Nearest PE wall below spot = support
            pe_walls_below = [
                w for w in result.oi_walls
                if w.option_type == "PE" and w.strike < snapshot.spot_price
            ]
            if pe_walls_below:
                result.nearest_pe_wall = max(pe_walls_below, key=lambda w: w.strike)

            logger.info(
                f"✓ OI analysis {snapshot.symbol}: "
                f"PCR={snapshot.pcr_oi:.2f} [{result.pcr_analysis.signal.value}] | "
                f"CE wall={result.nearest_ce_wall.strike if result.nearest_ce_wall else 'N/A'} | "
                f"PE wall={result.nearest_pe_wall.strike if result.nearest_pe_wall else 'N/A'}"
            )
            return result

        except Exception as e:
            logger.error(f"OI analysis failed for {snapshot.symbol}: {e}", exc_info=True)
            return OIAnalysisResult(
                symbol=snapshot.symbol, expiry=snapshot.expiry,
                timestamp=snapshot.timestamp, spot_price=snapshot.spot_price,
                success=False, error=str(e),
            )

    # ─────────────────────────────────────────────────────────
    # PCR ANALYSIS
    # ─────────────────────────────────────────────────────────

    def _compute_pcr(self, snapshot: OptionChainSnapshot) -> PCRAnalysis:
        pcr = snapshot.pcr_oi

        # Signal classification
        if pcr >= PCR_EXTREME_BULL:
            signal = PCRSignal.BULLISH
            strength = "EXTREME"
            interp = f"PCR {pcr:.2f} — Extreme fear, strong contrarian BUY signal"
        elif pcr >= PCR_BULLISH_THRESHOLD:
            signal = PCRSignal.BULLISH
            strength = "STRONG"
            interp = f"PCR {pcr:.2f} — Elevated put buying, bullish contrarian signal"
        elif pcr <= PCR_EXTREME_BEAR:
            signal = PCRSignal.BEARISH
            strength = "EXTREME"
            interp = f"PCR {pcr:.2f} — Extreme greed, strong contrarian SELL signal"
        elif pcr <= PCR_BEARISH_THRESHOLD:
            signal = PCRSignal.BEARISH
            strength = "STRONG"
            interp = f"PCR {pcr:.2f} — Low put buying, bearish contrarian signal"
        else:
            signal = PCRSignal.NEUTRAL
            strength = "WEAK"
            interp = f"PCR {pcr:.2f} — Neutral zone (0.8–1.3), no directional bias"

        # Max pain gap
        max_pain_gap = None
        if snapshot.max_pain and snapshot.spot_price:
            max_pain_gap = round(
                (snapshot.spot_price - snapshot.max_pain) / snapshot.max_pain * 100, 2
            )

        return PCRAnalysis(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            pcr_oi=pcr,
            pcr_volume=snapshot.pcr_volume,
            signal=signal,
            signal_strength=strength,
            interpretation=interp,
            total_ce_oi=snapshot.total_ce_oi,
            total_pe_oi=snapshot.total_pe_oi,
            total_ce_volume=snapshot.total_ce_volume,
            total_pe_volume=snapshot.total_pe_volume,
            max_pain=snapshot.max_pain,
            spot_price=snapshot.spot_price,
            max_pain_gap_pct=max_pain_gap,
        )

    # ─────────────────────────────────────────────────────────
    # OI WALLS
    # ─────────────────────────────────────────────────────────

    def _find_oi_walls(
        self,
        snapshot: OptionChainSnapshot,
        percentile: int = OI_WALL_PERCENTILE,
    ) -> List[OIWall]:
        """Find strikes with unusually high OI — these act as S/R walls."""
        if not snapshot.rows:
            return []

        spot = snapshot.spot_price
        ce_ois = sorted([r.ce_oi for r in snapshot.rows if r.ce_oi > 0])
        pe_ois = sorted([r.pe_oi for r in snapshot.rows if r.pe_oi > 0])

        if not ce_ois or not pe_ois:
            return []

        # Percentile thresholds
        ce_threshold = ce_ois[int(len(ce_ois) * percentile / 100)]
        pe_threshold = pe_ois[int(len(pe_ois) * percentile / 100)]

        walls: List[OIWall] = []
        for row in snapshot.rows:
            # CE wall (resistance)
            if row.ce_oi >= ce_threshold and row.ce_oi > 0:
                dist_pct = round((row.strike - spot) / spot * 100, 2)
                max_ce = ce_ois[-1] if ce_ois else 1
                oi_change_pct = round(row.ce_oi_change / row.ce_oi * 100, 1) if row.ce_oi else 0
                strength = (
                    "STRONG" if row.ce_oi >= ce_ois[int(len(ce_ois) * 0.95)]
                    else "MODERATE" if row.ce_oi >= ce_ois[int(len(ce_ois) * 0.85)]
                    else "WEAK"
                )
                walls.append(OIWall(
                    strike=row.strike,
                    option_type="CE",
                    oi=row.ce_oi,
                    oi_change=row.ce_oi_change,
                    oi_change_pct=oi_change_pct,
                    wall_type="RESISTANCE",
                    distance_from_spot_pct=dist_pct,
                    strength=strength,
                ))

            # PE wall (support)
            if row.pe_oi >= pe_threshold and row.pe_oi > 0:
                dist_pct = round((row.strike - spot) / spot * 100, 2)
                oi_change_pct = round(row.pe_oi_change / row.pe_oi * 100, 1) if row.pe_oi else 0
                strength = (
                    "STRONG" if row.pe_oi >= pe_ois[int(len(pe_ois) * 0.95)]
                    else "MODERATE" if row.pe_oi >= pe_ois[int(len(pe_ois) * 0.85)]
                    else "WEAK"
                )
                walls.append(OIWall(
                    strike=row.strike,
                    option_type="PE",
                    oi=row.pe_oi,
                    oi_change=row.pe_oi_change,
                    oi_change_pct=oi_change_pct,
                    wall_type="SUPPORT",
                    distance_from_spot_pct=dist_pct,
                    strength=strength,
                ))

        return sorted(walls, key=lambda w: w.strike)

    # ─────────────────────────────────────────────────────────
    # OI SIGNAL CLASSIFICATION
    # ─────────────────────────────────────────────────────────

    def _classify_strike_signals(
        self,
        snapshot: OptionChainSnapshot,
    ) -> List[StrikeOISignal]:
        """
        Classify each strike's OI action into one of 4 signals.

        Long Buildup:    price ↑, OI ↑  → fresh longs added
        Short Buildup:   price ↓, OI ↑  → fresh shorts added
        Short Covering:  price ↑, OI ↓  → shorts covering
        Long Unwinding:  price ↓, OI ↓  → longs exiting
        """
        signals: List[StrikeOISignal] = []

        for row in snapshot.rows:
            for opt_type, oi, oi_change, ltp in [
                ("CE", row.ce_oi, row.ce_oi_change, row.ce_ltp),
                ("PE", row.pe_oi, row.pe_oi_change, row.pe_ltp),
            ]:
                if oi == 0:
                    continue

                oi_chg_pct = oi_change / oi * 100 if oi > 0 else 0
                if abs(oi_chg_pct) < OI_CHANGE_SIGNIFICANT:
                    continue  # Skip insignificant changes

                # Approximate price change from strike distance to spot
                price_chg_pct = (snapshot.spot_price - row.strike) / row.strike * 100

                oi_up = oi_chg_pct > 0
                price_up = price_chg_pct > PRICE_CHANGE_UP
                price_dn = price_chg_pct < PRICE_CHANGE_DOWN

                if price_up and oi_up:
                    sig = OISignal.LONG_BUILDUP
                    reason = f"Price ↑ + OI ↑{oi_chg_pct:+.1f}% → Fresh longs"
                elif price_dn and oi_up:
                    sig = OISignal.SHORT_BUILDUP
                    reason = f"Price ↓ + OI ↑{oi_chg_pct:+.1f}% → Fresh shorts"
                elif price_up and not oi_up:
                    sig = OISignal.SHORT_COVERING
                    reason = f"Price ↑ + OI ↓{oi_chg_pct:+.1f}% → Short covering"
                elif price_dn and not oi_up:
                    sig = OISignal.LONG_UNWINDING
                    reason = f"Price ↓ + OI ↓{oi_chg_pct:+.1f}% → Long unwinding"
                else:
                    continue

                signals.append(StrikeOISignal(
                    strike=row.strike,
                    option_type=opt_type,
                    signal=sig,
                    price_change_pct=round(price_chg_pct, 2),
                    oi_change_pct=round(oi_chg_pct, 2),
                    reasoning=reason,
                ))

        return signals

    # ─────────────────────────────────────────────────────────
    # IV SKEW
    # ─────────────────────────────────────────────────────────

    def _compute_iv_skew(self, snapshot: OptionChainSnapshot) -> Optional[float]:
        """
        IV Skew = ATM Put IV - ATM Call IV.
        Positive skew = puts more expensive = fear / bearish bias.
        Negative skew = calls more expensive = greed / bullish bias.
        """
        if not snapshot.atm_strike:
            return None
        atm_rows = [r for r in snapshot.rows if r.strike == snapshot.atm_strike]
        if not atm_rows:
            return None
        row = atm_rows[0]
        if row.ce_iv > 0 and row.pe_iv > 0:
            return round(row.pe_iv - row.ce_iv, 2)
        return None

    # ─────────────────────────────────────────────────────────
    # KEY LEVELS
    # ─────────────────────────────────────────────────────────

    def _build_key_levels(
        self,
        snapshot: OptionChainSnapshot,
        walls: List[OIWall],
    ) -> Dict[str, float]:
        """Build a concise key level map for the dashboard."""
        levels: Dict[str, float] = {}
        spot = snapshot.spot_price

        if snapshot.atm_strike:
            levels["ATM"] = snapshot.atm_strike
        if snapshot.max_pain:
            levels["Max Pain"] = snapshot.max_pain

        # Top CE walls above spot (resistance)
        ce_above = sorted(
            [w for w in walls if w.option_type == "CE" and w.strike > spot],
            key=lambda w: w.strike,
        )
        for i, w in enumerate(ce_above[:3], 1):
            levels[f"CE Wall R{i}"] = w.strike

        # Top PE walls below spot (support)
        pe_below = sorted(
            [w for w in walls if w.option_type == "PE" and w.strike < spot],
            key=lambda w: w.strike,
            reverse=True,
        )
        for i, w in enumerate(pe_below[:3], 1):
            levels[f"PE Wall S{i}"] = w.strike

        return levels

    # ─────────────────────────────────────────────────────────
    # SUMMARY FOR TELEGRAM
    # ─────────────────────────────────────────────────────────

    def format_telegram_alert(self, result: OIAnalysisResult) -> str:
        """Format OI analysis as a Telegram message."""
        if not result.success or not result.pcr_analysis:
            return f"❌ OI analysis failed for {result.symbol}"

        pcr = result.pcr_analysis
        spot = result.spot_price

        emoji_map = {
            PCRSignal.BULLISH: "🟢",
            PCRSignal.BEARISH: "🔴",
            PCRSignal.NEUTRAL: "🟡",
        }
        emoji = emoji_map.get(pcr.signal, "⚪")

        lines = [
            f"{emoji} **OI Analysis — {result.symbol}** [{result.expiry}]",
            f"Spot: ₹{spot:,.2f} | ATM: {result.atm_strike:,.0f}" if result.atm_strike else f"Spot: ₹{spot:,.2f}",
            "",
            f"📊 **PCR:** {pcr.pcr_oi:.2f} [{pcr.signal_strength} {pcr.signal.value}]",
            f"📝 {pcr.interpretation}",
        ]

        if pcr.max_pain:
            lines.append(f"🎯 Max Pain: ₹{pcr.max_pain:,.0f} (gap: {pcr.max_pain_gap_pct:+.1f}%)")

        if result.nearest_ce_wall:
            w = result.nearest_ce_wall
            lines.append(f"🔴 CE Wall (Resistance): {w.strike:,.0f} — OI {w.oi:,} [{w.strength}]")

        if result.nearest_pe_wall:
            w = result.nearest_pe_wall
            lines.append(f"🟢 PE Wall (Support): {w.strike:,.0f} — OI {w.oi:,} [{w.strength}]")

        if result.iv_skew is not None:
            skew_dir = "Put skew (bearish)" if result.iv_skew > 0 else "Call skew (bullish)"
            lines.append(f"📐 IV Skew: {result.iv_skew:+.1f} → {skew_dir}")

        lines.append(f"\n🕐 {result.timestamp.strftime('%H:%M IST')}")
        return "\n".join(lines)
