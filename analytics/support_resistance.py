"""
NIFTY Quant Lab - Support / Resistance Engine
===============================================
8 S/R methodologies fused into a unified level scoring system.

Methods implemented:
  1. Swing High / Low (configurable lookback)
  2. Fibonacci Retracement (0.236, 0.382, 0.5, 0.618, 0.786)
  3. Previous Day High / Low
  4. Weekly Levels (Monday–Friday session)
  5. Monthly Levels
  6. Pivot Points (Classic Floor Trader)
  7. Dynamic EMA Support (20/50/200)
  8. Volume Zones (high-volume price clusters)

Output: Scored S/R levels with STRONG_SUPPORT / WEAK_SUPPORT / RESISTANCE / BREAKOUT tags.

Architecture note:
- gs-quant risk measure pattern: each method = one processor
- Results merged and ranked by strength score (0–10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nifty_quant_lab.database.models import SRLevel

logger = logging.getLogger("nql.sr_engine")


@dataclass
class SRLevelResult:
    """Single S/R level result."""
    price: float
    level_type: SRLevel
    method: str
    strength: float = 1.0       # 0-10 composite score
    touches: int = 1
    fib_level: Optional[str] = None
    notes: str = ""

    def __repr__(self) -> str:
        return (
            f"<SRLevel {self.level_type.value} @ {self.price:.2f} "
            f"[{self.method}] strength={self.strength:.1f}>"
        )


@dataclass
class SRAnalysisResult:
    """Full S/R analysis for one symbol."""
    symbol: str
    current_price: float
    levels: List[SRLevelResult] = field(default_factory=list)
    strong_supports: List[SRLevelResult] = field(default_factory=list)
    weak_supports: List[SRLevelResult] = field(default_factory=list)
    resistances: List[SRLevelResult] = field(default_factory=list)
    breakout_levels: List[SRLevelResult] = field(default_factory=list)
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    risk_reward_estimate: Optional[float] = None
    success: bool = True
    error: Optional[str] = None


class SupportResistanceEngine:
    """
    Multi-method S/R level detection engine.

    Each method produces raw price levels. Levels within a cluster
    tolerance (0.5% of price) are merged and their strength summed.
    """

    CLUSTER_TOLERANCE_PCT: float = 0.005  # 0.5%
    MIN_SWING_PERIOD: int = 5
    VOLUME_ZONE_BINS: int = 50

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        current_price: Optional[float] = None,
    ) -> SRAnalysisResult:
        """
        Full S/R analysis pipeline.

        Args:
            df: Daily OHLCV DataFrame (ideally 252+ bars for meaningful levels)
            symbol: Symbol name
            current_price: Override current price (default: last close)
        """
        if df is None or len(df) < 30:
            return SRAnalysisResult(
                symbol=symbol,
                current_price=current_price or 0,
                success=False,
                error="Insufficient data (need 30+ bars)",
            )

        try:
            price = current_price or float(df["close"].iloc[-1])
            raw_levels: List[SRLevelResult] = []

            # Run all 8 methods
            raw_levels.extend(self._swing_hl(df))
            raw_levels.extend(self._fibonacci(df))
            raw_levels.extend(self._prev_day_hl(df))
            raw_levels.extend(self._weekly_levels(df))
            raw_levels.extend(self._monthly_levels(df))
            raw_levels.extend(self._pivot_points(df))
            raw_levels.extend(self._ema_support(df))
            raw_levels.extend(self._volume_zones(df))

            # Cluster, merge, and classify levels
            merged = self._cluster_levels(raw_levels, price)
            classified = self._classify_levels(merged, price)

            result = SRAnalysisResult(
                symbol=symbol,
                current_price=price,
                levels=classified,
            )

            for lvl in classified:
                if lvl.level_type == SRLevel.STRONG_SUPPORT:
                    result.strong_supports.append(lvl)
                elif lvl.level_type == SRLevel.WEAK_SUPPORT:
                    result.weak_supports.append(lvl)
                elif lvl.level_type == SRLevel.RESISTANCE:
                    result.resistances.append(lvl)
                elif lvl.level_type == SRLevel.BREAKOUT:
                    result.breakout_levels.append(lvl)

            # Find nearest levels
            supports_below = [l.price for l in classified
                              if l.price < price and l.level_type in
                              (SRLevel.STRONG_SUPPORT, SRLevel.WEAK_SUPPORT)]
            resistances_above = [l.price for l in classified
                                  if l.price > price and l.level_type == SRLevel.RESISTANCE]

            result.nearest_support = max(supports_below) if supports_below else None
            result.nearest_resistance = min(resistances_above) if resistances_above else None

            if result.nearest_support and result.nearest_resistance:
                potential_gain = result.nearest_resistance - price
                potential_loss = price - result.nearest_support
                if potential_loss > 0:
                    result.risk_reward_estimate = potential_gain / potential_loss

            logger.debug(
                f"✓ {symbol}: {len(classified)} S/R levels | "
                f"support={result.nearest_support} resistance={result.nearest_resistance}"
            )
            return result

        except Exception as e:
            logger.error(f"✗ {symbol} S/R analysis failed: {e}", exc_info=True)
            return SRAnalysisResult(
                symbol=symbol, current_price=current_price or 0,
                success=False, error=str(e)
            )

    # ─────────────────────────────────────────────────────────
    # METHOD 1: SWING HIGH / LOW
    # ─────────────────────────────────────────────────────────

    def _swing_hl(self, df: pd.DataFrame, lookback: int = 20) -> List[SRLevelResult]:
        """Detect pivot swing highs and lows."""
        levels = []
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)
        period = min(self.MIN_SWING_PERIOD, lookback // 4)

        for i in range(period, n - period):
            # Swing High
            if highs[i] == max(highs[i - period : i + period + 1]):
                levels.append(SRLevelResult(
                    price=float(highs[i]),
                    level_type=SRLevel.RESISTANCE,
                    method="swing_hl",
                    strength=2.0,
                    notes="Swing High",
                ))
            # Swing Low
            if lows[i] == min(lows[i - period : i + period + 1]):
                levels.append(SRLevelResult(
                    price=float(lows[i]),
                    level_type=SRLevel.STRONG_SUPPORT,
                    method="swing_hl",
                    strength=2.0,
                    notes="Swing Low",
                ))
        return levels

    # ─────────────────────────────────────────────────────────
    # METHOD 2: FIBONACCI RETRACEMENT
    # ─────────────────────────────────────────────────────────

    def _fibonacci(self, df: pd.DataFrame, lookback: int = 120) -> List[SRLevelResult]:
        """Fibonacci retracement from recent swing high to swing low."""
        window = df.tail(lookback)
        swing_high = float(window["high"].max())
        swing_low = float(window["low"].min())
        diff = swing_high - swing_low

        fib_ratios = {
            "0.236": 0.236,
            "0.382": 0.382,
            "0.500": 0.500,
            "0.618": 0.618,
            "0.786": 0.786,
        }

        levels = []
        for label, ratio in fib_ratios.items():
            price = swing_high - ratio * diff
            strength = 3.0 if ratio in (0.382, 0.618) else 2.0
            levels.append(SRLevelResult(
                price=round(price, 2),
                level_type=SRLevel.STRONG_SUPPORT if ratio >= 0.382 else SRLevel.WEAK_SUPPORT,
                method="fibonacci",
                strength=strength,
                fib_level=label,
                notes=f"Fib {label} ({swing_low:.0f}–{swing_high:.0f})",
            ))
        return levels

    # ─────────────────────────────────────────────────────────
    # METHOD 3: PREVIOUS DAY HIGH / LOW
    # ─────────────────────────────────────────────────────────

    def _prev_day_hl(self, df: pd.DataFrame) -> List[SRLevelResult]:
        """Previous trading day's high and low — PDH/PDL."""
        if len(df) < 2:
            return []
        prev = df.iloc[-2]
        return [
            SRLevelResult(
                price=float(prev["high"]),
                level_type=SRLevel.RESISTANCE,
                method="prev_day_hl",
                strength=3.0,
                notes="Previous Day High (PDH)",
            ),
            SRLevelResult(
                price=float(prev["low"]),
                level_type=SRLevel.STRONG_SUPPORT,
                method="prev_day_hl",
                strength=3.0,
                notes="Previous Day Low (PDL)",
            ),
        ]

    # ─────────────────────────────────────────────────────────
    # METHOD 4: WEEKLY LEVELS
    # ─────────────────────────────────────────────────────────

    def _weekly_levels(self, df: pd.DataFrame, n_weeks: int = 8) -> List[SRLevelResult]:
        """Weekly high/low for last N weeks."""
        df_w = df.resample("W").agg({"high": "max", "low": "min"}).dropna()
        levels = []
        for _, row in df_w.tail(n_weeks).iterrows():
            levels.append(SRLevelResult(
                price=float(row["high"]),
                level_type=SRLevel.RESISTANCE,
                method="weekly",
                strength=1.5,
                notes="Weekly High",
            ))
            levels.append(SRLevelResult(
                price=float(row["low"]),
                level_type=SRLevel.WEAK_SUPPORT,
                method="weekly",
                strength=1.5,
                notes="Weekly Low",
            ))
        return levels

    # ─────────────────────────────────────────────────────────
    # METHOD 5: MONTHLY LEVELS
    # ─────────────────────────────────────────────────────────

    def _monthly_levels(self, df: pd.DataFrame, n_months: int = 6) -> List[SRLevelResult]:
        """Monthly high/low for last N months."""
        df_m = df.resample("ME").agg({"high": "max", "low": "min"}).dropna()
        levels = []
        for _, row in df_m.tail(n_months).iterrows():
            levels.append(SRLevelResult(
                price=float(row["high"]),
                level_type=SRLevel.RESISTANCE,
                method="monthly",
                strength=3.5,
                notes="Monthly High",
            ))
            levels.append(SRLevelResult(
                price=float(row["low"]),
                level_type=SRLevel.STRONG_SUPPORT,
                method="monthly",
                strength=3.5,
                notes="Monthly Low",
            ))
        return levels

    # ─────────────────────────────────────────────────────────
    # METHOD 6: PIVOT POINTS
    # ─────────────────────────────────────────────────────────

    def _pivot_points(self, df: pd.DataFrame) -> List[SRLevelResult]:
        """Classic floor trader pivot points from previous session."""
        if len(df) < 2:
            return []
        prev = df.iloc[-2]
        ph, pl, pc = float(prev["high"]), float(prev["low"]), float(prev["close"])
        pivot = (ph + pl + pc) / 3

        levels_map = {
            "R3": ph + 2 * (pivot - pl),
            "R2": pivot + (ph - pl),
            "R1": 2 * pivot - pl,
            "P":  pivot,
            "S1": 2 * pivot - ph,
            "S2": pivot - (ph - pl),
            "S3": pl - 2 * (ph - pivot),
        }

        results = []
        for name, price in levels_map.items():
            is_support = name.startswith("S") or name == "P"
            strength = 4.0 if name in ("R1", "S1", "P") else 2.5
            results.append(SRLevelResult(
                price=round(price, 2),
                level_type=SRLevel.STRONG_SUPPORT if is_support else SRLevel.RESISTANCE,
                method="pivot",
                strength=strength,
                notes=f"Pivot {name}",
            ))
        return results

    # ─────────────────────────────────────────────────────────
    # METHOD 7: DYNAMIC EMA SUPPORT
    # ─────────────────────────────────────────────────────────

    def _ema_support(self, df: pd.DataFrame) -> List[SRLevelResult]:
        """EMA20, EMA50, EMA200 as dynamic support/resistance."""
        close = df["close"].astype(float)
        current = float(close.iloc[-1])
        levels = []

        ema_configs = [
            (20, "EMA20", 2.0),
            (50, "EMA50", 3.0),
            (200, "EMA200", 5.0),
        ]

        for period, name, strength in ema_configs:
            if len(close) < period:
                continue
            ema_val = float(close.ewm(span=period, adjust=False).mean().iloc[-1])
            is_support = current > ema_val
            levels.append(SRLevelResult(
                price=round(ema_val, 2),
                level_type=SRLevel.STRONG_SUPPORT if is_support else SRLevel.RESISTANCE,
                method="ema_dynamic",
                strength=strength,
                notes=f"{name} (dynamic {'support' if is_support else 'resistance'})",
            ))
        return levels

    # ─────────────────────────────────────────────────────────
    # METHOD 8: VOLUME ZONES
    # ─────────────────────────────────────────────────────────

    def _volume_zones(
        self,
        df: pd.DataFrame,
        bins: int = 50,
        top_n: int = 5,
    ) -> List[SRLevelResult]:
        """
        High-volume price zones — areas where significant volume traded.
        Uses price histogram weighted by volume (simplified Market Profile).
        """
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        current = float(close.iloc[-1])

        if volume.sum() == 0:
            return []

        # Build volume profile
        price_min, price_max = float(close.min()), float(close.max())
        price_bins = np.linspace(price_min, price_max, bins + 1)
        vol_profile = np.zeros(bins)

        for i in range(len(close)):
            bin_idx = min(int((close.iloc[i] - price_min) / (price_max - price_min) * bins), bins - 1)
            vol_profile[bin_idx] += volume.iloc[i]

        # Top N volume nodes
        top_indices = np.argsort(vol_profile)[-top_n:]
        levels = []

        for idx in top_indices:
            zone_price = (price_bins[idx] + price_bins[idx + 1]) / 2
            vol_strength = float(vol_profile[idx] / vol_profile.max() * 5)
            is_support = zone_price < current
            levels.append(SRLevelResult(
                price=round(float(zone_price), 2),
                level_type=SRLevel.STRONG_SUPPORT if is_support else SRLevel.RESISTANCE,
                method="volume_zone",
                strength=vol_strength,
                notes=f"High Volume Zone (rank {top_n - list(top_indices).index(idx)})",
            ))
        return levels

    # ─────────────────────────────────────────────────────────
    # CLUSTERING & CLASSIFICATION
    # ─────────────────────────────────────────────────────────

    def _cluster_levels(
        self,
        raw_levels: List[SRLevelResult],
        current_price: float,
    ) -> List[SRLevelResult]:
        """
        Merge levels within tolerance band.
        Strength is summed; method list is combined.
        """
        if not raw_levels:
            return []

        tolerance = current_price * self.CLUSTER_TOLERANCE_PCT
        sorted_levels = sorted(raw_levels, key=lambda x: x.price)
        merged: List[SRLevelResult] = []

        cluster: List[SRLevelResult] = [sorted_levels[0]]
        for lvl in sorted_levels[1:]:
            if lvl.price - cluster[-1].price <= tolerance:
                cluster.append(lvl)
            else:
                merged.append(self._merge_cluster(cluster))
                cluster = [lvl]
        if cluster:
            merged.append(self._merge_cluster(cluster))

        return merged

    @staticmethod
    def _merge_cluster(cluster: List[SRLevelResult]) -> SRLevelResult:
        """Merge a list of nearby levels into one."""
        if len(cluster) == 1:
            return cluster[0]

        avg_price = np.mean([l.price for l in cluster])
        total_strength = min(sum(l.strength for l in cluster), 10.0)
        total_touches = sum(l.touches for l in cluster)
        methods = list({l.method for l in cluster})

        # Vote on level_type
        support_votes = sum(
            1 for l in cluster
            if l.level_type in (SRLevel.STRONG_SUPPORT, SRLevel.WEAK_SUPPORT)
        )
        type_vote = SRLevel.STRONG_SUPPORT if support_votes >= len(cluster) / 2 else SRLevel.RESISTANCE

        return SRLevelResult(
            price=round(float(avg_price), 2),
            level_type=type_vote,
            method="+".join(methods),
            strength=total_strength,
            touches=total_touches,
            notes=f"Cluster of {len(cluster)} levels",
        )

    def _classify_levels(
        self,
        levels: List[SRLevelResult],
        current_price: float,
    ) -> List[SRLevelResult]:
        """
        Classify each level based on position relative to current price
        and its strength score.
        """
        for lvl in levels:
            distance_pct = abs(lvl.price - current_price) / current_price

            if lvl.price > current_price:
                lvl.level_type = SRLevel.RESISTANCE
                # If price just broke above a resistance, mark as breakout
                if distance_pct < 0.003:
                    lvl.level_type = SRLevel.BREAKOUT
            else:
                # Below price = support
                if lvl.strength >= 4.0:
                    lvl.level_type = SRLevel.STRONG_SUPPORT
                else:
                    lvl.level_type = SRLevel.WEAK_SUPPORT

        return sorted(levels, key=lambda x: x.price, reverse=True)
