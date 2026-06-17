"""
NIFTY Quant Lab — Volatility Analytics
=========================================
Time series volatility calculations adapted from gs-quant's
gs_quant.timeseries.econometrics and gs_quant.timeseries.technicals modules.

Source formulas (from goldmansachs/gs-quant, Apache 2.0 licensed):

  Realized volatility (technicals.volatility / econometrics.volatility):
    Y_t = sqrt( 1/(N-1) * Σ(R_t - mean(R))² ) * sqrt(252) * 100
    where R_t are returns over a rolling window of size N, annualized to %.

  Exponential volatility (technicals.exponential_volatility):
    Y_t = annualize( exponential_std( returns(x), beta ) ) * 100
    where exponential_std uses x.ewm(alpha=1-beta, adjust=False).std()

We extend gs-quant's close-to-close approach with two additional
estimators that are more efficient for NSE's daily H/L/C data:
  - Parkinson (1980): uses High/Low range, ~5x more efficient than close-to-close
  - Garman-Klass (1980): uses OHLC, captures opening jumps too

This module is purely computational (no I/O) — it operates on
pandas Series/DataFrames the same way the IndicatorEngine does,
and is designed to be called from indicators/engine.py or
signals/scanner.py without needing the DB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from nifty_quant_lab.utils.logger import get_logger

logger = get_logger("volatility")

TRADING_DAYS_PER_YEAR = 252

# ── Vol regime percentile bands (computed against the symbol's own
#    trailing 1Y history, not a fixed absolute number — NIFTY's "normal"
#    vol is not the same as a small-cap's "normal" vol)
REGIME_LOW_PCTL    = 25
REGIME_HIGH_PCTL   = 75
REGIME_EXTREME_PCTL = 95


# ─────────────────────────────────────────────────────────────
# CORE GS-QUANT-STYLE FUNCTIONS
# (ported from gs_quant.timeseries.econometrics / technicals)
# ─────────────────────────────────────────────────────────────

def returns(prices: pd.Series, log_returns: bool = False) -> pd.Series:
    """
    Simple or logarithmic returns of a price series.
    Mirrors gs_quant.timeseries.algebra.returns().
    """
    if log_returns:
        return np.log(prices / prices.shift(1))
    return prices.pct_change()


def realized_volatility(
    prices: pd.Series,
    window: int = 20,
    annualize_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Rolling annualized realized volatility — gs-quant's `volatility()`.

    Y_t = sqrt( 1/(N-1) * Σ(R_t - mean(R))² ) * sqrt(252) * 100

    A reading of 20.0 means 20% annualized volatility.

    Args:
        prices: close price series
        window: rolling window size (gs-quant default convention: 20 ≈ 1 month)
        annualize_factor: trading days per year (252 for equities/index)
    """
    r = returns(prices)
    rolling_std = r.rolling(window=window, min_periods=window).std(ddof=1)
    return (rolling_std * np.sqrt(annualize_factor) * 100).round(2)


def exponential_std(series: pd.Series, beta: float = 0.75) -> pd.Series:
    """
    Exponentially weighted standard deviation.
    Mirrors gs_quant.timeseries.statistics.exponential_std.

    beta close to 1 → slow-moving, weights distant past heavily.
    beta close to 0 → fast-moving, reacts almost entirely to latest value.
    """
    if not (0 <= beta < 1):
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    return series.ewm(alpha=1 - beta, adjust=False).std()


def exponential_volatility(
    prices: pd.Series,
    beta: float = 0.75,
    annualize_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Exponentially weighted annualized volatility — gs-quant's `exponential_volatility()`.

    Reacts faster to recent vol changes than the rolling-window realized_volatility,
    since older observations are exponentially down-weighted rather than dropped
    abruptly when they exit the window.
    """
    r = returns(prices)
    ew_std = exponential_std(r, beta)
    return (ew_std * np.sqrt(annualize_factor) * 100).round(2)


# ─────────────────────────────────────────────────────────────
# RANGE-BASED ESTIMATORS (extend gs-quant with H/L/O/C efficiency)
# ─────────────────────────────────────────────────────────────

def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
    annualize_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Parkinson (1980) range-based volatility estimator.

    σ²_t = 1/(4·ln2) · ln(H_t/L_t)²

    Uses the full day's High/Low range rather than just the close,
    making it ~5x more statistically efficient than close-to-close
    realized vol for the same number of observations. Cannot capture
    overnight gaps, so it understates vol around results/news days.
    """
    log_hl = np.log(high / low)
    daily_var = (log_hl ** 2) / (4 * np.log(2))
    rolling_var = daily_var.rolling(window=window, min_periods=window).mean()
    return (np.sqrt(rolling_var * annualize_factor) * 100).round(2)


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    annualize_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Garman-Klass (1980) OHLC volatility estimator.

    σ²_t = 0.5·ln(H/L)² − (2·ln2 − 1)·ln(C/O)²

    More efficient than Parkinson because it also uses the open-close
    move, capturing some of the overnight gap information Parkinson misses.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    daily_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    rolling_var = daily_var.rolling(window=window, min_periods=window).mean()
    # Variance can occasionally go slightly negative on quiet/gappy days; clip at 0
    rolling_var = rolling_var.clip(lower=0)
    return (np.sqrt(rolling_var * annualize_factor) * 100).round(2)


# ─────────────────────────────────────────────────────────────
# VOL REGIME CLASSIFICATION
# ─────────────────────────────────────────────────────────────

@dataclass
class VolRegimeResult:
    """Output of vol regime classification for the latest bar."""
    symbol: str
    current_vol: float                  # latest realized vol reading
    vol_percentile: float                # where current vol sits in its own 1Y history (0-100)
    regime: str                          # LOW / NORMAL / HIGH / EXTREME
    trend: str                           # EXPANDING / CONTRACTING / STABLE
    vol_of_vol: Optional[float] = None   # how volatile the vol itself is
    ewma_vol: Optional[float] = None     # faster-reacting comparison reading
    parkinson_vol: Optional[float] = None
    interpretation: str = ""


def classify_vol_regime(
    vol_series: pd.Series,
    symbol: str = "UNKNOWN",
    ewma_series: Optional[pd.Series] = None,
    parkinson_series: Optional[pd.Series] = None,
    lookback: int = 252,
) -> VolRegimeResult:
    """
    Classify the current volatility regime relative to the symbol's own
    trailing history (not a fixed absolute threshold — appropriate since
    NIFTY's "high vol" and a small-cap's "high vol" are different numbers).

    Regime bands (percentile of trailing `lookback` readings):
        <25th   → LOW       (compressed, often precedes expansion / breakout)
        25-75th → NORMAL
        75-95th → HIGH       (elevated, wider stops warranted)
        >95th   → EXTREME    (panic / event-driven, signal quality degrades)

    Trend: compares latest vol to its own 5-day average to detect
    expansion vs contraction even within a stable regime band.
    """
    clean = vol_series.dropna()
    if clean.empty:
        return VolRegimeResult(
            symbol=symbol, current_vol=0.0, vol_percentile=50.0,
            regime="NORMAL", trend="STABLE",
            interpretation="Insufficient data for vol regime classification",
        )

    history = clean.tail(lookback)
    current_vol = float(clean.iloc[-1])

    # Percentile rank of current vol within its own trailing history
    pctl = float((history <= current_vol).sum() / len(history) * 100)

    if pctl >= REGIME_EXTREME_PCTL:
        regime = "EXTREME"
    elif pctl >= REGIME_HIGH_PCTL:
        regime = "HIGH"
    elif pctl <= REGIME_LOW_PCTL:
        regime = "LOW"
    else:
        regime = "NORMAL"

    # Trend: is vol expanding or contracting right now?
    trend = "STABLE"
    if len(clean) >= 10:
        recent_avg = clean.tail(5).mean()
        prior_avg = clean.tail(10).head(5).mean()
        if prior_avg > 0:
            change_pct = (recent_avg - prior_avg) / prior_avg * 100
            if change_pct > 10:
                trend = "EXPANDING"
            elif change_pct < -10:
                trend = "CONTRACTING"

    # Vol-of-vol: std of vol readings over the lookback, as a stability gauge
    vol_of_vol = float(history.std()) if len(history) > 1 else None

    ewma_latest = float(ewma_series.dropna().iloc[-1]) if (
        ewma_series is not None and not ewma_series.dropna().empty
    ) else None
    parkinson_latest = float(parkinson_series.dropna().iloc[-1]) if (
        parkinson_series is not None and not parkinson_series.dropna().empty
    ) else None

    interpretation = _build_interpretation(regime, trend, current_vol, pctl)

    return VolRegimeResult(
        symbol=symbol,
        current_vol=round(current_vol, 2),
        vol_percentile=round(pctl, 1),
        regime=regime,
        trend=trend,
        vol_of_vol=round(vol_of_vol, 2) if vol_of_vol is not None else None,
        ewma_vol=round(ewma_latest, 2) if ewma_latest is not None else None,
        parkinson_vol=round(parkinson_latest, 2) if parkinson_latest is not None else None,
        interpretation=interpretation,
    )


def _build_interpretation(regime: str, trend: str, vol: float, pctl: float) -> str:
    base = {
        "LOW": f"Vol at {vol:.1f}% ({pctl:.0f}th pctl) — compressed, often precedes a breakout",
        "NORMAL": f"Vol at {vol:.1f}% ({pctl:.0f}th pctl) — within typical range",
        "HIGH": f"Vol at {vol:.1f}% ({pctl:.0f}th pctl) — elevated, widen stops / size down",
        "EXTREME": f"Vol at {vol:.1f}% ({pctl:.0f}th pctl) — panic regime, signal quality degrades",
    }[regime]
    if trend == "EXPANDING":
        base += " | expanding"
    elif trend == "CONTRACTING":
        base += " | contracting"
    return base


# ─────────────────────────────────────────────────────────────
# COMBINED ENGINE — single entry point for the indicator pipeline
# ─────────────────────────────────────────────────────────────

@dataclass
class VolatilityResult:
    """Full volatility output bundle for one symbol's OHLCV history."""
    symbol: str
    df: pd.DataFrame                     # original df + vol columns appended
    regime: Optional[VolRegimeResult] = None
    success: bool = True
    error: Optional[str] = None

    @classmethod
    def err(cls, symbol: str, error: str) -> "VolatilityResult":
        return cls(symbol=symbol, df=pd.DataFrame(), success=False, error=error)


class VolatilityEngine:
    """
    Computes the full volatility suite on an OHLCV DataFrame and appends
    it as new columns, the same convention as IndicatorEngine.compute().

    Usage:
        engine = VolatilityEngine()
        result = engine.compute(df, symbol="NIFTY50")
        df_with_vol = result.df
        print(result.regime.regime)   # "LOW" / "NORMAL" / "HIGH" / "EXTREME"
    """

    def compute(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        windows: Tuple[int, ...] = (20, 60),
        ewma_beta: float = 0.75,
        min_bars: int = 60,
    ) -> VolatilityResult:
        """
        Compute realized vol (20d, 60d by default), EWMA vol, Parkinson vol,
        Garman-Klass vol, and classify the current regime.

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close]
            symbol: for logging/regime labelling
            windows: rolling windows for realized_volatility, e.g. (20, 60)
            ewma_beta: decay factor for exponential_volatility
            min_bars: minimum bars required
        """
        if df is None or df.empty:
            return VolatilityResult.err(symbol, "Empty DataFrame")
        if len(df) < min_bars:
            return VolatilityResult.err(
                symbol, f"Insufficient data: {len(df)} bars (min {min_bars})"
            )

        try:
            out = df.copy()
            close = out["close"].astype(float)
            high = out["high"].astype(float)
            low = out["low"].astype(float)
            open_ = out["open"].astype(float)

            # Close-to-close realized vol at each requested window
            for w in windows:
                out[f"rvol_{w}d"] = realized_volatility(close, window=w)

            # EWMA vol — faster-reacting than the rolling windows above
            out["ewma_vol"] = exponential_volatility(close, beta=ewma_beta)

            # Range-based estimators (more efficient, same window as the
            # shortest realized-vol window for a fair side-by-side comparison)
            primary_window = min(windows)
            out["parkinson_vol"] = parkinson_volatility(high, low, window=primary_window)
            out["gk_vol"] = garman_klass_volatility(
                open_, high, low, close, window=primary_window
            )

            # Use the primary realized-vol column as the regime classification basis
            primary_rvol_col = f"rvol_{primary_window}d"
            regime = classify_vol_regime(
                out[primary_rvol_col],
                symbol=symbol,
                ewma_series=out["ewma_vol"],
                parkinson_series=out["parkinson_vol"],
            )

            logger.debug(
                f"✓ {symbol}: vol suite computed | "
                f"rvol_{primary_window}d={regime.current_vol:.1f}% [{regime.regime}/{regime.trend}]"
            )
            return VolatilityResult(symbol=symbol, df=out, regime=regime)

        except Exception as e:
            logger.error(f"✗ {symbol} volatility computation failed: {e}", exc_info=True)
            return VolatilityResult.err(symbol, str(e))

    def quick_regime(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        window: int = 20,
    ) -> Optional[VolRegimeResult]:
        """
        Lightweight path for the scanner — just the regime classification,
        skipping Parkinson/GK/EWMA when only the gate decision is needed.
        """
        if df is None or len(df) < window * 2:
            return None
        close = df["close"].astype(float)
        rvol = realized_volatility(close, window=window)
        return classify_vol_regime(rvol, symbol=symbol)
