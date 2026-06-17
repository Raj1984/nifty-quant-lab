"""
NIFTY Quant Lab - Technical Indicators Engine
================================================
Computes all 17+ indicators in one pass over OHLCV data.

Architecture inspired by:
- vectorbt's IndicatorFactory (pipeline + caching pattern)
- gs-quant's processor.py (abstract base + typed outputs)
- qlib's feature engineering (Alpha158 feature set concepts)

Indicators implemented:
  EMA (9/20/50/200), SMA (20/50/200)
  RSI (14), Stochastic RSI
  MACD (12,26,9)
  Bollinger Bands (20,2)
  ATR (14)
  VWAP
  ADX (14) + DI+/DI-
  Supertrend (7,3)
  Ichimoku Cloud (9,26,52)
  CCI (20)
  OBV
  Pivot Points (Classic)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("nql.indicators")


# ─────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class IndicatorResult:
    """
    Typed container for all computed indicators.
    gs-quant ProcessorResult pattern — structured, never a raw dict.
    """
    symbol: str
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    success: bool = True
    error: Optional[str] = None

    @classmethod
    def ok(cls, symbol: str, df: pd.DataFrame) -> "IndicatorResult":
        return cls(symbol=symbol, df=df, success=True)

    @classmethod
    def err(cls, symbol: str, error: str) -> "IndicatorResult":
        return cls(symbol=symbol, success=False, error=error)


# ─────────────────────────────────────────────────────────────
# LOW-LEVEL INDICATOR FUNCTIONS
# Numpy-native for speed — no library dependency for core calcs
# ─────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, Signal line, Histogram."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Upper, Middle, Lower, Width, %B."""
    middle = _sma(close, period)
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / middle
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return upper, middle, lower, width, pct_b


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False, min_periods=period).mean()


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """ADX, +DI, -DI."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    dm_plus = high - prev_high
    dm_minus = prev_low - low
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)

    atr = tr.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    di_plus = 100 * dm_plus.ewm(com=period - 1, adjust=False).mean() / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(com=period - 1, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(com=period - 1, adjust=False, min_periods=period).mean()

    return adx, di_plus, di_minus


def _supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 7,
    multiplier: float = 3.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Supertrend indicator.
    Returns: (supertrend_line, direction) where direction=1 means uptrend.
    """
    atr = _atr(high, low, close, period)
    hl2 = (high + low) / 2

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    supertrend = pd.Series(np.nan, index=close.index)
    direction = pd.Series(1, index=close.index)

    for i in range(1, len(close)):
        # Upper band
        if upper_basic.iloc[i] < upper_band.iloc[i - 1] or close.iloc[i - 1] > upper_band.iloc[i - 1]:
            upper_band.iloc[i] = upper_basic.iloc[i]
        else:
            upper_band.iloc[i] = upper_band.iloc[i - 1]

        # Lower band
        if lower_basic.iloc[i] > lower_band.iloc[i - 1] or close.iloc[i - 1] < lower_band.iloc[i - 1]:
            lower_band.iloc[i] = lower_basic.iloc[i]
        else:
            lower_band.iloc[i] = lower_band.iloc[i - 1]

        # Direction
        if np.isnan(supertrend.iloc[i - 1]):
            direction.iloc[i] = -1
        elif supertrend.iloc[i - 1] == upper_band.iloc[i - 1]:
            direction.iloc[i] = -1 if close.iloc[i] <= upper_band.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if close.iloc[i] >= lower_band.iloc[i] else -1

        supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

    return supertrend, direction


def _vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Session VWAP — resets daily."""
    typical = (high + low + close) / 3
    tp_vol = typical * volume
    if isinstance(close.index, pd.DatetimeIndex):
        dates = close.index.normalize()
        cum_tp_vol = tp_vol.groupby(dates).cumsum()
        cum_vol = volume.groupby(dates).cumsum()
    else:
        cum_tp_vol = tp_vol.cumsum()
        cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _stochastic_rsi(
    rsi: pd.Series,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """Stochastic RSI — %K and %D."""
    min_rsi = rsi.rolling(window=period, min_periods=period).min()
    max_rsi = rsi.rolling(window=period, min_periods=period).max()
    stoch = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan) * 100
    k = stoch.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(window=smooth_d, min_periods=smooth_d).mean()
    return k, d


def _cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Commodity Channel Index."""
    typical = (high + low + close) / 3
    mean_typical = typical.rolling(window=period, min_periods=period).mean()
    mean_dev = typical.rolling(window=period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (typical - mean_typical) / (0.015 * mean_dev.replace(0, np.nan))


def _ichimoku(
    high: pd.Series,
    low: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Ichimoku Cloud: Tenkan, Kijun, Senkou A, Senkou B, Chikou.
    """
    def _mid(h, l, n):
        return (h.rolling(n).max() + l.rolling(n).min()) / 2

    tenkan_sen = _mid(high, low, tenkan)
    kijun_sen = _mid(high, low, kijun)
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b = _mid(high, low, senkou_b).shift(kijun)
    chikou = high.shift(-kijun)

    return tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou


def _pivot_points(
    prev_high: float,
    prev_low: float,
    prev_close: float,
) -> Dict[str, float]:
    """Classic Pivot Points — Floor Trading Method."""
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    r2 = pivot + (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s1 = 2 * pivot - prev_high
    s2 = pivot - (prev_high - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    return {"pivot": pivot, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}


# ─────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────

class IndicatorEngine:
    """
    Computes all technical indicators in a single vectorized pass.

    Usage:
        engine = IndicatorEngine()
        result = engine.compute(df, symbol="NIFTY50")
        df_with_indicators = result.df
    """

    def compute(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        min_bars: int = 60,
    ) -> IndicatorResult:
        """
        Compute all indicators on OHLCV DataFrame.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
                Index should be DatetimeIndex.
            symbol: Symbol name for logging.
            min_bars: Minimum bars required.

        Returns:
            IndicatorResult with augmented DataFrame.
        """
        if df is None or df.empty:
            return IndicatorResult.err(symbol, "Empty DataFrame")

        if len(df) < min_bars:
            return IndicatorResult.err(
                symbol, f"Insufficient data: {len(df)} bars (min {min_bars})"
            )

        try:
            result_df = df.copy()
            close = result_df["close"].astype(float)
            high = result_df["high"].astype(float)
            low = result_df["low"].astype(float)
            volume = result_df["volume"].astype(float)

            # ── EMAs
            result_df["ema_9"] = _ema(close, 9)
            result_df["ema_20"] = _ema(close, 20)
            result_df["ema_50"] = _ema(close, 50)
            result_df["ema_200"] = _ema(close, 200)

            # ── SMAs
            result_df["sma_20"] = _sma(close, 20)
            result_df["sma_50"] = _sma(close, 50)
            result_df["sma_200"] = _sma(close, 200)

            # ── RSI
            rsi = _rsi(close, 14)
            result_df["rsi_14"] = rsi
            result_df["rsi_9"] = _rsi(close, 9)

            # ── Stochastic RSI
            k, d = _stochastic_rsi(rsi)
            result_df["stoch_rsi"] = (k + d) / 2
            result_df["stoch_rsi_k"] = k
            result_df["stoch_rsi_d"] = d

            # ── MACD
            macd_line, signal_line, histogram = _macd(close)
            result_df["macd_line"] = macd_line
            result_df["macd_signal"] = signal_line
            result_df["macd_histogram"] = histogram

            # ── Bollinger Bands
            bb_upper, bb_mid, bb_lower, bb_width, bb_pctb = _bollinger_bands(close)
            result_df["bb_upper"] = bb_upper
            result_df["bb_middle"] = bb_mid
            result_df["bb_lower"] = bb_lower
            result_df["bb_width"] = bb_width
            result_df["bb_pct_b"] = bb_pctb

            # ── ATR
            result_df["atr_14"] = _atr(high, low, close, 14)

            # ── ADX
            adx, di_plus, di_minus = _adx(high, low, close, 14)
            result_df["adx_14"] = adx
            result_df["adx_di_plus"] = di_plus
            result_df["adx_di_minus"] = di_minus

            # ── Supertrend
            st_line, st_dir = _supertrend(high, low, close, period=7, multiplier=3.0)
            result_df["supertrend"] = st_line
            result_df["supertrend_direction"] = st_dir

            # ── VWAP
            result_df["vwap"] = _vwap(high, low, close, volume)

            # ── OBV
            result_df["obv"] = _obv(close, volume)

            # ── Volume indicators
            result_df["volume_sma_20"] = _sma(volume, 20)
            result_df["volume_ratio"] = volume / result_df["volume_sma_20"].replace(0, np.nan)

            # ── CCI
            result_df["cci_20"] = _cci(high, low, close, 20)

            # ── Ichimoku
            tenkan, kijun, senkou_a, senkou_b, chikou = _ichimoku(high, low)
            result_df["ichimoku_tenkan"] = tenkan
            result_df["ichimoku_kijun"] = kijun
            result_df["ichimoku_senkou_a"] = senkou_a
            result_df["ichimoku_senkou_b"] = senkou_b
            result_df["ichimoku_chikou"] = chikou

            # ── Pivot Points (applied row-by-row using previous day's data)
            result_df["pivot_classic"] = np.nan
            result_df["pivot_r1"] = np.nan
            result_df["pivot_r2"] = np.nan
            result_df["pivot_r3"] = np.nan
            result_df["pivot_s1"] = np.nan
            result_df["pivot_s2"] = np.nan
            result_df["pivot_s3"] = np.nan

            for i in range(1, len(result_df)):
                prev = result_df.iloc[i - 1]
                pp = _pivot_points(float(prev["high"]), float(prev["low"]), float(prev["close"]))
                result_df.iloc[i, result_df.columns.get_loc("pivot_classic")] = pp["pivot"]
                result_df.iloc[i, result_df.columns.get_loc("pivot_r1")] = pp["r1"]
                result_df.iloc[i, result_df.columns.get_loc("pivot_r2")] = pp["r2"]
                result_df.iloc[i, result_df.columns.get_loc("pivot_r3")] = pp["r3"]
                result_df.iloc[i, result_df.columns.get_loc("pivot_s1")] = pp["s1"]
                result_df.iloc[i, result_df.columns.get_loc("pivot_s2")] = pp["s2"]
                result_df.iloc[i, result_df.columns.get_loc("pivot_s3")] = pp["s3"]

            # ── Round all float columns to 2dp
            float_cols = result_df.select_dtypes(include=[float]).columns
            result_df[float_cols] = result_df[float_cols].round(2)

            logger.debug(f"✓ {symbol}: {len(result_df.columns)} indicator columns computed")
            return IndicatorResult.ok(symbol, result_df)

        except Exception as e:
            logger.error(f"✗ {symbol} indicator computation failed: {e}", exc_info=True)
            return IndicatorResult.err(symbol, str(e))

    def compute_latest(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Optional[Dict]:
        """Return only the most recent row as a dict — for scanner."""
        result = self.compute(df, symbol)
        if not result.success or result.df.empty:
            return None
        return result.df.iloc[-1].to_dict()

    def detect_macd_cross(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect MACD bullish crossover signals.
        Returns boolean series: True where histogram crosses from negative to positive.
        """
        result = self.compute(df)
        if not result.success:
            return pd.Series(False, index=df.index)
        hist = result.df["macd_histogram"]
        return (hist > 0) & (hist.shift(1) <= 0)

    def is_above_supertrend(self, df: pd.DataFrame) -> pd.Series:
        """Returns boolean series: True where price is above Supertrend."""
        result = self.compute(df)
        if not result.success:
            return pd.Series(False, index=df.index)
        return result.df["supertrend_direction"] == 1
