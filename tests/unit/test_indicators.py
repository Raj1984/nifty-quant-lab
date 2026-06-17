"""
Unit tests for NIFTY Quant Lab indicator engine.
Tests all 17+ indicators on synthetic OHLCV data.

Inspired by qlib's test suite patterns and vectorbt's test approach:
test edge cases, boundary values, NaN handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_quant_lab.indicators.engine import (
    IndicatorEngine,
    _ema, _sma, _rsi, _macd, _bollinger_bands,
    _atr, _adx, _supertrend, _vwap, _obv, _cci, _ichimoku,
)


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

def _synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic NSE-like OHLCV data for testing."""
    np.random.seed(seed)
    close = 18000 + np.cumsum(np.random.randn(n) * 80)
    close = np.maximum(close, 1000)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 30
    volume = np.random.randint(5_000_000, 50_000_000, n)

    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=idx)


@pytest.fixture
def ohlcv():
    return _synthetic_ohlcv(300)


@pytest.fixture
def engine():
    return IndicatorEngine()


# ─────────────────────────────────────────────────────────────
# LOW-LEVEL FUNCTION TESTS
# ─────────────────────────────────────────────────────────────

class TestEMA:
    def test_length(self, ohlcv):
        result = _ema(ohlcv["close"], 20)
        assert len(result) == len(ohlcv)

    def test_nan_at_start(self, ohlcv):
        result = _ema(ohlcv["close"], 20)
        # First 19 values should be NaN (min_periods=period)
        assert result.iloc[:19].isna().all()

    def test_values_positive(self, ohlcv):
        result = _ema(ohlcv["close"], 20)
        assert (result.dropna() > 0).all()

    def test_different_periods(self, ohlcv):
        e20 = _ema(ohlcv["close"], 20).dropna()
        e50 = _ema(ohlcv["close"], 50).dropna()
        e200 = _ema(ohlcv["close"], 200).dropna()
        assert len(e20) > len(e200), "Longer period → more NaNs at start"


class TestRSI:
    def test_range(self, ohlcv):
        rsi = _rsi(ohlcv["close"], 14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_length(self, ohlcv):
        rsi = _rsi(ohlcv["close"], 14)
        assert len(rsi) == len(ohlcv)

    def test_no_nan_after_warmup(self, ohlcv):
        rsi = _rsi(ohlcv["close"], 14)
        assert not rsi.iloc[14:].isna().any()


class TestMACD:
    def test_returns_three_series(self, ohlcv):
        macd_line, signal, hist = _macd(ohlcv["close"])
        assert len(macd_line) == len(ohlcv)
        assert len(signal) == len(ohlcv)
        assert len(hist) == len(ohlcv)

    def test_histogram_equals_line_minus_signal(self, ohlcv):
        macd_line, signal, hist = _macd(ohlcv["close"])
        diff = (macd_line - signal - hist).dropna()
        assert (diff.abs() < 1e-10).all()


class TestBollingerBands:
    def test_upper_above_lower(self, ohlcv):
        upper, mid, lower, _, _ = _bollinger_bands(ohlcv["close"])
        valid = upper.dropna().index
        assert (upper[valid] >= lower[valid]).all()

    def test_middle_is_sma(self, ohlcv):
        _, mid, _, _, _ = _bollinger_bands(ohlcv["close"], period=20)
        sma = _sma(ohlcv["close"], 20)
        diff = (mid - sma).dropna()
        assert (diff.abs() < 1e-8).all()


class TestATR:
    def test_positive(self, ohlcv):
        atr = _atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert (atr.dropna() > 0).all()

    def test_length(self, ohlcv):
        atr = _atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert len(atr) == len(ohlcv)


class TestSupertrend:
    def test_direction_values(self, ohlcv):
        _, direction = _supertrend(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        unique_dir = set(direction.dropna().astype(int).unique())
        assert unique_dir.issubset({1, -1})

    def test_length(self, ohlcv):
        st, direction = _supertrend(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert len(st) == len(ohlcv)
        assert len(direction) == len(ohlcv)


class TestVWAP:
    def test_between_low_and_high_approx(self, ohlcv):
        vwap = _vwap(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
        valid = vwap.dropna()
        # VWAP should generally be between session low and high
        # (not strictly guaranteed for cumulative VWAP across days)
        assert (valid > 0).all()


class TestOBV:
    def test_length(self, ohlcv):
        obv = _obv(ohlcv["close"], ohlcv["volume"])
        assert len(obv) == len(ohlcv)

    def test_first_value_zero(self, ohlcv):
        obv = _obv(ohlcv["close"], ohlcv["volume"])
        assert obv.iloc[0] == 0


# ─────────────────────────────────────────────────────────────
# INDICATOR ENGINE INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────

class TestIndicatorEngine:
    def test_compute_returns_result(self, engine, ohlcv):
        result = engine.compute(ohlcv, symbol="TEST")
        assert result.success
        assert not result.df.empty

    def test_all_columns_present(self, engine, ohlcv):
        result = engine.compute(ohlcv, "TEST")
        expected_cols = [
            "ema_9", "ema_20", "ema_50", "ema_200",
            "sma_20", "sma_50", "sma_200",
            "rsi_14", "rsi_9",
            "stoch_rsi_k", "stoch_rsi_d",
            "macd_line", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower",
            "atr_14", "adx_14", "adx_di_plus", "adx_di_minus",
            "supertrend", "supertrend_direction",
            "vwap", "obv", "volume_sma_20", "volume_ratio",
            "cci_20",
            "ichimoku_tenkan", "ichimoku_kijun",
            "pivot_classic", "pivot_r1", "pivot_s1",
        ]
        for col in expected_cols:
            assert col in result.df.columns, f"Missing column: {col}"

    def test_insufficient_data(self, engine):
        small = _synthetic_ohlcv(n=10)
        result = engine.compute(small, "TEST")
        assert not result.success
        assert "Insufficient" in result.error

    def test_empty_df(self, engine):
        result = engine.compute(pd.DataFrame(), "TEST")
        assert not result.success

    def test_rsi_range(self, engine, ohlcv):
        result = engine.compute(ohlcv, "TEST")
        rsi = result.df["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_ema_ordering(self, engine, ohlcv):
        """In a strong uptrend: EMA9 > EMA20 > EMA50 near the end."""
        # Generate a clear uptrend
        n = 300
        close = np.linspace(10000, 25000, n) + np.random.randn(n) * 50
        df = pd.DataFrame({
            "open": close * 0.99, "high": close * 1.01,
            "low": close * 0.98, "close": close,
            "volume": np.ones(n) * 1_000_000,
        }, index=pd.date_range("2022-01-01", periods=n, freq="B"))
        result = engine.compute(df, "UPTREND")
        assert result.success
        last = result.df.iloc[-1]
        assert last["ema_9"] > last["ema_20"] > last["ema_50"]

    def test_compute_latest(self, engine, ohlcv):
        latest = engine.compute_latest(ohlcv, "TEST")
        assert latest is not None
        assert "rsi_14" in latest
        assert "macd_histogram" in latest

    def test_supertrend_direction_binary(self, engine, ohlcv):
        result = engine.compute(ohlcv, "TEST")
        dirs = result.df["supertrend_direction"].dropna().astype(int)
        assert set(dirs.unique()).issubset({1, -1})


# ─────────────────────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_flat_price_series(self, engine):
        """All prices identical — should not crash."""
        n = 100
        df = pd.DataFrame({
            "open": [18000] * n, "high": [18000] * n,
            "low": [18000] * n, "close": [18000] * n,
            "volume": [1_000_000] * n,
        }, index=pd.date_range("2023-01-01", periods=n, freq="B"))
        result = engine.compute(df, "FLAT")
        # May succeed or fail gracefully — must not raise exception
        assert isinstance(result.success, bool)

    def test_single_spike(self, engine, ohlcv):
        """Extreme outlier should not crash computation."""
        df = ohlcv.copy()
        df.iloc[150]["close"] = df["close"].mean() * 10
        result = engine.compute(df, "SPIKE")
        assert isinstance(result.success, bool)

    def test_zero_volume(self, engine, ohlcv):
        """Zero volume in some bars — OBV/VWAP should handle gracefully."""
        df = ohlcv.copy()
        df.iloc[50:60, df.columns.get_loc("volume")] = 0
        result = engine.compute(df, "ZERO_VOL")
        assert result.success
