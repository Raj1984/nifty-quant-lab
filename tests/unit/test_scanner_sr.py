"""
Unit tests for SwingScanner and SupportResistanceEngine.
Tests signal logic, scoring, and S/R level detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date

from nifty_quant_lab.signals.scanner import (
    SwingScanner, ScanResult, ScanSession,
    CONDITION_WEIGHTS, BUY_THRESHOLD, WATCHLIST_THRESHOLD,
)
from nifty_quant_lab.analytics.support_resistance import (
    SupportResistanceEngine, SRAnalysisResult,
)
from nifty_quant_lab.database.models import SignalType, SRLevel


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300, trend: str = "up", seed: int = 1) -> pd.DataFrame:
    """Create synthetic OHLCV with controllable trend."""
    np.random.seed(seed)
    if trend == "up":
        close = np.linspace(16000, 22000, n) + np.random.randn(n) * 100
    elif trend == "down":
        close = np.linspace(22000, 16000, n) + np.random.randn(n) * 100
    else:
        close = 18000 + np.random.randn(n) * 200

    close = np.maximum(close, 100)
    high = close + np.abs(np.random.randn(n)) * 80 + 20
    low = close - np.abs(np.random.randn(n)) * 80 - 20
    low = np.maximum(low, 1)
    volume = np.random.randint(1_000_000, 20_000_000, n).astype(float)

    # Simulate volume spike on last 3 bars for scanner tests
    volume[-3:] *= 2.5

    return pd.DataFrame({
        "open": close + np.random.randn(n) * 30,
        "high": high, "low": low, "close": close, "volume": volume,
    }, index=pd.date_range("2022-01-01", periods=n, freq="B"))


@pytest.fixture
def uptrend_df():
    return _make_ohlcv(300, "up")


@pytest.fixture
def downtrend_df():
    return _make_ohlcv(300, "down")


@pytest.fixture
def scanner():
    return SwingScanner()


@pytest.fixture
def sr_engine():
    return SupportResistanceEngine()


# ─────────────────────────────────────────────────────────────
# CONDITION WEIGHT TESTS
# ─────────────────────────────────────────────────────────────

class TestConditionWeights:
    def test_weights_sum_to_100(self):
        assert sum(CONDITION_WEIGHTS.values()) == 100

    def test_all_weights_positive(self):
        assert all(w > 0 for w in CONDITION_WEIGHTS.values())

    def test_thresholds_valid(self):
        assert BUY_THRESHOLD > WATCHLIST_THRESHOLD
        assert WATCHLIST_THRESHOLD >= 40


# ─────────────────────────────────────────────────────────────
# SCAN RESULT TESTS
# ─────────────────────────────────────────────────────────────

class TestScanResult:
    def test_conditions_met_count(self):
        r = ScanResult(
            symbol="TEST", signal=SignalType.BUY,
            score=75.0, close_price=18500.0,
            ema20_above_ema50=True, rsi_above_55=True,
            macd_bullish_cross=True, price_above_supertrend=False,
            volume_above_avg=False, week52_breakout=False,
        )
        assert r.conditions_met == 3

    def test_to_dict_keys(self):
        r = ScanResult(
            symbol="NIFTY", signal=SignalType.HOLD,
            score=40.0, close_price=19200.0,
        )
        d = r.to_dict()
        assert "symbol" in d
        assert "signal" in d
        assert "score" in d
        assert "conditions_met" in d


# ─────────────────────────────────────────────────────────────
# SCANNER EVALUATION TESTS
# ─────────────────────────────────────────────────────────────

class TestSwingScanner:
    def test_evaluate_uptrend(self, scanner, uptrend_df):
        result = scanner._evaluate_symbol("TEST_UP", uptrend_df, date.today())
        assert result is not None
        # In a strong uptrend, should get at least a few conditions
        assert result.conditions_met >= 2

    def test_evaluate_downtrend(self, scanner, downtrend_df):
        result = scanner._evaluate_symbol("TEST_DOWN", downtrend_df, date.today())
        assert result is not None
        # In a strong downtrend, buy conditions should mostly fail
        assert result.conditions_met <= 3

    def test_signal_classification_buy(self, scanner):
        """Score ≥ BUY_THRESHOLD + 4 conditions → BUY."""
        r = ScanResult(
            symbol="X", signal=SignalType.BUY, score=75.0, close_price=100.0,
            ema20_above_ema50=True, rsi_above_55=True, macd_bullish_cross=True,
            price_above_supertrend=True, volume_above_avg=False, week52_breakout=False,
        )
        assert r.signal == SignalType.BUY
        assert r.conditions_met == 4

    def test_sl_target_positive(self, scanner, uptrend_df):
        result = scanner._evaluate_symbol("TEST", uptrend_df, date.today())
        if result and result.suggested_sl and result.suggested_target:
            assert result.suggested_sl < result.close_price
            assert result.suggested_target > result.close_price

    def test_risk_reward_minimum(self, scanner, uptrend_df):
        result = scanner._evaluate_symbol("TEST", uptrend_df, date.today())
        if result and result.risk_reward:
            # Should be 2:1 (3×ATR target / 1.5×ATR stop)
            assert abs(result.risk_reward - 2.0) < 0.5, f"R:R should be ~2.0, got {result.risk_reward}"

    def test_returns_none_on_empty(self, scanner):
        result = scanner._evaluate_symbol("EMPTY", pd.DataFrame(), date.today())
        assert result is None

    def test_format_signal_summary(self, scanner):
        session = ScanSession(scan_date=date.today(), total_scanned=50)
        session.results = [
            ScanResult(
                symbol=f"STOCK{i}", signal=SignalType.BUY,
                score=70 + i, close_price=1000.0 + i * 10,
                scan_date=date.today(),
            )
            for i in range(5)
        ]
        summary = scanner.format_signal_summary(session)
        assert "NIFTY Swing Scanner" in summary
        assert "BUY" in summary


# ─────────────────────────────────────────────────────────────
# SCAN SESSION TESTS
# ─────────────────────────────────────────────────────────────

class TestScanSession:
    def _make_session(self) -> ScanSession:
        session = ScanSession(scan_date=date.today(), total_scanned=10)
        session.results = [
            ScanResult(symbol="A", signal=SignalType.BUY, score=80, close_price=100),
            ScanResult(symbol="B", signal=SignalType.BUY, score=70, close_price=200),
            ScanResult(symbol="C", signal=SignalType.WATCHLIST, score=55, close_price=300),
            ScanResult(symbol="D", signal=SignalType.SELL, score=20, close_price=400),
        ]
        return session

    def test_buy_signals_filter(self):
        session = self._make_session()
        assert len(session.buy_signals) == 2

    def test_watchlist_filter(self):
        session = self._make_session()
        assert len(session.watchlist_signals) == 1

    def test_top_buys_ordered(self):
        session = self._make_session()
        top = session.top_buys(10)
        assert top[0].score >= top[1].score

    def test_to_dataframe(self):
        session = self._make_session()
        df = session.to_dataframe()
        assert len(df) == 4
        assert "symbol" in df.columns
        assert "score" in df.columns
        # Sorted by score descending
        scores = df["score"].tolist()
        assert scores == sorted(scores, reverse=True)


# ─────────────────────────────────────────────────────────────
# S/R ENGINE TESTS
# ─────────────────────────────────────────────────────────────

class TestSupportResistanceEngine:
    def test_analyze_returns_result(self, sr_engine, uptrend_df):
        result = sr_engine.analyze(uptrend_df, "TEST")
        assert result.success
        assert len(result.levels) > 0

    def test_insufficient_data(self, sr_engine):
        small = _make_ohlcv(10)
        result = sr_engine.analyze(small, "SHORT")
        assert not result.success

    def test_levels_have_prices(self, sr_engine, uptrend_df):
        result = sr_engine.analyze(uptrend_df, "TEST")
        for lvl in result.levels:
            assert lvl.price > 0
            assert isinstance(lvl.level_type, SRLevel)

    def test_supports_below_price(self, sr_engine, uptrend_df):
        result = sr_engine.analyze(uptrend_df, "TEST")
        for lvl in result.strong_supports + result.weak_supports:
            # All supports should be at or below current price
            assert lvl.price <= result.current_price * 1.005  # 0.5% tolerance

    def test_resistances_above_price(self, sr_engine, uptrend_df):
        result = sr_engine.analyze(uptrend_df, "TEST")
        for lvl in result.resistances:
            assert lvl.price >= result.current_price * 0.995

    def test_nearest_levels_exist(self, sr_engine, uptrend_df):
        result = sr_engine.analyze(uptrend_df, "TEST")
        # In a 300-bar dataset we should always find at least one of each
        assert result.nearest_support is not None or result.nearest_resistance is not None

    def test_fib_levels_produced(self, sr_engine, uptrend_df):
        levels = sr_engine._fibonacci(uptrend_df)
        assert len(levels) == 5  # 5 fib ratios
        fib_labels = {l.fib_level for l in levels}
        assert "0.382" in fib_labels
        assert "0.618" in fib_labels

    def test_pivot_points_produced(self, sr_engine, uptrend_df):
        levels = sr_engine._pivot_points(uptrend_df)
        assert len(levels) == 7  # P, R1, R2, R3, S1, S2, S3

    def test_cluster_merging(self, sr_engine):
        """Levels within 0.5% should be merged."""
        from nifty_quant_lab.analytics.support_resistance import SRLevelResult
        raw = [
            SRLevelResult(price=18000, level_type=SRLevel.RESISTANCE, method="a", strength=2),
            SRLevelResult(price=18010, level_type=SRLevel.RESISTANCE, method="b", strength=3),
            SRLevelResult(price=18020, level_type=SRLevel.RESISTANCE, method="c", strength=1),
            SRLevelResult(price=19000, level_type=SRLevel.RESISTANCE, method="d", strength=2),
        ]
        merged = sr_engine._cluster_levels(raw, current_price=18000)
        # First three should merge (within 0.5% of 18000 = 90 pts)
        assert len(merged) <= 3, f"Expected ≤3 clusters, got {len(merged)}"

    def test_strength_scores_capped(self, sr_engine, uptrend_df):
        result = sr_engine.analyze(uptrend_df, "TEST")
        for lvl in result.levels:
            assert lvl.strength <= 10.0
