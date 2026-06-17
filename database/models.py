"""
NIFTY Quant Lab - Database Models
===================================
SQLAlchemy 2.x declarative models for all market data, signals, and analytics.
Inspired by gs-quant's typed data contracts and OpenBB's standardized data model.
"""

from __future__ import annotations

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List

from sqlalchemy import (
    BigInteger, Boolean, Date, DateTime, Float, Index,
    Integer, Numeric, String, Text, UniqueConstraint,
    ForeignKey, Enum as SAEnum, JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import enum


class Base(DeclarativeBase):
    """Base class for all ORM models — OpenBB-style standardized foundation."""
    pass


# ─────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────

class AssetType(str, enum.Enum):
    INDEX = "INDEX"
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"
    ETF = "ETF"


class SignalType(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    WATCHLIST = "WATCHLIST"


class OISignal(str, enum.Enum):
    LONG_BUILDUP = "LONG_BUILDUP"
    SHORT_BUILDUP = "SHORT_BUILDUP"
    SHORT_COVERING = "SHORT_COVERING"
    LONG_UNWINDING = "LONG_UNWINDING"


class PCRSignal(str, enum.Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SRLevel(str, enum.Enum):
    STRONG_SUPPORT = "STRONG_SUPPORT"
    WEAK_SUPPORT = "WEAK_SUPPORT"
    RESISTANCE = "RESISTANCE"
    BREAKOUT = "BREAKOUT"


# ─────────────────────────────────────────────────────────────
# SYMBOLS
# ─────────────────────────────────────────────────────────────

class Symbol(Base):
    """Master symbol registry — every tradeable instrument."""
    __tablename__ = "symbols"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    asset_type: Mapped[AssetType] = mapped_column(SAEnum(AssetType), nullable=False)
    exchange: Mapped[str] = mapped_column(String(10), default="NSE")
    sector: Mapped[Optional[str]] = mapped_column(String(100))
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    isin: Mapped[Optional[str]] = mapped_column(String(12))
    lot_size: Mapped[int] = mapped_column(Integer, default=1)
    instrument_token: Mapped[Optional[int]] = mapped_column(BigInteger)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_fo_eligible: Mapped[bool] = mapped_column(Boolean, default=False)
    nifty50: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    historical_prices: Mapped[List["HistoricalPrice"]] = relationship(back_populates="symbol_ref")
    intraday_prices: Mapped[List["IntradayPrice"]] = relationship(back_populates="symbol_ref")

    __table_args__ = (
        UniqueConstraint("symbol", "exchange", name="uq_symbol_exchange"),
        Index("ix_symbol_asset_type", "asset_type"),
        Index("ix_symbol_nifty50", "nifty50"),
    )

    def __repr__(self) -> str:
        return f"<Symbol {self.symbol}:{self.exchange} ({self.asset_type})>"


# ─────────────────────────────────────────────────────────────
# PRICE DATA
# ─────────────────────────────────────────────────────────────

class HistoricalPrice(Base):
    """EOD OHLCV data — the foundational time series."""
    __tablename__ = "historical_prices"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("symbols.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, default=0)
    adjusted_close: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    delivery_volume: Mapped[Optional[int]] = mapped_column(BigInteger)
    delivery_pct: Mapped[Optional[float]] = mapped_column(Float)
    vwap: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))

    symbol_ref: Mapped["Symbol"] = relationship(back_populates="historical_prices")

    __table_args__ = (
        UniqueConstraint("symbol_id", "date", name="uq_historical_price"),
        Index("ix_hist_symbol_date", "symbol_id", "date"),
        Index("ix_hist_date", "date"),
    )


class IntradayPrice(Base):
    """Intraday OHLCV — supports 1m, 5m, 15m, 30m, 1h intervals."""
    __tablename__ = "intraday_prices"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("symbols.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    interval: Mapped[str] = mapped_column(String(5), nullable=False)  # 1m, 5m, etc.
    open: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, default=0)
    vwap: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))

    symbol_ref: Mapped["Symbol"] = relationship(back_populates="intraday_prices")

    __table_args__ = (
        UniqueConstraint("symbol_id", "timestamp", "interval", name="uq_intraday_price"),
        Index("ix_intraday_symbol_ts", "symbol_id", "timestamp"),
        Index("ix_intraday_interval", "interval"),
    )


# ─────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────

class TechnicalIndicator(Base):
    """Computed indicator values per symbol per date — vectorbt-inspired wide-column design."""
    __tablename__ = "technical_indicators"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("symbols.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    interval: Mapped[str] = mapped_column(String(5), default="1d")

    # EMAs
    ema_9: Mapped[Optional[float]] = mapped_column(Float)
    ema_20: Mapped[Optional[float]] = mapped_column(Float)
    ema_50: Mapped[Optional[float]] = mapped_column(Float)
    ema_200: Mapped[Optional[float]] = mapped_column(Float)

    # SMAs
    sma_20: Mapped[Optional[float]] = mapped_column(Float)
    sma_50: Mapped[Optional[float]] = mapped_column(Float)
    sma_200: Mapped[Optional[float]] = mapped_column(Float)

    # Momentum
    rsi_14: Mapped[Optional[float]] = mapped_column(Float)
    rsi_9: Mapped[Optional[float]] = mapped_column(Float)
    stoch_rsi: Mapped[Optional[float]] = mapped_column(Float)
    stoch_rsi_k: Mapped[Optional[float]] = mapped_column(Float)
    stoch_rsi_d: Mapped[Optional[float]] = mapped_column(Float)
    cci_20: Mapped[Optional[float]] = mapped_column(Float)

    # MACD
    macd_line: Mapped[Optional[float]] = mapped_column(Float)
    macd_signal: Mapped[Optional[float]] = mapped_column(Float)
    macd_histogram: Mapped[Optional[float]] = mapped_column(Float)

    # Bollinger Bands
    bb_upper: Mapped[Optional[float]] = mapped_column(Float)
    bb_middle: Mapped[Optional[float]] = mapped_column(Float)
    bb_lower: Mapped[Optional[float]] = mapped_column(Float)
    bb_width: Mapped[Optional[float]] = mapped_column(Float)
    bb_pct_b: Mapped[Optional[float]] = mapped_column(Float)

    # Volatility
    atr_14: Mapped[Optional[float]] = mapped_column(Float)

    # Trend
    adx_14: Mapped[Optional[float]] = mapped_column(Float)
    adx_di_plus: Mapped[Optional[float]] = mapped_column(Float)
    adx_di_minus: Mapped[Optional[float]] = mapped_column(Float)
    supertrend: Mapped[Optional[float]] = mapped_column(Float)
    supertrend_direction: Mapped[Optional[int]] = mapped_column(Integer)  # 1=up, -1=down

    # Volume
    vwap: Mapped[Optional[float]] = mapped_column(Float)
    obv: Mapped[Optional[float]] = mapped_column(Float)
    volume_sma_20: Mapped[Optional[float]] = mapped_column(Float)
    volume_ratio: Mapped[Optional[float]] = mapped_column(Float)

    # Ichimoku
    ichimoku_tenkan: Mapped[Optional[float]] = mapped_column(Float)
    ichimoku_kijun: Mapped[Optional[float]] = mapped_column(Float)
    ichimoku_senkou_a: Mapped[Optional[float]] = mapped_column(Float)
    ichimoku_senkou_b: Mapped[Optional[float]] = mapped_column(Float)
    ichimoku_chikou: Mapped[Optional[float]] = mapped_column(Float)

    # Pivot Points
    pivot_classic: Mapped[Optional[float]] = mapped_column(Float)
    pivot_r1: Mapped[Optional[float]] = mapped_column(Float)
    pivot_r2: Mapped[Optional[float]] = mapped_column(Float)
    pivot_r3: Mapped[Optional[float]] = mapped_column(Float)
    pivot_s1: Mapped[Optional[float]] = mapped_column(Float)
    pivot_s2: Mapped[Optional[float]] = mapped_column(Float)
    pivot_s3: Mapped[Optional[float]] = mapped_column(Float)

    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("symbol_id", "date", "interval", name="uq_indicator"),
        Index("ix_indicator_symbol_date", "symbol_id", "date"),
    )


# ─────────────────────────────────────────────────────────────
# SUPPORT / RESISTANCE
# ─────────────────────────────────────────────────────────────

class SupportResistance(Base):
    """S/R levels engine output — gs-quant risk measure inspired schema."""
    __tablename__ = "support_resistance"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("symbols.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    level_type: Mapped[SRLevel] = mapped_column(SAEnum(SRLevel), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    method: Mapped[str] = mapped_column(String(50))  # swing_hl, fib, pivot, ema, volume_zone
    strength: Mapped[float] = mapped_column(Float, default=1.0)  # 0-10 score
    touches: Mapped[int] = mapped_column(Integer, default=1)
    fib_level: Mapped[Optional[str]] = mapped_column(String(10))  # 0.236, 0.382, 0.5, 0.618
    notes: Mapped[Optional[str]] = mapped_column(Text)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_sr_symbol_date", "symbol_id", "date"),
        Index("ix_sr_level_type", "level_type"),
    )


# ─────────────────────────────────────────────────────────────
# SCANNER RESULTS
# ─────────────────────────────────────────────────────────────

class ScannerResult(Base):
    """Swing scanner output — structured signal with full condition breakdown."""
    __tablename__ = "scanner_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("symbols.id"), nullable=False)
    scan_date: Mapped[date] = mapped_column(Date, nullable=False)
    signal: Mapped[SignalType] = mapped_column(SAEnum(SignalType), nullable=False)
    score: Mapped[float] = mapped_column(Float, default=0.0)  # 0-100

    # Condition flags (each True/False)
    ema20_above_ema50: Mapped[bool] = mapped_column(Boolean, default=False)
    rsi_above_55: Mapped[bool] = mapped_column(Boolean, default=False)
    macd_bullish_cross: Mapped[bool] = mapped_column(Boolean, default=False)
    price_above_supertrend: Mapped[bool] = mapped_column(Boolean, default=False)
    volume_above_avg: Mapped[bool] = mapped_column(Boolean, default=False)
    week52_breakout: Mapped[bool] = mapped_column(Boolean, default=False)

    # Price snapshot at scan time
    close_price: Mapped[float] = mapped_column(Float)
    ema_20: Mapped[Optional[float]] = mapped_column(Float)
    ema_50: Mapped[Optional[float]] = mapped_column(Float)
    rsi: Mapped[Optional[float]] = mapped_column(Float)
    volume: Mapped[Optional[int]] = mapped_column(BigInteger)
    volume_avg_20: Mapped[Optional[float]] = mapped_column(Float)
    week52_high: Mapped[Optional[float]] = mapped_column(Float)

    notes: Mapped[Optional[str]] = mapped_column(Text)
    scanned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_scanner_date_signal", "scan_date", "signal"),
        Index("ix_scanner_symbol_date", "symbol_id", "scan_date"),
        Index("ix_scanner_score", "score"),
    )


# ─────────────────────────────────────────────────────────────
# OI / PCR DATA
# ─────────────────────────────────────────────────────────────

class OIData(Base):
    """Open Interest data for F&O analysis."""
    __tablename__ = "oi_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("symbols.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expiry_date: Mapped[date] = mapped_column(Date, nullable=False)
    strike_price: Mapped[Optional[float]] = mapped_column(Float)
    option_type: Mapped[Optional[str]] = mapped_column(String(2))  # CE / PE

    open_interest: Mapped[int] = mapped_column(BigInteger, default=0)
    oi_change: Mapped[int] = mapped_column(BigInteger, default=0)
    oi_change_pct: Mapped[Optional[float]] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(BigInteger, default=0)
    price: Mapped[Optional[float]] = mapped_column(Float)
    price_change_pct: Mapped[Optional[float]] = mapped_column(Float)

    oi_signal: Mapped[Optional[OISignal]] = mapped_column(SAEnum(OISignal))

    __table_args__ = (
        Index("ix_oi_symbol_ts", "symbol_id", "timestamp"),
        Index("ix_oi_expiry", "expiry_date"),
        Index("ix_oi_signal", "oi_signal"),
    )


class PCRData(Base):
    """Put-Call Ratio tracking."""
    __tablename__ = "pcr_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("symbols.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expiry_date: Mapped[date] = mapped_column(Date, nullable=False)

    pcr_oi: Mapped[float] = mapped_column(Float)        # OI-based PCR
    pcr_volume: Mapped[float] = mapped_column(Float)    # Volume-based PCR
    total_ce_oi: Mapped[int] = mapped_column(BigInteger, default=0)
    total_pe_oi: Mapped[int] = mapped_column(BigInteger, default=0)
    total_ce_volume: Mapped[int] = mapped_column(BigInteger, default=0)
    total_pe_volume: Mapped[int] = mapped_column(BigInteger, default=0)
    max_pain: Mapped[Optional[float]] = mapped_column(Float)
    pcr_signal: Mapped[Optional[PCRSignal]] = mapped_column(SAEnum(PCRSignal))

    __table_args__ = (
        Index("ix_pcr_symbol_ts", "symbol_id", "timestamp"),
    )


# ─────────────────────────────────────────────────────────────
# FUTURES DATA
# ─────────────────────────────────────────────────────────────

class FuturesData(Base):
    """NSE Futures analytics — basis, premium/discount, rollovers."""
    __tablename__ = "futures_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("symbols.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expiry_date: Mapped[date] = mapped_column(Date, nullable=False)

    spot_price: Mapped[float] = mapped_column(Float)
    futures_price: Mapped[float] = mapped_column(Float)
    basis: Mapped[float] = mapped_column(Float)           # futures - spot
    basis_pct: Mapped[float] = mapped_column(Float)       # basis / spot * 100
    open_interest: Mapped[int] = mapped_column(BigInteger, default=0)
    oi_change: Mapped[int] = mapped_column(BigInteger, default=0)
    volume: Mapped[int] = mapped_column(BigInteger, default=0)
    rollover_pct: Mapped[Optional[float]] = mapped_column(Float)

    __table_args__ = (
        Index("ix_futures_symbol_ts", "symbol_id", "timestamp"),
        Index("ix_futures_expiry", "expiry_date"),
    )


# ─────────────────────────────────────────────────────────────
# TRADE SIGNALS
# ─────────────────────────────────────────────────────────────

class TradeSignal(Base):
    """AI-generated trade signals — structured output from Phase 3 AI module."""
    __tablename__ = "trade_signals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("symbols.id"), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    signal: Mapped[SignalType] = mapped_column(SAEnum(SignalType), nullable=False)
    confidence: Mapped[float] = mapped_column(Float)      # 0-100
    risk_level: Mapped[str] = mapped_column(String(10))  # LOW/MEDIUM/HIGH
    entry_price: Mapped[Optional[float]] = mapped_column(Float)
    target_price: Mapped[Optional[float]] = mapped_column(Float)
    stop_loss: Mapped[Optional[float]] = mapped_column(Float)
    risk_reward: Mapped[Optional[float]] = mapped_column(Float)
    reasoning: Mapped[Optional[str]] = mapped_column(Text)
    input_snapshot: Mapped[Optional[dict]] = mapped_column(JSON)
    model_used: Mapped[Optional[str]] = mapped_column(String(50))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        Index("ix_signal_symbol_ts", "symbol_id", "generated_at"),
        Index("ix_signal_type", "signal"),
    )


# ─────────────────────────────────────────────────────────────
# BACKTEST RESULTS
# ─────────────────────────────────────────────────────────────

class BacktestResult(Base):
    """Backtest run metadata and performance summary — Lean/vectorbt hybrid schema."""
    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    symbol_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("symbols.id"))
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    initial_capital: Mapped[float] = mapped_column(Float, nullable=False)
    final_value: Mapped[float] = mapped_column(Float)

    # Core metrics
    cagr: Mapped[Optional[float]] = mapped_column(Float)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float)
    sortino_ratio: Mapped[Optional[float]] = mapped_column(Float)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float)
    calmar_ratio: Mapped[Optional[float]] = mapped_column(Float)
    profit_factor: Mapped[Optional[float]] = mapped_column(Float)
    win_rate: Mapped[Optional[float]] = mapped_column(Float)
    expectancy: Mapped[Optional[float]] = mapped_column(Float)
    total_trades: Mapped[Optional[int]] = mapped_column(Integer)
    winning_trades: Mapped[Optional[int]] = mapped_column(Integer)
    losing_trades: Mapped[Optional[int]] = mapped_column(Integer)
    avg_win: Mapped[Optional[float]] = mapped_column(Float)
    avg_loss: Mapped[Optional[float]] = mapped_column(Float)

    # Parameters used
    parameters: Mapped[Optional[dict]] = mapped_column(JSON)
    equity_curve: Mapped[Optional[dict]] = mapped_column(JSON)  # compressed
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_backtest_strategy", "strategy_name"),
        Index("ix_backtest_run_at", "run_at"),
    )


# ─────────────────────────────────────────────────────────────
# PORTFOLIO
# ─────────────────────────────────────────────────────────────

class Portfolio(Base):
    """Portfolio holdings tracker."""
    __tablename__ = "portfolio"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"))
    symbol_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("symbols.id"), nullable=False)
    asset_type: Mapped[AssetType] = mapped_column(SAEnum(AssetType), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_cost: Mapped[float] = mapped_column(Float, nullable=False)
    current_price: Mapped[Optional[float]] = mapped_column(Float)
    current_value: Mapped[Optional[float]] = mapped_column(Float)
    unrealized_pnl: Mapped[Optional[float]] = mapped_column(Float)
    unrealized_pnl_pct: Mapped[Optional[float]] = mapped_column(Float)
    day_pnl: Mapped[Optional[float]] = mapped_column(Float)
    broker: Mapped[Optional[str]] = mapped_column(String(20))  # zerodha, angel
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_portfolio_user_symbol", "user_id", "symbol_id"),
    )


# ─────────────────────────────────────────────────────────────
# USERS & SETTINGS
# ─────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    telegram_chat_id: Mapped[Optional[str]] = mapped_column(String(50))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    portfolio: Mapped[List["Portfolio"]] = relationship()
    settings: Mapped[List["UserSettings"]] = relationship(back_populates="user")


class UserSettings(Base):
    __tablename__ = "settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    key: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="settings")

    __table_args__ = (UniqueConstraint("user_id", "key", name="uq_user_setting"),)


# ─────────────────────────────────────────────────────────────
# ALERTS & REPORTS
# ─────────────────────────────────────────────────────────────

class Alert(Base):
    """Sent alert history — Telegram / email delivery log."""
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    alert_type: Mapped[str] = mapped_column(String(50))  # scanner, breakout, oi, ai, portfolio
    symbol_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("symbols.id"))
    message: Mapped[str] = mapped_column(Text, nullable=False)
    channel: Mapped[str] = mapped_column(String(20), default="telegram")
    delivered: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[Optional[str]] = mapped_column(Text)
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_alert_type_sent", "alert_type", "sent_at"),
    )


class DailyReport(Base):
    """Auto-generated daily report metadata."""
    __tablename__ = "daily_reports"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    report_date: Mapped[date] = mapped_column(Date, nullable=False, unique=True)
    nifty_close: Mapped[Optional[float]] = mapped_column(Float)
    nifty_change_pct: Mapped[Optional[float]] = mapped_column(Float)
    banknifty_close: Mapped[Optional[float]] = mapped_column(Float)
    banknifty_change_pct: Mapped[Optional[float]] = mapped_column(Float)
    india_vix: Mapped[Optional[float]] = mapped_column(Float)
    total_scanned: Mapped[Optional[int]] = mapped_column(Integer)
    buy_signals: Mapped[Optional[int]] = mapped_column(Integer)
    sell_signals: Mapped[Optional[int]] = mapped_column(Integer)
    watchlist_signals: Mapped[Optional[int]] = mapped_column(Integer)
    pdf_path: Mapped[Optional[str]] = mapped_column(String(500))
    excel_path: Mapped[Optional[str]] = mapped_column(String(500))
    html_path: Mapped[Optional[str]] = mapped_column(String(500))
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
