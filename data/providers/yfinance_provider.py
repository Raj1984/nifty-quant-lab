"""
NIFTY Quant Lab - Yahoo Finance Data Provider
===============================================
Primary free data provider for NSE historical and intraday data.
Uses yfinance with NSE symbol convention (symbol.NS / symbol.BO).

Supports:
- NIFTY50, BANKNIFTY, FINNIFTY indices
- NSE equity historical (10Y daily)
- NSE intraday: 1m (7d), 5m/15m (60d), 30m/1h (730d)
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from nifty_quant_lab.data.base_provider import (
    BaseDataProvider,
    DataFetchResult,
    ProviderMetadata,
)

logger = logging.getLogger("nql.provider.yfinance")

# NSE index ticker mappings (Yahoo Finance convention)
INDEX_TICKER_MAP: Dict[str, str] = {
    "NIFTY50": "^NSEI",
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCAP50": "^NSEMDCP50",
    "INDIA_VIX": "^INDIAVIX",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_FMCG": "^CNXFMCG",
    "NIFTY_METAL": "^CNXMETAL",
    "NIFTY_REALTY": "^CNXREALTY",
    "NIFTY_ENERGY": "^CNXENERGY",
    "NIFTY_PSU_BANK": "^CNXPSUBANK",
    "NIFTY_FIN_SERVICE": "^CNXFINANCE",
}

# NSE equity symbols whose Yahoo Finance ticker differs from symbol.NS convention
EQUITY_TICKER_OVERRIDES: Dict[str, str] = {}

# yfinance interval limits
YFINANCE_INTERVAL_LIMITS: Dict[str, int] = {
    "1m": 7,       # max 7 days
    "5m": 60,      # max 60 days
    "15m": 60,     # max 60 days
    "30m": 730,    # max 730 days
    "60m": 730,    # max 730 days
    "1h": 730,     # max 730 days
    "1d": 3650,    # 10 years
    "1wk": 3650,
    "1mo": 3650,
}

# Internal interval → yfinance interval string
INTERVAL_MAP: Dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "1d": "1d",
    "1w": "1wk",
    "1mo": "1mo",
}


def _nse_ticker(symbol: str) -> str:
    """Convert NSE symbol to Yahoo Finance ticker."""
    symbol = symbol.upper().strip()
    if symbol in INDEX_TICKER_MAP:
        return INDEX_TICKER_MAP[symbol]
    if symbol in EQUITY_TICKER_OVERRIDES:
        return EQUITY_TICKER_OVERRIDES[symbol]
    if symbol.startswith("^") or symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    return f"{symbol}.NS"


def _clean_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Standardize DataFrame from yfinance output."""
    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns from yfinance v0.2+
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    rename_map = {
        "adj_close": "adjusted_close",
        "adj close": "adjusted_close",
    }
    df = df.rename(columns=rename_map)

    # Ensure required columns
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            df[col] = float("nan")

    df["symbol"] = symbol
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()

    # Drop rows with all NaN OHLC
    df = df.dropna(subset=["open", "high", "low", "close"], how="all")

    # Cast numerics
    for col in ["open", "high", "low", "close", "adjusted_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

    return df


class YFinanceProvider(BaseDataProvider):
    """
    Yahoo Finance data provider for NSE market data.

    Wraps yfinance in thread pool for async compatibility.
    Implements OpenBB's provider interface pattern.
    """

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="yf")

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name="yfinance",
            supports_intraday=True,
            supports_historical=True,
            supports_options=True,
            supports_futures=False,
            max_records_per_call=5000,
            rate_limit_per_minute=100,
            requires_auth=False,
        )

    async def fetch_historical(
        self,
        symbol: str,
        start: date,
        end: date,
        exchange: str = "NSE",
    ) -> DataFetchResult:
        """Fetch EOD historical OHLCV data."""
        ticker = _nse_ticker(symbol)
        logger.debug(f"Fetching historical {ticker}: {start} → {end}")
        try:
            df = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: self._download_historical(ticker, start, end),
            )
            df = _clean_df(df, symbol)
            if df.empty:
                return DataFetchResult.err(f"No data returned for {symbol}", "yfinance")
            logger.info(f"✓ {symbol}: {len(df)} daily bars ({df.index[0].date()} → {df.index[-1].date()})")
            return DataFetchResult.ok(df, "yfinance")
        except Exception as e:
            logger.error(f"✗ {symbol} historical fetch failed: {e}")
            return DataFetchResult.err(str(e), "yfinance")

    def _download_historical(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        t = yf.Ticker(ticker)
        return t.history(
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            back_adjust=False,
        )

    async def fetch_intraday(
        self,
        symbol: str,
        interval: str = "5m",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: int = 30,
    ) -> DataFetchResult:
        """
        Fetch intraday OHLCV.
        Automatically handles yfinance max-days limits per interval.
        """
        self.validate_interval(interval)
        yf_interval = INTERVAL_MAP.get(interval, interval)
        max_days = YFINANCE_INTERVAL_LIMITS.get(interval, 60)

        # Clamp days to yfinance limits
        days = min(days, max_days)
        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=days)

        ticker = _nse_ticker(symbol)
        logger.debug(f"Fetching intraday {ticker} [{interval}]: {start} → {end}")

        try:
            df = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: self._download_intraday(ticker, yf_interval, start, end),
            )
            df = _clean_df(df, symbol)
            if df.empty:
                return DataFetchResult.err(f"No intraday data for {symbol}", "yfinance")
            # Filter to market hours IST (09:15–15:30)
            df = self._filter_market_hours(df)
            logger.info(f"✓ {symbol} [{interval}]: {len(df)} bars")
            return DataFetchResult.ok(df, "yfinance")
        except Exception as e:
            logger.error(f"✗ {symbol} intraday [{interval}] fetch failed: {e}")
            return DataFetchResult.err(str(e), "yfinance")

    def _download_intraday(
        self,
        ticker: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        t = yf.Ticker(ticker)
        return t.history(
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
        )

    @staticmethod
    def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
        """Keep only NSE RTH bars: 09:15–15:30 IST."""
        if df.empty:
            return df
        # Convert timezone if needed
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        idx_ist = idx.tz_convert("Asia/Kolkata")
        mask = (
            (idx_ist.hour > 9) | ((idx_ist.hour == 9) & (idx_ist.minute >= 15))
        ) & (
            (idx_ist.hour < 15) | ((idx_ist.hour == 15) & (idx_ist.minute <= 30))
        )
        return df[mask]

    async def fetch_multiple(
        self,
        symbols: List[str],
        start: date,
        end: date,
        exchange: str = "NSE",
    ) -> Dict[str, DataFetchResult]:
        """
        Bulk fetch using yfinance batch download.
        Much faster than sequential — one API call for all tickers.
        """
        tickers = [_nse_ticker(s) for s in symbols]
        ticker_to_symbol = dict(zip(tickers, symbols))

        logger.info(f"Bulk fetching {len(symbols)} symbols via yfinance...")
        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: yf.download(
                    tickers=tickers,
                    start=start.strftime("%Y-%m-%d"),
                    end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                    interval="1d",
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                ),
            )
        except Exception as e:
            logger.error(f"Bulk download failed: {e}")
            # Fall back to sequential
            return await super().fetch_multiple(symbols, start, end, exchange)

        results: Dict[str, DataFetchResult] = {}
        for ticker, symbol in ticker_to_symbol.items():
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    df = raw[ticker].copy() if ticker in raw.columns.get_level_values(0) else pd.DataFrame()
                else:
                    df = raw.copy()
                df = _clean_df(df, symbol)
                if df.empty:
                    results[symbol] = DataFetchResult.err(f"No data for {symbol}", "yfinance")
                else:
                    results[symbol] = DataFetchResult.ok(df, "yfinance")
            except Exception as e:
                results[symbol] = DataFetchResult.err(str(e), "yfinance")

        success = sum(1 for r in results.values() if r.success)
        logger.info(f"Bulk fetch complete: {success}/{len(symbols)} succeeded")
        return results

    @lru_cache(maxsize=128)
    def get_symbol_info(self, symbol: str) -> dict:
        """Fetch symbol fundamentals (cached)."""
        ticker = _nse_ticker(symbol)
        t = yf.Ticker(ticker)
        try:
            info = t.info
            return {
                "name": info.get("longName", symbol),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE"),
                "beta": info.get("beta"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
            }
        except Exception:
            return {"name": symbol, "sector": "", "industry": ""}

    def __del__(self):
        self._executor.shutdown(wait=False)
