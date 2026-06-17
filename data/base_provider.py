"""
NIFTY Quant Lab - Abstract Data Provider
==========================================
OpenBB-inspired provider abstraction layer. Every data source (yfinance, Zerodha,
Angel One, NSE direct) implements this interface — swap providers without touching
business logic.

Architecture inspired by:
- OpenBB's openbb_core/provider/abstract/data.py  (Data model)
- OpenBB's router.py  (provider registration)
- gs-quant's processor.py  (abstract processor pattern)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict


# ─────────────────────────────────────────────────────────────
# STANDARDIZED DATA MODELS  (OpenBB Data contract)
# ─────────────────────────────────────────────────────────────

class OHLCVData(BaseModel):
    """Standardized OHLCV record — every provider output maps to this."""
    model_config = ConfigDict(populate_by_name=True)

    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    adjusted_close: Optional[float] = None

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)


class ProviderMetadata(BaseModel):
    """Provider capabilities and rate-limit info."""
    name: str
    supports_intraday: bool = False
    supports_historical: bool = True
    supports_options: bool = False
    supports_futures: bool = False
    max_records_per_call: int = 500
    rate_limit_per_minute: int = 60
    requires_auth: bool = False


@dataclass
class DataFetchResult:
    """Wrapper around provider output — success or structured error."""
    success: bool
    data: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    provider: str = ""
    records: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, df: pd.DataFrame, provider: str) -> "DataFetchResult":
        return cls(success=True, data=df, provider=provider, records=len(df))

    @classmethod
    def err(cls, error: str, provider: str) -> "DataFetchResult":
        return cls(success=False, error=error, provider=provider)


# ─────────────────────────────────────────────────────────────
# ABSTRACT PROVIDER BASE
# ─────────────────────────────────────────────────────────────

class BaseDataProvider(abc.ABC):
    """
    Abstract base class for all data providers.

    gs-quant processor pattern: define the interface once, implement per source.
    Every concrete provider must implement all abstract methods.
    """

    @property
    @abc.abstractmethod
    def metadata(self) -> ProviderMetadata:
        """Provider capabilities declaration."""
        ...

    @abc.abstractmethod
    async def fetch_historical(
        self,
        symbol: str,
        start: date,
        end: date,
        exchange: str = "NSE",
    ) -> DataFetchResult:
        """Fetch EOD OHLCV data for a symbol."""
        ...

    @abc.abstractmethod
    async def fetch_intraday(
        self,
        symbol: str,
        interval: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: int = 30,
    ) -> DataFetchResult:
        """Fetch intraday OHLCV data."""
        ...

    async def fetch_multiple(
        self,
        symbols: List[str],
        start: date,
        end: date,
        exchange: str = "NSE",
    ) -> Dict[str, DataFetchResult]:
        """
        Bulk fetch — default implementation calls fetch_historical sequentially.
        Providers with bulk APIs should override this.
        """
        results: Dict[str, DataFetchResult] = {}
        for symbol in symbols:
            results[symbol] = await self.fetch_historical(symbol, start, end, exchange)
        return results

    def validate_interval(self, interval: str) -> None:
        """Validate interval string."""
        valid = {"1m", "5m", "15m", "30m", "1h", "1d", "1w", "1mo"}
        if interval not in valid:
            raise ValueError(f"Invalid interval '{interval}'. Valid: {valid}")

    @staticmethod
    def to_standard_df(records: List[Dict]) -> pd.DataFrame:
        """
        Convert list of dicts to standardized DataFrame.
        Ensures: date index, float dtypes, sorted ascending.
        """
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


# ─────────────────────────────────────────────────────────────
# PROVIDER REGISTRY
# ─────────────────────────────────────────────────────────────

class ProviderRegistry:
    """
    Singleton registry of all data providers.
    OpenBB router pattern — register once, resolve by name.
    """

    _instance: Optional["ProviderRegistry"] = None
    _providers: Dict[str, BaseDataProvider] = {}

    def __new__(cls) -> "ProviderRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, name: str, provider: BaseDataProvider) -> None:
        self._providers[name] = provider

    def get(self, name: str) -> BaseDataProvider:
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered. Available: {list(self._providers)}")
        return self._providers[name]

    def list_providers(self) -> List[str]:
        return list(self._providers.keys())

    def get_default(self) -> BaseDataProvider:
        """Return first available provider."""
        if not self._providers:
            raise RuntimeError("No data providers registered.")
        return next(iter(self._providers.values()))


# Singleton
provider_registry = ProviderRegistry()
