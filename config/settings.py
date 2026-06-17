"""
NIFTY Quant Lab - Configuration Settings
=========================================
Inspired by gs-quant's configuration patterns and OpenBB's provider architecture.
All values configurable via environment variables — no hardcoded secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


@dataclass(frozen=True)
class DatabaseConfig:
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "3306")))
    name: str = field(default_factory=lambda: os.getenv("DB_NAME", "nifty_quant_lab"))
    user: str = field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    pool_size: int = field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "10")))
    max_overflow: int = field(default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "20")))

    @property
    def url(self) -> str:
        # aiomysql async driver
        return (
            f"mysql+aiomysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}?charset=utf8mb4"
        )

    @property
    def sync_url(self) -> str:
        # PyMySQL sync driver (used by Alembic)
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}?charset=utf8mb4"
        )


@dataclass(frozen=True)
class RedisConfig:
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    alert_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_ALERT_CHAT_ID", ""))

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)


@dataclass(frozen=True)
class BrokerConfig:
    # Zerodha KiteConnect
    zerodha_api_key: str = field(default_factory=lambda: os.getenv("ZERODHA_API_KEY", ""))
    zerodha_api_secret: str = field(default_factory=lambda: os.getenv("ZERODHA_API_SECRET", ""))
    zerodha_user_id: str = field(default_factory=lambda: os.getenv("ZERODHA_USER_ID", "PR5573"))
    # Angel One
    angel_api_key: str = field(default_factory=lambda: os.getenv("ANGEL_API_KEY", ""))
    angel_client_id: str = field(default_factory=lambda: os.getenv("ANGEL_CLIENT_ID", ""))
    angel_password: str = field(default_factory=lambda: os.getenv("ANGEL_PASSWORD", ""))
    angel_totp_secret: str = field(default_factory=lambda: os.getenv("ANGEL_TOTP_SECRET", ""))


@dataclass(frozen=True)
class AIConfig:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1"))
    use_ollama: bool = field(
        default_factory=lambda: os.getenv("USE_OLLAMA", "false").lower() == "true"
    )


@dataclass(frozen=True)
class MarketConfig:
    """NSE market structure constants."""

    # Index instrument tokens (Zerodha)
    NIFTY_TOKEN: int = 256265
    BANKNIFTY_TOKEN: int = 260105
    FINNIFTY_TOKEN: int = 257801
    MIDCAP_TOKEN: int = 288009
    INDIA_VIX_TOKEN: int = 264969

    # Trading hours (IST)
    MARKET_OPEN: str = "09:15"
    MARKET_CLOSE: str = "15:30"
    PRE_MARKET_OPEN: str = "09:00"

    # Exchange
    EXCHANGE: str = "NSE"
    EXCHANGE_FO: str = "NFO"  # Futures & Options

    # Brokerage (Zerodha flat fee)
    BROKERAGE_PER_ORDER: float = 20.0
    STT_EQUITY_DELIVERY: float = 0.001  # 0.1%
    STT_EQUITY_INTRADAY: float = 0.00025  # 0.025%
    STT_FO: float = 0.000625  # 0.0625% on sell side
    SEBI_CHARGES: float = 0.000001  # ₹10 per crore
    STAMP_DUTY: float = 0.00003  # 0.003%
    NSE_TRANSACTION: float = 0.0000335  # NSE txn charge
    GST_RATE: float = 0.18

    # NIFTY lot size
    NIFTY_LOT_SIZE: int = 25
    BANKNIFTY_LOT_SIZE: int = 15
    FINNIFTY_LOT_SIZE: int = 40

    # Expiry schedule (post-2024 NSE revision)
    NIFTY_EXPIRY_DAY: str = "Tuesday"    # Weekly
    BANKNIFTY_EXPIRY_DAY: str = "Tuesday"  # Monthly last Tuesday

    # Data intervals supported
    INTRADAY_INTERVALS: List[str] = field(
        default_factory=lambda: ["1m", "5m", "15m", "30m", "1h"]
    )


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    # API server
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    api_reload: bool = field(
        default_factory=lambda: os.getenv("API_RELOAD", "false").lower() == "true"
    )
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def data_dir(self) -> Path:
        p = self.base_dir / "data" / "storage"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def logs_dir(self) -> Path:
        p = self.base_dir / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def reports_dir(self) -> Path:
        p = self.base_dir / "reports" / "output"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Sub-configs
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    market: MarketConfig = field(default_factory=MarketConfig)


# Singleton instance
settings = AppConfig()

# NSE 50 constituents (Phase 1 scan universe)
NIFTY50_SYMBOLS: List[str] = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "M&M",
    "LT", "BHARTIARTL", "HINDALCO", "AXISBANK", "SBIN",
    "HCLTECH", "BAJFINANCE", "TCS", "TATASTEEL", "ADANIENT",
    "NTPC", "ETERNAL", "SHRIRAMFIN", "ONGC", "ITC",
    "INDIGO", "ADANIPORTS", "BEL", "MARUTI", "JIOFIN",
    "KOTAKBANK", "BAJAJFINSV", "EICHERMOT", "TITAN", "COALINDIA",
    "TATACONSUM", "HINDUNILVR", "TECHM", "ULTRACEMCO", "ASIANPAINT",
    "NESTLEIND", "TMPV", "SBILIFE", "SUNPHARMA", "TRENT",
    "POWERGRID", "MAXHEALTH", "APOLLOHOSP", "WIPRO", "BAJAJ-AUTO",
    "JSWSTEEL", "HDFCLIFE", "GRASIM", "CIPLA", "DRREDDY",
]

# Sectoral indices
SECTORAL_INDICES: List[str] = [
    "NIFTY_IT", "NIFTY_BANK", "NIFTY_PHARMA", "NIFTY_AUTO",
    "NIFTY_FMCG", "NIFTY_METAL", "NIFTY_REALTY", "NIFTY_ENERGY",
    "NIFTY_INFRA", "NIFTY_MEDIA", "NIFTY_PSU_BANK", "NIFTY_FIN_SERVICE",
]
