"""
NIFTY Quant Lab — NSE Option Chain Scraper
============================================
Fetches live option chain data from NSE India.

IMPORTANT — endpoint history:
NSE retired the old `option-chain-indices` / `option-chain-equities` JSON
endpoints (the ones this module originally hit directly via httpx). They
now return 404. NSE replaced them with an `option-chain-v3` endpoint that
also requires a different cookie-setting path (`/option-chain`, not the
bare homepage) and a stricter header/cookie lifecycle than before.

Rather than re-reverse-engineer NSE's anti-scraping measures by hand, this
module wraps the actively-maintained `nse` PyPI package (BennyThadikaran/
NseIndiaApi, MIT licensed), which tracks NSE's endpoint changes and handles
cookie persistence correctly. Install: `pip install nse`.

The `nse` package is synchronous (uses `requests`), so calls are wrapped
in a ThreadPoolExecutor for async compatibility — the same pattern used
by data/providers/yfinance_provider.py elsewhere in this codebase.

Supported symbols: NIFTY, BANKNIFTY, FINNIFTY (lowercase at the API layer;
this module accepts either case and normalizes internally).

Known limitation: NSE blocks requests from server/datacenter IPs (AWS,
Azure, GCP, etc.) — see https://github.com/BennyThadikaran/NseIndiaApi/issues/9.
This works from residential/office IPs (e.g. running locally), but will
fail with connection errors if deployed to a cloud VM without a residential
proxy.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from nifty_quant_lab.utils.logger import get_logger

logger = get_logger("nse_scraper")

try:
    from nse import NSE as _NSEClient
    _HAS_NSE_PACKAGE = True
except ImportError:
    _HAS_NSE_PACKAGE = False

# ── Index symbols mapped to the nse package's lowercase convention
INDEX_SYMBOLS = {
    "NIFTY":      "nifty",
    "NIFTY50":    "nifty",
    "BANKNIFTY":  "banknifty",
    "FINNIFTY":   "finnifty",
    "NIFTYIT":    "niftyit",
    "MIDCPNIFTY": "nifty",  # nse package has no separate midcap futures symbol yet
}


# ─────────────────────────────────────────────────────────────
# DATA MODELS  (unchanged — downstream code depends on this shape)
# ─────────────────────────────────────────────────────────────

@dataclass
class OptionRow:
    """Single strike row from NSE option chain."""
    strike: float
    expiry: str

    # Call side
    ce_oi: int = 0
    ce_oi_change: int = 0
    ce_volume: int = 0
    ce_iv: float = 0.0
    ce_ltp: float = 0.0
    ce_bid: float = 0.0
    ce_ask: float = 0.0

    # Put side
    pe_oi: int = 0
    pe_oi_change: int = 0
    pe_volume: int = 0
    pe_iv: float = 0.0
    pe_ltp: float = 0.0
    pe_bid: float = 0.0
    pe_ask: float = 0.0

    @property
    def pcr_oi(self) -> Optional[float]:
        return round(self.pe_oi / self.ce_oi, 3) if self.ce_oi > 0 else None

    @property
    def oi_diff(self) -> int:
        """CE OI - PE OI. Positive = more call writers = bearish bias."""
        return self.ce_oi - self.pe_oi


@dataclass
class OptionChainSnapshot:
    """Full option chain for one symbol + expiry at one point in time."""
    symbol: str
    expiry: str
    timestamp: datetime
    spot_price: float
    rows: List[OptionRow] = field(default_factory=list)

    # Aggregates computed after fetch
    total_ce_oi: int = 0
    total_pe_oi: int = 0
    total_ce_volume: int = 0
    total_pe_volume: int = 0
    pcr_oi: float = 0.0
    pcr_volume: float = 0.0
    max_pain: Optional[float] = None
    atm_strike: Optional[float] = None

    @property
    def available_expiries(self) -> List[str]:
        return sorted({r.expiry for r in self.rows})

    def rows_for_expiry(self, expiry: str) -> List[OptionRow]:
        return [r for r in self.rows if r.expiry == expiry]

    def atm_rows(self, n: int = 10) -> List[OptionRow]:
        """Return N strikes above and below ATM."""
        if not self.atm_strike:
            return self.rows
        sorted_rows = sorted(self.rows, key=lambda r: abs(r.strike - self.atm_strike))
        return sorted_rows[:n * 2]


@dataclass
class ScraperResult:
    success: bool
    data: Optional[OptionChainSnapshot] = None
    error: Optional[str] = None
    fetched_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def ok(cls, data: OptionChainSnapshot) -> "ScraperResult":
        return cls(success=True, data=data)

    @classmethod
    def err(cls, error: str) -> "ScraperResult":
        return cls(success=False, error=error)


# ─────────────────────────────────────────────────────────────
# NSE SESSION WRAPPER  (wraps the `nse` package's sync client)
# ─────────────────────────────────────────────────────────────

class NSESession:
    """
    Thin async wrapper around the `nse` package's NSE() client.

    The underlying client manages its own cookie persistence to disk
    (in a temp directory here) and handles NSE's cookie expiry/refresh
    cycle internally — we don't need to reimplement that logic.
    """

    def __init__(self):
        if not _HAS_NSE_PACKAGE:
            raise ImportError(
                "The 'nse' package is required for live NSE data. "
                "Install with: pip install nse"
            )
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="nse")
        self._cookie_dir = Path(tempfile.gettempdir()) / "nql_nse_cookies"
        self._cookie_dir.mkdir(exist_ok=True)
        self._client: Optional[_NSEClient] = None

    def _get_client(self) -> "_NSEClient":
        if self._client is None:
            self._client = _NSEClient(download_folder=str(self._cookie_dir), server=False)
        return self._client

    async def option_chain(self, symbol: str, expiry_date: Optional[datetime] = None) -> dict:
        """Fetch raw option chain JSON for a symbol via the nse package."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._get_client().optionChain(symbol, expiry_date),
        )

    async def quote_derivative(self, symbol: str) -> dict:
        """Fetch raw futures/derivative quote JSON for a symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._get_client().quote(symbol, type="fno"),
        )

    async def market_status(self) -> list:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._get_client().status(),
        )

    async def close(self) -> None:
        if self._client is not None:
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._client.exit
            )
            self._client = None
        self._executor.shutdown(wait=False)


# ─────────────────────────────────────────────────────────────
# NSE OPTION CHAIN SCRAPER
# ─────────────────────────────────────────────────────────────

class NSEOptionChainScraper:
    """
    Fetches and parses NSE option chain data.

    Usage:
        scraper = NSEOptionChainScraper()
        result = await scraper.fetch_option_chain("NIFTY")
        if result.success:
            snapshot = result.data
            print(f"PCR: {snapshot.pcr_oi}")
    """

    def __init__(self, session: Optional[NSESession] = None):
        self.session = session or NSESession()

    async def fetch_option_chain(
        self,
        symbol: str,
        expiry: Optional[str] = None,
    ) -> ScraperResult:
        """
        Fetch full option chain for an index.

        Args:
            symbol: NIFTY, BANKNIFTY, FINNIFTY (case-insensitive)
            expiry: Specific expiry date string e.g. "26-Jun-2025".
                    If None, the nse package auto-resolves the nearest expiry.
        """
        nse_symbol = INDEX_SYMBOLS.get(symbol.upper(), symbol.lower())

        try:
            raw = await self.session.option_chain(nse_symbol)
        except Exception as e:
            logger.error(f"NSE option chain fetch failed for {symbol}: {e}")
            return ScraperResult.err(str(e))

        try:
            snapshot = self._parse_option_chain(raw, symbol, expiry)
            logger.info(
                f"✓ {symbol} option chain: {len(snapshot.rows)} strikes | "
                f"spot={snapshot.spot_price} | PCR={snapshot.pcr_oi:.2f} | "
                f"expiry={snapshot.expiry}"
            )
            return ScraperResult.ok(snapshot)
        except Exception as e:
            logger.error(f"NSE parse error for {symbol}: {e}", exc_info=True)
            return ScraperResult.err(f"Parse error: {e}")

    def _parse_option_chain(
        self,
        raw: dict,
        symbol: str,
        target_expiry: Optional[str],
    ) -> OptionChainSnapshot:
        """
        Parse NSE JSON response into OptionChainSnapshot.

        Schema is unchanged from the old endpoint — records.data[] with
        strikePrice/expiryDate/CE{}/PE{} — only the transport changed.
        """
        records = raw.get("records", {})
        data = raw.get("filtered", {}).get("data", records.get("data", []))

        spot = float(
            records.get("underlyingValue", 0)
            or raw.get("filtered", {}).get("CE", {}).get("underlyingValue", 0)
            or 0
        )

        expiry_dates = records.get("expiryDates", [])
        chosen_expiry = target_expiry or (expiry_dates[0] if expiry_dates else "")

        rows: List[OptionRow] = []
        total_ce_oi = total_pe_oi = 0
        total_ce_vol = total_pe_vol = 0

        for item in data:
            strike = float(item.get("strikePrice", 0))
            expiry_str = item.get("expiryDate", chosen_expiry)

            if target_expiry and expiry_str != target_expiry:
                continue

            ce = item.get("CE", {})
            pe = item.get("PE", {})

            row = OptionRow(
                strike=strike,
                expiry=expiry_str,
                ce_oi=int(ce.get("openInterest", 0)),
                ce_oi_change=int(ce.get("changeinOpenInterest", 0)),
                ce_volume=int(ce.get("totalTradedVolume", 0)),
                ce_iv=float(ce.get("impliedVolatility", 0)),
                ce_ltp=float(ce.get("lastPrice", 0)),
                ce_bid=float(ce.get("bidprice", 0)),
                ce_ask=float(ce.get("askPrice", 0)),
                pe_oi=int(pe.get("openInterest", 0)),
                pe_oi_change=int(pe.get("changeinOpenInterest", 0)),
                pe_volume=int(pe.get("totalTradedVolume", 0)),
                pe_iv=float(pe.get("impliedVolatility", 0)),
                pe_ltp=float(pe.get("lastPrice", 0)),
                pe_bid=float(pe.get("bidprice", 0)),
                pe_ask=float(pe.get("askPrice", 0)),
            )
            rows.append(row)
            total_ce_oi += row.ce_oi
            total_pe_oi += row.pe_oi
            total_ce_vol += row.ce_volume
            total_pe_vol += row.pe_volume

        pcr_oi = round(total_pe_oi / total_ce_oi, 3) if total_ce_oi > 0 else 0.0
        pcr_vol = round(total_pe_vol / total_ce_vol, 3) if total_ce_vol > 0 else 0.0
        atm = min(rows, key=lambda r: abs(r.strike - spot)).strike if rows else None

        snapshot = OptionChainSnapshot(
            symbol=symbol,
            expiry=chosen_expiry,
            timestamp=datetime.now(),
            spot_price=spot,
            rows=rows,
            total_ce_oi=total_ce_oi,
            total_pe_oi=total_pe_oi,
            total_ce_volume=total_ce_vol,
            total_pe_volume=total_pe_vol,
            pcr_oi=pcr_oi,
            pcr_volume=pcr_vol,
            atm_strike=atm,
        )
        snapshot.max_pain = self._compute_max_pain(rows)
        return snapshot

    @staticmethod
    def _compute_max_pain(rows: List[OptionRow]) -> Optional[float]:
        """
        Max Pain = strike where total option buyer loss is maximised.
        For each candidate strike, sum (intrinsic value × OI) across all
        CE + PE; the strike with minimum total payout is max pain.
        """
        if not rows:
            return None

        strikes = sorted({r.strike for r in rows})
        oi_by_strike: Dict[float, OptionRow] = {r.strike: r for r in rows}

        min_pain = float("inf")
        max_pain_strike = strikes[0]

        for test_strike in strikes:
            total_pain = 0.0
            for strike, row in oi_by_strike.items():
                total_pain += max(test_strike - strike, 0) * row.ce_oi
                total_pain += max(strike - test_strike, 0) * row.pe_oi
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike

        return max_pain_strike

    async def fetch_market_status(self) -> Dict:
        """Check if NSE market is open."""
        try:
            result = await self.session.market_status()
            return {"data": result}
        except Exception as e:
            return {"error": str(e)}

    async def fetch_all_indices(self) -> Dict[str, ScraperResult]:
        """Fetch option chains for NIFTY + BANKNIFTY concurrently."""
        tasks = {
            sym: self.fetch_option_chain(sym)
            for sym in ["NIFTY", "BANKNIFTY", "FINNIFTY"]
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            sym: (r if isinstance(r, ScraperResult) else ScraperResult.err(str(r)))
            for sym, r in zip(tasks.keys(), results)
        }

    async def close(self) -> None:
        await self.session.close()
