"""
volatility.py — test with 1Y real NIFTY50 data from yfinance
"""

import sys, types, logging

# ── Patch missing project logger ──────────────────────────────
def _get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.DEBUG)
    return logger

nql_pkg   = types.ModuleType("nifty_quant_lab")
nql_utils = types.ModuleType("nifty_quant_lab.utils")
nql_log   = types.ModuleType("nifty_quant_lab.utils.logger")
nql_log.get_logger = _get_logger
nql_pkg.utils = nql_utils
nql_utils.logger = nql_log
sys.modules.update({
    "nifty_quant_lab":              nql_pkg,
    "nifty_quant_lab.utils":        nql_utils,
    "nifty_quant_lab.utils.logger": nql_log,
})

import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("volatility", pathlib.Path(__file__).parent / "volatility.py")
vol_mod = importlib.util.module_from_spec(spec)
sys.modules["volatility"] = vol_mod
spec.loader.exec_module(vol_mod)

import yfinance as yf
import pandas as pd

SEP = "─" * 65

# ── Fetch 1Y NIFTY50 + BANKNIFTY ──────────────────────────────
SYMBOLS = {"NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK"}

for label, ticker in SYMBOLS.items():
    print(f"\n{'═'*65}")
    print(f"  {label}  ({ticker})  — 1 Year Daily")
    print(f"{'═'*65}")

    raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
    if raw.empty:
        print("  ✗ No data returned")
        continue

    # yfinance returns MultiIndex columns when auto_adjust=True; flatten
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    # Rename 'adj close' → 'close' if present
    if "adj close" in raw.columns:
        raw = raw.rename(columns={"adj close": "close"})

    df = raw[["open", "high", "low", "close", "volume"]].dropna()
    print(f"  Bars loaded : {len(df)}  ({df.index[0].date()} → {df.index[-1].date()})")
    print(f"  Latest close: {df['close'].iloc[-1]:,.1f}")

    engine = vol_mod.VolatilityEngine()
    result = engine.compute(df, symbol=label, windows=(20, 60))

    if not result.success:
        print(f"  ✗ Engine error: {result.error}")
        continue

    out = result.df
    reg = result.regime

    # ── Vol table: last 10 rows ────────────────────────────────
    print(f"\n{SEP}")
    print("  Volatility columns — last 10 bars")
    print(SEP)
    cols = ["close", "rvol_20d", "rvol_60d", "ewma_vol", "parkinson_vol", "gk_vol"]
    print(out[cols].tail(10).to_string())

    # ── Summary stats ──────────────────────────────────────────
    print(f"\n{SEP}")
    print("  1Y Stats for realized vol (20d)")
    print(SEP)
    rv = out["rvol_20d"].dropna()
    print(f"  Min    : {rv.min():.1f}%")
    print(f"  Max    : {rv.max():.1f}%")
    print(f"  Mean   : {rv.mean():.1f}%")
    print(f"  Median : {rv.median():.1f}%")
    print(f"  25th   : {rv.quantile(0.25):.1f}%")
    print(f"  75th   : {rv.quantile(0.75):.1f}%")
    print(f"  95th   : {rv.quantile(0.95):.1f}%")

    # ── Regime result ──────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Current Vol Regime")
    print(SEP)
    print(f"  Realized vol (20d) : {reg.current_vol}%")
    print(f"  EWMA vol           : {reg.ewma_vol}%")
    print(f"  Parkinson vol      : {reg.parkinson_vol}%")
    print(f"  Percentile (1Y)    : {reg.vol_percentile}th")
    print(f"  Regime             : {reg.regime}")
    print(f"  Trend              : {reg.trend}")
    print(f"  Vol-of-vol         : {reg.vol_of_vol}")
    print(f"  → {reg.interpretation}")

    # ── Estimator comparison: are they in agreement? ──────────────
    print(f"\n{SEP}")
    print("  Estimator convergence (last bar)")
    print(SEP)
    latest = out[["rvol_20d", "ewma_vol", "parkinson_vol", "gk_vol"]].iloc[-1]
    spread = latest.max() - latest.min()
    print(f"  rvol_20d    : {latest['rvol_20d']}%")
    print(f"  ewma_vol    : {latest['ewma_vol']}%")
    print(f"  parkinson   : {latest['parkinson_vol']}%")
    print(f"  garman-klass: {latest['gk_vol']}%")
    print(f"  Spread      : {spread:.2f}%  {'⚠ diverged' if spread > 5 else '✓ converged'}")
