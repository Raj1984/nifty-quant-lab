"""
NIFTY Quant Lab — Main Entrypoint
====================================
Unified CLI for all platform operations.

Usage:
  python main.py api          — Start FastAPI server
  python main.py setup        — Full first-time data setup
  python main.py scan         — Run swing scanner once
  python main.py download     — Download EOD data
  python main.py indicators   — Compute all indicators
  python main.py report       — Generate daily report
  python main.py dashboard    — Launch Streamlit dashboard
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure project root on Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nifty_quant_lab.utils.logger import setup_logging

setup_logging()

import logging
logger = logging.getLogger("nql.main")


def cmd_api():
    """Start FastAPI + uvicorn server."""
    import uvicorn
    from nifty_quant_lab.config.settings import settings
    logger.info("Starting NIFTY Quant Lab API...")
    uvicorn.run(
        "nifty_quant_lab.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )


def cmd_setup():
    """Full first-time setup: symbols → 10Y history → indicators → scan."""
    async def _run():
        from nifty_quant_lab.database.connection import create_all_tables
        from nifty_quant_lab.data.downloader import NSEDataDownloader
        from nifty_quant_lab.indicators.service import IndicatorService
        from nifty_quant_lab.signals.scanner import SwingScanner
        from nifty_quant_lab.signals.scanner_service import ScannerPersistenceService

        logger.info("=" * 60)
        logger.info("NIFTY Quant Lab — First-Time Setup")
        logger.info("=" * 60)

        # Step 1: Create DB tables
        logger.info("Step 1/5: Creating database tables...")
        await create_all_tables()

        # Step 2: Sync symbol registry
        logger.info("Step 2/5: Syncing symbol registry...")
        dl = NSEDataDownloader()
        await dl.sync_symbol_registry()

        # Step 3: Download historical data
        logger.info("Step 3/5: Downloading 10Y historical data...")
        await dl.download_historical(years=10)

        # Step 4: Compute indicators
        logger.info("Step 4/5: Computing technical indicators...")
        svc = IndicatorService()
        await svc.compute_all_symbols()

        # Step 5: Initial scan
        logger.info("Step 5/5: Running initial swing scan...")
        scanner = SwingScanner()
        session = await scanner.scan_universe()
        persist = ScannerPersistenceService()
        await persist.save_scan_session(session)

        logger.info("=" * 60)
        logger.info(f"Setup complete! {len(session.buy_signals)} BUY signals found.")
        logger.info("Run 'python main.py api' to start the API server.")
        logger.info("Run 'python main.py dashboard' to launch the dashboard.")
        logger.info("=" * 60)

    asyncio.run(_run())


def cmd_scan():
    """Run swing scanner and print results."""
    async def _run():
        from nifty_quant_lab.signals.scanner import SwingScanner
        from nifty_quant_lab.signals.scanner_service import ScannerPersistenceService

        scanner = SwingScanner()
        session = await scanner.scan_universe()

        sys.stdout.buffer.write((scanner.format_signal_summary(session) + "\n").encode("utf-8"))

        persist = ScannerPersistenceService()
        saved = await persist.save_scan_session(session)
        logger.info(f"Saved {saved} results to DB")

    asyncio.run(_run())


def cmd_download():
    """Download latest EOD data."""
    async def _run():
        from nifty_quant_lab.data.downloader import NSEDataDownloader
        dl = NSEDataDownloader()
        await dl.update_today()

    asyncio.run(_run())


def cmd_indicators():
    """Compute all technical indicators."""
    async def _run():
        from nifty_quant_lab.indicators.service import IndicatorService
        svc = IndicatorService()
        results = await svc.compute_all_symbols()
        ok = sum(1 for v in results.values() if v)
        logger.info(f"Indicators: {ok}/{len(results)} symbols processed")

    asyncio.run(_run())


def cmd_report():
    """Generate and send daily report."""
    async def _run():
        from nifty_quant_lab.reports.generator import DailyReportGenerator
        gen = DailyReportGenerator()
        paths = await gen.generate_and_send()
        logger.info(f"Report generated: {paths}")

    asyncio.run(_run())


def cmd_dashboard():
    """Launch Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    logger.info(f"Launching dashboard: {dashboard_path}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", "8501",
        "--server.headless", "true",
    ])


def main():
    commands = {
        "api": cmd_api,
        "setup": cmd_setup,
        "scan": cmd_scan,
        "download": cmd_download,
        "indicators": cmd_indicators,
        "report": cmd_report,
        "dashboard": cmd_dashboard,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("NIFTY Quant Lab v1.0")
        print("\nUsage: python main.py <command>")
        print("\nAvailable commands:")
        for cmd, fn in commands.items():
            print(f"  {cmd:<15} {fn.__doc__.strip().splitlines()[0]}")
        sys.exit(0)

    cmd = sys.argv[1]
    logger.info(f"Running command: {cmd}")
    commands[cmd]()


if __name__ == "__main__":
    main()
