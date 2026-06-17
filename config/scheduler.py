"""
NIFTY Quant Lab - Scheduled Jobs
===================================
APScheduler job definitions for automated data pipeline.

Schedule:
  08:30 IST  — Pre-market data sync (symbol registry)
  15:35 IST  — Post-market EOD data download
  16:00 IST  — Indicator computation
  16:15 IST  — Swing scanner run
  16:30 IST  — S/R level computation
  17:00 IST  — Daily report generation + Telegram alerts
  Every 5m   — Intraday price update (market hours)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import pytz

from nifty_quant_lab.config.settings import settings
from nifty_quant_lab.utils.logger import get_logger

logger = get_logger("scheduler")

IST = pytz.timezone("Asia/Kolkata")


class QuantLabScheduler:
    """
    Centralized job scheduler.
    All jobs are IST-timezone aware.
    """

    def __init__(self):
        self.scheduler = AsyncIOScheduler(timezone=IST)
        self._running = False

    def setup(self) -> None:
        """Register all scheduled jobs."""

        # ── EOD data download (15:35 IST weekdays)
        self.scheduler.add_job(
            self._job_eod_download,
            CronTrigger(hour=15, minute=35, day_of_week="mon-fri", timezone=IST),
            id="eod_download",
            name="EOD Data Download",
            replace_existing=True,
            misfire_grace_time=600,
        )

        # ── Indicator computation (16:00 IST weekdays)
        self.scheduler.add_job(
            self._job_indicators,
            CronTrigger(hour=16, minute=0, day_of_week="mon-fri", timezone=IST),
            id="indicator_compute",
            name="Indicator Computation",
            replace_existing=True,
            misfire_grace_time=600,
        )

        # ── Swing scanner (16:15 IST weekdays)
        self.scheduler.add_job(
            self._job_scanner,
            CronTrigger(hour=16, minute=15, day_of_week="mon-fri", timezone=IST),
            id="swing_scanner",
            name="Swing Scanner",
            replace_existing=True,
            misfire_grace_time=600,
        )

        # ── Daily report (17:00 IST weekdays)
        self.scheduler.add_job(
            self._job_daily_report,
            CronTrigger(hour=17, minute=0, day_of_week="mon-fri", timezone=IST),
            id="daily_report",
            name="Daily Report",
            replace_existing=True,
            misfire_grace_time=1800,
        )

        # ── Intraday update every 5 minutes during market hours
        self.scheduler.add_job(
            self._job_intraday_update,
            CronTrigger(
                hour="9-15", minute="*/5",
                day_of_week="mon-fri",
                timezone=IST,
            ),
            id="intraday_update",
            name="Intraday Data Update",
            replace_existing=True,
        )

        # ── Phase 2: OI/PCR fetch every 5 minutes during market hours
        self.scheduler.add_job(
            self._job_oi_fetch,
            CronTrigger(
                hour="9-15", minute="*/5",
                day_of_week="mon-fri",
                timezone=IST,
            ),
            id="oi_fetch",
            name="OI / PCR Fetch",
            replace_existing=True,
        )

        logger.info(
            f"Scheduled {len(self.scheduler.get_jobs())} jobs: "
            + ", ".join(j.name for j in self.scheduler.get_jobs())
        )

    def start(self) -> None:
        if not self._running:
            self.scheduler.start()
            self._running = True
            logger.info("Scheduler started.")

    def stop(self) -> None:
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Scheduler stopped.")

    # ──────────────────────────────────────────────────────────
    # JOB IMPLEMENTATIONS
    # ──────────────────────────────────────────────────────────

    async def _job_eod_download(self) -> None:
        logger.info("⚙ JOB: EOD data download starting...")
        try:
            from nifty_quant_lab.data.downloader import NSEDataDownloader
            dl = NSEDataDownloader()
            await dl.update_today()
            logger.info("✓ JOB: EOD data download complete")
        except Exception as e:
            logger.error(f"✗ JOB: EOD download failed: {e}", exc_info=True)
            await self._send_error_alert("EOD Download", str(e))

    async def _job_indicators(self) -> None:
        logger.info("⚙ JOB: Indicator computation starting...")
        try:
            from nifty_quant_lab.indicators.service import IndicatorService
            svc = IndicatorService()
            await svc.compute_all_symbols()
            logger.info("✓ JOB: Indicator computation complete")
        except Exception as e:
            logger.error(f"✗ JOB: Indicator compute failed: {e}", exc_info=True)

    async def _job_scanner(self) -> None:
        logger.info("⚙ JOB: Swing scanner starting...")
        try:
            from nifty_quant_lab.signals.scanner import SwingScanner
            from nifty_quant_lab.telegram.alerts import TelegramAlerter

            scanner = SwingScanner()
            session = await scanner.scan_universe()

            # Persist results
            from nifty_quant_lab.signals.scanner_service import ScannerPersistenceService
            persist_svc = ScannerPersistenceService()
            await persist_svc.save_scan_session(session)

            # Send Telegram alert
            alerter = TelegramAlerter()
            if alerter.is_configured:
                msg = scanner.format_signal_summary(session)
                await alerter.send_message(msg)

            logger.info(f"✓ JOB: Scanner complete — {len(session.buy_signals)} BUY signals")
        except Exception as e:
            logger.error(f"✗ JOB: Scanner failed: {e}", exc_info=True)

    async def _job_daily_report(self) -> None:
        logger.info("⚙ JOB: Daily report generation starting...")
        try:
            from nifty_quant_lab.reports.generator import DailyReportGenerator
            gen = DailyReportGenerator()
            await gen.generate_and_send()
            logger.info("✓ JOB: Daily report complete")
        except Exception as e:
            logger.error(f"✗ JOB: Daily report failed: {e}", exc_info=True)

    async def _job_intraday_update(self) -> None:
        """Lightweight intraday candle update during market hours."""
        now_ist = datetime.now(IST)
        # Skip if outside RTH
        if not (9 <= now_ist.hour <= 15):
            return
        if now_ist.hour == 15 and now_ist.minute > 30:
            return
        try:
            from nifty_quant_lab.data.downloader import NSEDataDownloader
            dl = NSEDataDownloader()
            # Only update key indices intraday
            await dl.download_intraday(
                symbols=["NIFTY50", "BANKNIFTY", "FINNIFTY"],
                intervals=["5m"],
                days={"5m": 2},
            )
        except Exception as e:
            logger.debug(f"Intraday update error: {e}")  # Debug only — runs frequently

    async def _job_oi_fetch(self) -> None:
        """Fetch OI/PCR from NSE and persist to DB — runs every 5 min market hours."""
        from datetime import datetime
        import pytz
        now_ist = datetime.now(pytz.timezone("Asia/Kolkata"))
        if not (9 <= now_ist.hour <= 15):
            return
        if now_ist.hour == 15 and now_ist.minute > 30:
            return
        try:
            from nifty_quant_lab.data.providers.nse_scraper import NSEOptionChainScraper
            from nifty_quant_lab.analytics.oi_analytics import OIAnalyticsEngine
            from nifty_quant_lab.signals.oi_service import OIPersistenceService

            scraper = NSEOptionChainScraper()
            engine = OIAnalyticsEngine()
            svc = OIPersistenceService()

            for sym in ["NIFTY", "BANKNIFTY"]:
                result = await scraper.fetch_option_chain(sym)
                if result.success and result.data:
                    analysis = engine.analyze(result.data)
                    await svc.save_option_chain(result.data, analysis)
                    await svc.save_pcr(result.data)
                    logger.debug(f"OI saved: {sym} PCR={result.data.pcr_oi:.2f}")

            await scraper.close()
        except Exception as e:
            logger.debug(f"OI fetch error (non-critical): {e}")

    async def _send_error_alert(self, job_name: str, error: str) -> None:
        """Send error notification to Telegram."""
        try:
            from nifty_quant_lab.telegram.alerts import TelegramAlerter
            alerter = TelegramAlerter()
            if alerter.is_configured:
                msg = f"🚨 **NIFTY Quant Lab Error**\nJob: {job_name}\nError: {error[:200]}"
                await alerter.send_message(msg)
        except Exception:
            pass  # Don't fail on alert failure


# Singleton
scheduler = QuantLabScheduler()
