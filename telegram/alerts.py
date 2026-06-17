"""
NIFTY Quant Lab - Telegram Alert System
==========================================
Sends market alerts, scan results, and daily reports via Telegram Bot API.

Commands handled:
  /scan     — Run swing scanner on demand
  /nifty    — NIFTY50 quote + key levels
  /banknifty — BANKNIFTY quote + key levels
  /portfolio — Portfolio summary
  /report   — Latest daily report

Alert types:
  Scanner alerts, Breakout alerts, OI alerts, AI signals, Portfolio alerts
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from nifty_quant_lab.config.settings import settings
from nifty_quant_lab.utils.logger import get_logger

logger = get_logger("telegram")

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
MAX_MSG_LENGTH = 4096  # Telegram limit


class TelegramAlerter:
    """
    Async Telegram Bot API client.

    Handles:
    - Message sending with Markdown formatting
    - Long message splitting
    - Retry logic for rate limits
    - Document/photo sending for reports
    """

    def __init__(self):
        self.token = settings.telegram.bot_token
        self.chat_id = settings.telegram.chat_id
        self.alert_chat_id = settings.telegram.alert_chat_id or self.chat_id
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def _api_url(self, method: str) -> str:
        return TELEGRAM_API.format(token=self.token, method=method)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def send_message(
        self,
        text: str,
        chat_id: Optional[str] = None,
        parse_mode: str = "Markdown",
        disable_web_preview: bool = True,
    ) -> bool:
        """
        Send a Telegram message. Automatically splits long messages.
        """
        if not self.is_configured:
            logger.debug("Telegram not configured — skipping message")
            return False

        target = chat_id or self.chat_id
        chunks = self._split_message(text)

        client = await self._get_client()
        success = True

        for chunk in chunks:
            payload = {
                "chat_id": target,
                "text": chunk,
                "parse_mode": parse_mode,
                "disable_web_page_preview": disable_web_preview,
            }
            try:
                resp = await client.post(self._api_url("sendMessage"), json=payload)
                resp.raise_for_status()
                await asyncio.sleep(0.3)  # Telegram rate limit buffer
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.json().get("parameters", {}).get("retry_after", 5))
                    logger.warning(f"Telegram rate limit — waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    # Retry once
                    try:
                        await client.post(self._api_url("sendMessage"), json=payload)
                    except Exception:
                        success = False
                else:
                    logger.error(f"Telegram send error: {e}")
                    success = False
            except Exception as e:
                logger.error(f"Telegram send failed: {e}")
                success = False

        return success

    async def send_document(
        self,
        file_path: str,
        caption: str = "",
        chat_id: Optional[str] = None,
    ) -> bool:
        """Send a file (PDF report, Excel, etc.)."""
        if not self.is_configured:
            return False

        target = chat_id or self.chat_id
        client = await self._get_client()
        try:
            with open(file_path, "rb") as f:
                files = {"document": f}
                data = {
                    "chat_id": target,
                    "caption": caption[:1000],
                    "parse_mode": "Markdown",
                }
                resp = await client.post(
                    self._api_url("sendDocument"), data=data, files=files
                )
                resp.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Telegram document send failed: {e}")
            return False

    async def send_photo(
        self,
        file_path: str,
        caption: str = "",
        chat_id: Optional[str] = None,
    ) -> bool:
        """Send a chart image."""
        if not self.is_configured:
            return False

        target = chat_id or self.chat_id
        client = await self._get_client()
        try:
            with open(file_path, "rb") as f:
                files = {"photo": f}
                data = {
                    "chat_id": target,
                    "caption": caption[:1000],
                    "parse_mode": "Markdown",
                }
                resp = await client.post(
                    self._api_url("sendPhoto"), data=data, files=files
                )
                resp.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Telegram photo send failed: {e}")
            return False

    @staticmethod
    def _split_message(text: str, limit: int = MAX_MSG_LENGTH) -> List[str]:
        """Split long messages at line boundaries."""
        if len(text) <= limit:
            return [text]
        chunks = []
        while text:
            if len(text) <= limit:
                chunks.append(text)
                break
            split_at = text.rfind("\n", 0, limit)
            if split_at == -1:
                split_at = limit
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")
        return chunks

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ──────────────────────────────────────────────────────────
    # PRE-FORMATTED ALERT TEMPLATES
    # ──────────────────────────────────────────────────────────

    async def send_breakout_alert(
        self,
        symbol: str,
        price: float,
        breakout_type: str,
        level: float,
        sl: float,
        target: float,
    ) -> None:
        """Send a breakout/breakdown alert."""
        emoji = "🚀" if "break" in breakout_type.lower() else "📉"
        msg = (
            f"{emoji} **Breakout Alert — {symbol}**\n\n"
            f"Type: {breakout_type}\n"
            f"Price: ₹{price:,.2f}\n"
            f"Level: ₹{level:,.2f}\n"
            f"Stop Loss: ₹{sl:,.2f}\n"
            f"Target: ₹{target:,.2f}\n"
            f"R:R = {(target - price) / (price - sl):.1f}x\n"
            f"🕐 {datetime.now().strftime('%H:%M IST')}"
        )
        await self.send_message(msg, chat_id=self.alert_chat_id)

    async def send_oi_alert(
        self,
        symbol: str,
        oi_signal: str,
        price_change_pct: float,
        oi_change_pct: float,
    ) -> None:
        """Send OI activity alert."""
        emoji_map = {
            "LONG_BUILDUP": "💪 Long Build-Up",
            "SHORT_BUILDUP": "🐻 Short Build-Up",
            "SHORT_COVERING": "🔄 Short Covering",
            "LONG_UNWINDING": "⚠️ Long Unwinding",
        }
        label = emoji_map.get(oi_signal, oi_signal)
        msg = (
            f"📊 **OI Alert — {symbol}**\n\n"
            f"Signal: {label}\n"
            f"Price Change: {price_change_pct:+.2f}%\n"
            f"OI Change: {oi_change_pct:+.2f}%\n"
            f"🕐 {datetime.now().strftime('%H:%M IST')}"
        )
        await self.send_message(msg, chat_id=self.alert_chat_id)

    async def send_ai_signal(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        reasoning: str,
        entry: float,
        sl: float,
        target: float,
    ) -> None:
        """Send AI trade signal."""
        emoji = "🤖🟢" if signal == "BUY" else "🤖🔴" if signal == "SELL" else "🤖🟡"
        msg = (
            f"{emoji} **AI Signal — {symbol}**\n\n"
            f"Signal: **{signal}**\n"
            f"Confidence: {confidence:.0f}%\n"
            f"Entry: ₹{entry:,.2f}\n"
            f"Stop Loss: ₹{sl:,.2f}\n"
            f"Target: ₹{target:,.2f}\n\n"
            f"📝 Reasoning:\n{reasoning[:300]}\n\n"
            f"🕐 {datetime.now().strftime('%H:%M IST')}"
        )
        await self.send_message(msg, chat_id=self.alert_chat_id)


class TelegramCommandHandler:
    """
    Handles incoming Telegram bot commands via webhook / polling.
    Commands: /scan /nifty /banknifty /portfolio /report
    """

    def __init__(self, alerter: Optional[TelegramAlerter] = None):
        self.alerter = alerter or TelegramAlerter()

    async def handle_command(self, message: Dict[str, Any]) -> None:
        """Parse and dispatch incoming Telegram command."""
        try:
            text = message.get("text", "")
            chat_id = str(message["chat"]["id"])
            command = text.split()[0].lower().replace("@", "").split("@")[0] if text else ""

            dispatch = {
                "/scan": self._cmd_scan,
                "/nifty": self._cmd_nifty,
                "/banknifty": self._cmd_banknifty,
                "/portfolio": self._cmd_portfolio,
                "/report": self._cmd_report,
                "/help": self._cmd_help,
            }

            handler = dispatch.get(command)
            if handler:
                await handler(chat_id, message)
            else:
                await self.alerter.send_message(
                    "Unknown command. Use /help for available commands.",
                    chat_id=chat_id,
                )
        except Exception as e:
            logger.error(f"Command handler error: {e}")

    async def _cmd_help(self, chat_id: str, _: Dict) -> None:
        msg = (
            "🤖 **NIFTY Quant Lab Bot**\n\n"
            "/scan — Run swing scanner\n"
            "/nifty — NIFTY50 analysis\n"
            "/banknifty — BANKNIFTY analysis\n"
            "/portfolio — Portfolio summary\n"
            "/report — Latest daily report\n"
            "/help — This message"
        )
        await self.alerter.send_message(msg, chat_id=chat_id)

    async def _cmd_scan(self, chat_id: str, _: Dict) -> None:
        await self.alerter.send_message("🔄 Running scanner...", chat_id=chat_id)
        try:
            from nifty_quant_lab.signals.scanner import SwingScanner
            scanner = SwingScanner()
            session = await scanner.scan_universe()
            msg = scanner.format_signal_summary(session)
            await self.alerter.send_message(msg, chat_id=chat_id)
        except Exception as e:
            await self.alerter.send_message(f"❌ Scanner error: {e}", chat_id=chat_id)

    async def _cmd_nifty(self, chat_id: str, _: Dict) -> None:
        await self._send_index_quote("NIFTY50", "NIFTY 50", chat_id)

    async def _cmd_banknifty(self, chat_id: str, _: Dict) -> None:
        await self._send_index_quote("BANKNIFTY", "BANK NIFTY", chat_id)

    async def _send_index_quote(self, symbol: str, label: str, chat_id: str) -> None:
        try:
            from nifty_quant_lab.data.providers.yfinance_provider import YFinanceProvider
            from datetime import date, timedelta

            provider = YFinanceProvider()
            end = date.today()
            start = end - timedelta(days=30)
            result = await provider.fetch_historical(symbol, start, end)

            if not result.success or result.data is None or result.data.empty:
                await self.alerter.send_message(f"❌ Could not fetch {label} data", chat_id=chat_id)
                return

            latest = result.data.iloc[-1]
            prev = result.data.iloc[-2]
            change = float(latest["close"]) - float(prev["close"])
            change_pct = change / float(prev["close"]) * 100
            emoji = "🟢" if change >= 0 else "🔴"

            msg = (
                f"{emoji} **{label}**\n\n"
                f"Close: ₹{latest['close']:,.2f}\n"
                f"Change: {change:+,.2f} ({change_pct:+.2f}%)\n"
                f"High: ₹{latest['high']:,.2f}\n"
                f"Low: ₹{latest['low']:,.2f}\n"
                f"Volume: {int(latest['volume']):,}\n"
                f"🕐 {end.strftime('%d %b %Y')}"
            )
            await self.alerter.send_message(msg, chat_id=chat_id)
        except Exception as e:
            await self.alerter.send_message(f"❌ Error: {e}", chat_id=chat_id)

    async def _cmd_portfolio(self, chat_id: str, _: Dict) -> None:
        await self.alerter.send_message(
            "📊 Portfolio feature coming in Phase 4. Stay tuned!",
            chat_id=chat_id,
        )

    async def _cmd_report(self, chat_id: str, _: Dict) -> None:
        await self.alerter.send_message(
            "📄 Report generation coming in Phase 3. Stay tuned!",
            chat_id=chat_id,
        )
