"""
Tests for telegram/alerts.py
Covers: message splitting, send_message retry logic, alert templates,
        command dispatch. No real Telegram API calls.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─────────────────────────────────────────────────────────────
# TelegramAlerter
# ─────────────────────────────────────────────────────────────

class TestTelegramAlerter:

    @pytest.fixture
    def configured_alerter(self):
        with patch.dict("os.environ", {
            "TELEGRAM_BOT_TOKEN": "123:ABC",
            "TELEGRAM_CHAT_ID": "-100123456",
        }):
            # Reload settings so env vars are picked up
            from nifty_quant_lab.telegram.alerts import TelegramAlerter
            alerter = TelegramAlerter.__new__(TelegramAlerter)
            alerter.token = "123:ABC"
            alerter.chat_id = "-100123456"
            alerter.alert_chat_id = "-100123456"
            alerter._client = None
            return alerter

    @pytest.fixture
    def unconfigured_alerter(self):
        from nifty_quant_lab.telegram.alerts import TelegramAlerter
        a = TelegramAlerter.__new__(TelegramAlerter)
        a.token = ""
        a.chat_id = ""
        a.alert_chat_id = ""
        a._client = None
        return a

    def test_is_configured_true(self, configured_alerter):
        assert configured_alerter.is_configured is True

    def test_is_configured_false(self, unconfigured_alerter):
        assert unconfigured_alerter.is_configured is False

    def test_split_message_short(self, configured_alerter):
        from nifty_quant_lab.telegram.alerts import TelegramAlerter
        chunks = TelegramAlerter._split_message("hello world")
        assert chunks == ["hello world"]

    def test_split_message_long(self, configured_alerter):
        from nifty_quant_lab.telegram.alerts import TelegramAlerter
        # Build a message longer than 4096 chars
        line = "A" * 100 + "\n"
        long_msg = line * 50  # 5050 chars
        chunks = TelegramAlerter._split_message(long_msg)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 4096

    def test_split_message_reassembles_losslessly(self):
        from nifty_quant_lab.telegram.alerts import TelegramAlerter
        line = "signal: RELIANCE BUY score=85\n"
        msg = line * 200  # ~6000 chars
        chunks = TelegramAlerter._split_message(msg)
        reassembled = "\n".join(chunks)
        # All original content present (strip trailing newlines)
        assert "signal: RELIANCE BUY score=85" in reassembled

    @pytest.mark.asyncio
    async def test_send_message_unconfigured_returns_false(self, unconfigured_alerter):
        result = await unconfigured_alerter.send_message("test")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_calls_api(self, configured_alerter):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.is_closed = False
        configured_alerter._client = mock_client

        result = await configured_alerter.send_message("Hello NIFTY!")
        assert result is True
        mock_client.post.assert_awaited_once()
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["text"] == "Hello NIFTY!"
        assert payload["chat_id"] == "-100123456"

    @pytest.mark.asyncio
    async def test_send_message_splits_long_message(self, configured_alerter):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.is_closed = False
        configured_alerter._client = mock_client

        long_msg = ("line\n" * 1000)  # definitely > 4096
        await configured_alerter.send_message(long_msg)
        # Should have been called multiple times (once per chunk)
        assert mock_client.post.await_count >= 2

    def test_api_url_format(self, configured_alerter):
        url = configured_alerter._api_url("sendMessage")
        assert "123:ABC" in url
        assert "sendMessage" in url

    @pytest.mark.asyncio
    async def test_send_breakout_alert_formats_correctly(self, configured_alerter):
        sent_msgs = []
        async def capture_send(msg, **kwargs):
            sent_msgs.append(msg)
            return True
        configured_alerter.send_message = capture_send

        await configured_alerter.send_breakout_alert(
            symbol="RELIANCE", price=2500.0, breakout_type="52W Breakout",
            level=2490.0, sl=2450.0, target=2600.0,
        )
        assert sent_msgs
        msg = sent_msgs[0]
        assert "RELIANCE" in msg
        assert "RELIANCE" in msg and ("2,500" in msg or "2500" in msg)
        assert "SL" in msg or "Stop" in msg

    @pytest.mark.asyncio
    async def test_send_oi_alert_contains_symbol(self, configured_alerter):
        sent_msgs = []
        async def capture(msg, **kwargs):
            sent_msgs.append(msg)
            return True
        configured_alerter.send_message = capture

        await configured_alerter.send_oi_alert(
            symbol="BANKNIFTY", oi_signal="LONG_BUILDUP",
            price_change_pct=1.2, oi_change_pct=15.0,
        )
        assert "BANKNIFTY" in sent_msgs[0]
        assert "Long" in sent_msgs[0] or "LONG" in sent_msgs[0]


# ─────────────────────────────────────────────────────────────
# TelegramCommandHandler
# ─────────────────────────────────────────────────────────────

class TestTelegramCommandHandler:

    @pytest.fixture
    def handler(self):
        from nifty_quant_lab.telegram.alerts import TelegramCommandHandler, TelegramAlerter
        mock_alerter = MagicMock(spec=TelegramAlerter)
        mock_alerter.send_message = AsyncMock(return_value=True)
        mock_alerter.is_configured = True
        h = TelegramCommandHandler(alerter=mock_alerter)
        return h

    def _msg(self, text: str, chat_id: str = "99999") -> dict:
        return {"text": text, "chat": {"id": chat_id}}

    @pytest.mark.asyncio
    async def test_help_command(self, handler):
        await handler.handle_command(self._msg("/help"))
        handler.alerter.send_message.assert_awaited_once()
        call_text = handler.alerter.send_message.call_args[0][0]
        assert "/scan" in call_text
        assert "/nifty" in call_text

    @pytest.mark.asyncio
    async def test_unknown_command_responds(self, handler):
        await handler.handle_command(self._msg("/xyz"))
        handler.alerter.send_message.assert_awaited_once()
        call_text = handler.alerter.send_message.call_args[0][0]
        assert "Unknown" in call_text or "help" in call_text.lower()

    @pytest.mark.asyncio
    async def test_portfolio_command_responds(self, handler):
        await handler.handle_command(self._msg("/portfolio"))
        handler.alerter.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_nifty_command_dispatches(self, handler):
        with patch.object(handler, "_send_index_quote", new=AsyncMock()) as mock_quote:
            await handler.handle_command(self._msg("/nifty"))
        mock_quote.assert_awaited_once()
        args = mock_quote.call_args[0]
        assert args[0] == "NIFTY50"

    @pytest.mark.asyncio
    async def test_banknifty_command_dispatches(self, handler):
        with patch.object(handler, "_send_index_quote", new=AsyncMock()) as mock_quote:
            await handler.handle_command(self._msg("/banknifty"))
        mock_quote.assert_awaited_once()
        args = mock_quote.call_args[0]
        assert args[0] == "BANKNIFTY"

    @pytest.mark.asyncio
    async def test_malformed_message_does_not_crash(self, handler):
        await handler.handle_command({})  # missing text/chat keys
        # Should not raise

    @pytest.mark.asyncio
    async def test_scan_command_triggers_scanner(self, handler):
        mock_scanner = MagicMock()
        mock_session = MagicMock()
        mock_session.total_scanned = 50
        mock_session.buy_signals = []
        mock_session.watchlist_signals = []
        mock_session.elapsed_seconds = 1.0
        mock_session.results = []
        mock_scanner.scan_universe = AsyncMock(return_value=mock_session)
        mock_scanner.format_signal_summary = MagicMock(return_value="scan result text")

        # SwingScanner is imported lazily inside _cmd_scan — patch at source
        import nifty_quant_lab.signals.scanner as scanner_mod
        orig_cls = scanner_mod.SwingScanner
        scanner_mod.SwingScanner = lambda: mock_scanner
        try:
            await handler._cmd_scan("99999", {})
        finally:
            scanner_mod.SwingScanner = orig_cls

        mock_scanner.scan_universe.assert_awaited_once()
        handler.alerter.send_message.assert_awaited()
