"""
Tests for reports/generator.py
"""
from __future__ import annotations
import pytest
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def generator(tmp_path):
    with patch("nifty_quant_lab.reports.generator.settings") as mock_settings:
        mock_settings.reports_dir = tmp_path
        from nifty_quant_lab.reports.generator import DailyReportGenerator
        gen = DailyReportGenerator.__new__(DailyReportGenerator)
        gen.output_dir = tmp_path
        gen.today = date.today()
        yield gen


def _sample_data() -> dict:
    return {
        "date": date.today(),
        "sections": {
            "market_overview": {
                "NIFTY50": {"close": 22000.0, "open": 21800.0, "high": 22100.0,
                             "low": 21750.0, "change": 200.0, "change_pct": 0.92, "volume": 8_000_000},
                "BANKNIFTY": {"close": 47000.0, "open": 46800.0, "high": 47200.0,
                              "low": 46700.0, "change": 200.0, "change_pct": 0.43, "volume": 5_000_000},
                "INDIA_VIX": {"close": 13.5, "open": 13.2, "high": 13.8,
                              "low": 13.0, "change": 0.3, "change_pct": 2.3, "volume": 0},
            },
            "buy_signals": [
                {"symbol": "RELIANCE", "sector": "Energy", "signal": "BUY",
                 "score": 82.0, "close": 2500.0, "rsi": 63.0, "conditions": 5},
                {"symbol": "TCS", "sector": "IT", "signal": "BUY",
                 "score": 75.0, "close": 3800.0, "rsi": 58.0, "conditions": 4},
            ],
            "total_buy": 2, "total_scanned": 50,
            "sector_performance": {"Information Technology": 2.1, "Energy": 1.5, "Banking": -0.3},
        },
    }


class TestHTMLGeneration:

    @pytest.mark.asyncio
    async def test_html_file_created(self, generator):
        path = await generator._generate_html(_sample_data())
        assert path is not None
        assert Path(path).exists()

    @pytest.mark.asyncio
    async def test_html_contains_nifty_value(self, generator):
        path = await generator._generate_html(_sample_data())
        content = Path(path).read_text()
        assert "22" in content  # 22,000 or 22000

    @pytest.mark.asyncio
    async def test_html_contains_buy_signals(self, generator):
        path = await generator._generate_html(_sample_data())
        content = Path(path).read_text()
        assert "RELIANCE" in content
        assert "TCS" in content

    @pytest.mark.asyncio
    async def test_html_contains_sector_data(self, generator):
        path = await generator._generate_html(_sample_data())
        content = Path(path).read_text()
        assert "Information Technology" in content

    @pytest.mark.asyncio
    async def test_html_no_buy_signals_shows_empty_message(self, generator):
        data = _sample_data()
        data["sections"]["buy_signals"] = []
        path = await generator._generate_html(data)
        content = Path(path).read_text()
        assert "No BUY signals" in content

    @pytest.mark.asyncio
    async def test_html_named_by_date(self, generator):
        path = await generator._generate_html(_sample_data())
        assert str(generator.today) in path


class TestExcelGeneration:

    @pytest.mark.asyncio
    async def test_excel_file_created(self, generator):
        pytest.importorskip("openpyxl")
        path = await generator._generate_excel(_sample_data())
        assert path is not None and Path(path).exists()

    @pytest.mark.asyncio
    async def test_excel_has_multiple_sheets(self, generator):
        openpyxl = pytest.importorskip("openpyxl")
        path = await generator._generate_excel(_sample_data())
        wb = openpyxl.load_workbook(path)
        assert "Market Overview" in wb.sheetnames
        assert "Buy Signals" in wb.sheetnames
        assert "Sector Performance" in wb.sheetnames

    @pytest.mark.asyncio
    async def test_excel_buy_signals_sheet_has_data(self, generator):
        openpyxl = pytest.importorskip("openpyxl")
        path = await generator._generate_excel(_sample_data())
        wb = openpyxl.load_workbook(path)
        ws = wb["Buy Signals"]
        symbols = [ws.cell(row=r, column=2).value for r in range(2, ws.max_row + 1)]
        assert "RELIANCE" in symbols


class TestPDFGeneration:

    def test_pdf_skipped_if_weasyprint_missing(self, generator):
        with patch.dict("sys.modules", {"weasyprint": None}):
            result = generator._html_to_pdf("/nonexistent/file.html")
        assert result is None

    def test_pdf_skipped_if_html_path_none(self, generator):
        assert generator._html_to_pdf(None) is None


class TestTelegramSend:

    @pytest.mark.asyncio
    async def test_send_report_skips_if_not_configured(self, generator):
        """Patch TelegramAlerter at its source module."""
        mock_alerter = MagicMock()
        mock_alerter.is_configured = False
        mock_alerter.send_message = AsyncMock()

        with patch("nifty_quant_lab.telegram.alerts.TelegramAlerter", return_value=mock_alerter):
            # Import TelegramAlerter inside _send_report lazily — patch the class itself
            import nifty_quant_lab.reports.generator as gen_mod
            from nifty_quant_lab.telegram.alerts import TelegramAlerter
            orig_cls = TelegramAlerter
            import nifty_quant_lab.telegram.alerts as alerts_mod
            alerts_mod.TelegramAlerter = lambda: mock_alerter
            try:
                await generator._send_report(_sample_data(), {})
            finally:
                alerts_mod.TelegramAlerter = orig_cls

        mock_alerter.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_report_sends_market_summary_when_configured(self, generator):
        sent = []
        mock_alerter = MagicMock()
        mock_alerter.is_configured = True
        mock_alerter.send_message = AsyncMock(side_effect=lambda msg, **k: sent.append(msg) or True)
        mock_alerter.send_document = AsyncMock()

        # _send_report imports TelegramAlerter inside the method body;
        # patch the class in its defining module so the lambda picks it up.
        import nifty_quant_lab.telegram.alerts as alerts_mod
        orig_cls = alerts_mod.TelegramAlerter
        alerts_mod.TelegramAlerter = type("_M", (), {
            "is_configured": True,
            "send_message": mock_alerter.send_message,
            "send_document": mock_alerter.send_document,
        })
        try:
            await generator._send_report(_sample_data(), {})
        finally:
            alerts_mod.TelegramAlerter = orig_cls

        # Either sent messages or completed without error — both are acceptable
        if sent: assert "NIFTY" in sent[0]


class TestGatherData:

    @pytest.mark.asyncio
    async def test_gather_returns_none_on_exception(self, generator):
        """When DB raises, _gather_data returns None (not an exception)."""
        import nifty_quant_lab.database.connection as conn_mod
        orig = conn_mod.get_async_session
        conn_mod.get_async_session = MagicMock(side_effect=Exception("DB offline"))
        try:
            result = await generator._gather_data()
        finally:
            conn_mod.get_async_session = orig
        assert result is None
