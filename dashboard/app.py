"""
NIFTY Quant Lab - Streamlit Dashboard (Phase 1)
=================================================
Interactive market analytics dashboard.

Pages:
  🏠 Home          — Market overview, NIFTY/BANKNIFTY charts
  📊 Scanner       — Swing scan results with filtering
  📈 Indicators    — Per-symbol indicator deep-dive
  🎯 S/R Levels    — Support & Resistance analysis
  ⚙️  Admin        — Manual triggers, data status

Run: streamlit run dashboard/app.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NIFTY Quant Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS
st.markdown("""
<style>
  .metric-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 16px 20px; }
  .green { color: #3fb950; }
  .red { color: #f85149; }
  .badge { padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 700; }
  .badge-buy { background: #1a4731; color: #3fb950; }
  .badge-sell { background: #4d1919; color: #f85149; }
  .badge-watch { background: #1a2e4a; color: #58a6ff; }
  section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ASYNC HELPER
# ─────────────────────────────────────────────────────────────

def run_async(coro):
    """Run async code from Streamlit's sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────
# DATA LOADERS (cached)
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_ohlcv(symbol: str, days: int = 365) -> pd.DataFrame:
    """Load OHLCV from yfinance (no DB dependency for dashboard)."""
    from nifty_quant_lab.data.providers.yfinance_provider import YFinanceProvider
    provider = YFinanceProvider()
    end = date.today()
    start = end - timedelta(days=days)
    result = run_async(provider.fetch_historical(symbol, start, end))
    if result.success and result.data is not None:
        return result.data
    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_scan_results() -> pd.DataFrame:
    """Load latest scan results from DB."""
    try:
        from nifty_quant_lab.signals.scanner_service import ScannerPersistenceService
        svc = ScannerPersistenceService()
        rows = run_async(svc.get_latest_results(limit=100))
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def compute_indicators(symbol: str, days: int = 365) -> pd.DataFrame:
    """Compute indicators live from yfinance data."""
    from nifty_quant_lab.indicators.engine import IndicatorEngine
    df = load_ohlcv(symbol, days)
    if df.empty:
        return pd.DataFrame()
    engine = IndicatorEngine()
    result = engine.compute(df, symbol)
    return result.df if result.success else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────

def candlestick_chart(df: pd.DataFrame, symbol: str, indicators: Optional[pd.DataFrame] = None):
    """Full candlestick chart with EMA overlays and volume."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[f"{symbol} — Price", "Volume", "RSI (14)"],
    )

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name=symbol,
        increasing_line_color="#3fb950",
        decreasing_line_color="#f85149",
    ), row=1, col=1)

    # EMA overlays
    if indicators is not None and not indicators.empty:
        for ema_col, color, name in [
            ("ema_20", "#f0883e", "EMA 20"),
            ("ema_50", "#58a6ff", "EMA 50"),
            ("ema_200", "#bc8cff", "EMA 200"),
        ]:
            if ema_col in indicators.columns:
                fig.add_trace(go.Scatter(
                    x=indicators.index, y=indicators[ema_col],
                    mode="lines", name=name,
                    line=dict(color=color, width=1.5),
                ), row=1, col=1)

        # Supertrend
        if "supertrend" in indicators.columns and "supertrend_direction" in indicators.columns:
            up = indicators[indicators["supertrend_direction"] == 1]
            dn = indicators[indicators["supertrend_direction"] == -1]
            if not up.empty:
                fig.add_trace(go.Scatter(
                    x=up.index, y=up["supertrend"], mode="lines",
                    name="Supertrend ▲", line=dict(color="#3fb950", width=1, dash="dot"),
                ), row=1, col=1)
            if not dn.empty:
                fig.add_trace(go.Scatter(
                    x=dn.index, y=dn["supertrend"], mode="lines",
                    name="Supertrend ▼", line=dict(color="#f85149", width=1, dash="dot"),
                ), row=1, col=1)

    # Volume
    vol_colors = [
        "#3fb950" if c >= o else "#f85149"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"],
        marker_color=vol_colors, name="Volume",
        showlegend=False,
    ), row=2, col=1)

    # RSI
    if indicators is not None and "rsi_14" in indicators.columns:
        fig.add_trace(go.Scatter(
            x=indicators.index, y=indicators["rsi_14"],
            mode="lines", name="RSI 14",
            line=dict(color="#ffa657", width=1.5),
        ), row=3, col=1)
        # OB/OS lines
        for level, color in [(70, "#f85149"), (30, "#3fb950")]:
            fig.add_hline(y=level, line_dash="dot", line_color=color,
                          opacity=0.6, row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=700,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#21262d")
    fig.update_yaxes(showgrid=True, gridcolor="#21262d")
    return fig


def macd_chart(indicators: pd.DataFrame) -> go.Figure:
    """MACD histogram + lines chart."""
    fig = go.Figure()
    if indicators.empty:
        return fig
    colors = ["#3fb950" if v >= 0 else "#f85149"
              for v in indicators.get("macd_histogram", [])]
    fig.add_trace(go.Bar(
        x=indicators.index, y=indicators.get("macd_histogram"),
        marker_color=colors, name="Histogram",
    ))
    fig.add_trace(go.Scatter(
        x=indicators.index, y=indicators.get("macd_line"),
        mode="lines", name="MACD", line=dict(color="#58a6ff", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=indicators.index, y=indicators.get("macd_signal"),
        mode="lines", name="Signal", line=dict(color="#f0883e", width=1.5),
    ))
    fig.add_hline(y=0, line_color="#8b949e", line_dash="dash", opacity=0.5)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=250, margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", y=1.02),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 NIFTY Quant Lab")
    st.markdown("*IntelliSapphire Trading Analytics*")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Market Overview", "📊 Scanner", "📈 Symbol Analysis", "🎯 S/R Levels", "📉 OI Dashboard", "⚙️ Admin"],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown("**Quick Stats**")
    today_str = date.today().strftime("%d %b %Y")
    st.caption(f"📅 {today_str}")
    st.caption("🔄 Auto-refresh: 5 min")


# ─────────────────────────────────────────────────────────────
# PAGE: MARKET OVERVIEW
# ─────────────────────────────────────────────────────────────

if page == "🏠 Market Overview":
    st.title("🏠 Market Overview")

    col1, col2, col3 = st.columns(3)

    for sym, label, col in [
        ("NIFTY50", "NIFTY 50", col1),
        ("BANKNIFTY", "BANK NIFTY", col2),
        ("INDIA_VIX", "India VIX", col3),
    ]:
        with col:
            df = load_ohlcv(sym, days=5)
            if not df.empty and len(df) >= 2:
                close = float(df["close"].iloc[-1])
                prev = float(df["close"].iloc[-2])
                chg = close - prev
                chg_pct = chg / prev * 100
                st.metric(
                    label=label,
                    value=f"₹{close:,.2f}",
                    delta=f"{chg:+,.2f} ({chg_pct:+.2f}%)",
                )
            else:
                st.metric(label=label, value="N/A")

    st.divider()

    # Main NIFTY chart
    sym_sel = st.selectbox("Select Index", ["NIFTY50", "BANKNIFTY", "FINNIFTY"], index=0)
    period = st.select_slider("Period", options=[30, 60, 90, 180, 365], value=180)

    with st.spinner(f"Loading {sym_sel}..."):
        df_main = load_ohlcv(sym_sel, days=period)
        ind_main = compute_indicators(sym_sel, days=period)

    if not df_main.empty:
        fig = candlestick_chart(df_main, sym_sel, ind_main)
        st.plotly_chart(fig, use_container_width=True)

        if not ind_main.empty:
            st.subheader("MACD")
            st.plotly_chart(macd_chart(ind_main), use_container_width=True)
    else:
        st.warning(f"No data for {sym_sel}. Check your internet connection.")


# ─────────────────────────────────────────────────────────────
# PAGE: SCANNER
# ─────────────────────────────────────────────────────────────

elif page == "📊 Scanner":
    st.title("📊 Swing Scanner")
    st.caption("NSE swing trade signals — 6-condition scoring system")

    col_r, col_b = st.columns([3, 1])
    with col_b:
        if st.button("🔄 Run Scanner Now", type="primary", use_container_width=True):
            with st.spinner("Scanning NIFTY50 universe..."):
                from nifty_quant_lab.signals.scanner import SwingScanner
                scanner = SwingScanner()
                session = run_async(scanner.scan_universe())
                st.session_state["scan_session"] = session
                st.cache_data.clear()
            st.success(f"✓ Scanned {session.total_scanned} symbols in {session.elapsed_seconds:.1f}s")

    scan_df = load_scan_results()

    if not scan_df.empty:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            sig_filter = st.multiselect(
                "Signal", ["BUY", "SELL", "WATCHLIST", "HOLD"],
                default=["BUY", "WATCHLIST"],
            )
        with col_f2:
            min_score = st.slider("Min Score", 0, 100, 50)
        with col_f3:
            min_cond = st.slider("Min Conditions", 1, 6, 3)

        filtered = scan_df.copy()
        if sig_filter:
            filtered = filtered[filtered["signal"].isin(sig_filter)]
        if min_score > 0:
            filtered = filtered[filtered["score"] >= min_score]
        if "conditions_met" in filtered.columns:
            filtered = filtered[filtered["conditions_met"] >= min_cond]

        st.markdown(f"**{len(filtered)} signals** match filters")

        if not filtered.empty:
            display_cols = ["symbol", "signal", "score", "close_price", "conditions_met",
                            "rsi", "sector", "notes"]
            display = filtered[[c for c in display_cols if c in filtered.columns]]

            def _color_signal(val):
                colors = {"BUY": "background-color:#1a4731;color:#3fb950",
                          "SELL": "background-color:#4d1919;color:#f85149",
                          "WATCHLIST": "background-color:#1a2e4a;color:#58a6ff"}
                return colors.get(val, "")

            st.dataframe(
                display.style.applymap(_color_signal, subset=["signal"]),
                use_container_width=True,
                height=500,
            )
    else:
        st.info("No scan results in DB. Click **Run Scanner Now** to scan the universe.")

        # Show live session if in state
        if "scan_session" in st.session_state:
            session = st.session_state["scan_session"]
            df_live = session.to_dataframe()
            if not df_live.empty:
                st.dataframe(df_live, use_container_width=True, height=400)


# ─────────────────────────────────────────────────────────────
# PAGE: SYMBOL ANALYSIS
# ─────────────────────────────────────────────────────────────

elif page == "📈 Symbol Analysis":
    st.title("📈 Symbol Analysis")

    from nifty_quant_lab.config.settings import NIFTY50_SYMBOLS

    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.selectbox(
            "Select Symbol",
            ["NIFTY50", "BANKNIFTY"] + list(NIFTY50_SYMBOLS),
            index=0,
        )
    with col2:
        period = st.select_slider("Period", [60, 90, 180, 365, 730], value=365)

    with st.spinner(f"Loading {symbol}..."):
        df = load_ohlcv(symbol, days=period)
        ind = compute_indicators(symbol, days=period)

    if df.empty:
        st.error(f"No data for {symbol}")
    else:
        # Key metrics
        latest = df.iloc[-1]
        prev_r = df.iloc[-2]
        close = float(latest["close"])
        chg = close - float(prev_r["close"])
        chg_pct = chg / float(prev_r["close"]) * 100

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Close", f"₹{close:,.2f}", f"{chg:+.2f}")
        m2.metric("High", f"₹{float(latest['high']):,.2f}")
        m3.metric("Low", f"₹{float(latest['low']):,.2f}")
        m4.metric("Volume", f"{int(latest['volume']):,}")

        if not ind.empty:
            latest_ind = ind.iloc[-1]
            rsi = latest_ind.get("rsi_14", None)
            m5.metric("RSI 14", f"{rsi:.1f}" if rsi else "N/A")

        # Price chart
        st.plotly_chart(candlestick_chart(df, symbol, ind), use_container_width=True)

        if not ind.empty:
            # MACD + Bollinger
            tab1, tab2, tab3 = st.tabs(["MACD", "Bollinger Bands", "Indicator Table"])

            with tab1:
                st.plotly_chart(macd_chart(ind), use_container_width=True)

            with tab2:
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(x=ind.index, y=ind["close"],
                                            mode="lines", name="Close",
                                            line=dict(color="#e6edf3", width=1.5)))
                if "bb_upper" in ind.columns:
                    fig_bb.add_trace(go.Scatter(x=ind.index, y=ind["bb_upper"],
                                                mode="lines", name="Upper BB",
                                                line=dict(color="#58a6ff", width=1, dash="dot")))
                    fig_bb.add_trace(go.Scatter(x=ind.index, y=ind["bb_middle"],
                                                mode="lines", name="Middle",
                                                line=dict(color="#8b949e", width=1)))
                    fig_bb.add_trace(go.Scatter(x=ind.index, y=ind["bb_lower"],
                                                mode="lines", name="Lower BB",
                                                line=dict(color="#f85149", width=1, dash="dot"),
                                                fill="tonexty", fillcolor="rgba(88,166,255,0.05)"))
                fig_bb.update_layout(
                    template="plotly_dark", paper_bgcolor="#0d1117",
                    plot_bgcolor="#0d1117", height=300,
                    margin=dict(l=0, r=0, t=20, b=0),
                )
                st.plotly_chart(fig_bb, use_container_width=True)

            with tab3:
                display_cols = [
                    "close", "ema_20", "ema_50", "ema_200",
                    "rsi_14", "macd_histogram", "atr_14", "adx_14",
                    "supertrend_direction", "bb_pct_b",
                ]
                display = ind[[c for c in display_cols if c in ind.columns]].tail(30)
                st.dataframe(display.iloc[::-1], use_container_width=True, height=400)


# ─────────────────────────────────────────────────────────────
# PAGE: S/R LEVELS
# ─────────────────────────────────────────────────────────────

elif page == "🎯 S/R Levels":
    st.title("🎯 Support & Resistance Levels")

    from nifty_quant_lab.config.settings import NIFTY50_SYMBOLS
    symbol = st.selectbox(
        "Select Symbol",
        ["NIFTY50", "BANKNIFTY"] + list(NIFTY50_SYMBOLS),
    )
    lookback = st.slider("Lookback (trading days)", 60, 500, 252)

    with st.spinner("Computing S/R levels..."):
        df = load_ohlcv(symbol, days=lookback + 50)

    if df.empty:
        st.error(f"No data for {symbol}")
    else:
        from nifty_quant_lab.analytics.support_resistance import SupportResistanceEngine
        engine = SupportResistanceEngine()
        analysis = engine.analyze(df.tail(lookback), symbol=symbol)

        if not analysis.success:
            st.error(f"S/R analysis failed: {analysis.error}")
        else:
            current = analysis.current_price
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"₹{current:,.2f}")
            c2.metric(
                "Nearest Support",
                f"₹{analysis.nearest_support:,.2f}" if analysis.nearest_support else "N/A",
                delta=f"{(analysis.nearest_support - current) / current * 100:.2f}%" if analysis.nearest_support else None,
            )
            c3.metric(
                "Nearest Resistance",
                f"₹{analysis.nearest_resistance:,.2f}" if analysis.nearest_resistance else "N/A",
            )

            if analysis.risk_reward_estimate:
                st.info(f"📐 Estimated R:R to nearest levels: **{analysis.risk_reward_estimate:.2f}x**")

            # Price chart with S/R overlays
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index[-lookback:], y=df["close"].iloc[-lookback:],
                mode="lines", name="Close",
                line=dict(color="#e6edf3", width=1.5),
            ))

            color_map = {
                "STRONG_SUPPORT": ("#3fb950", "solid"),
                "WEAK_SUPPORT": ("#85e89d", "dot"),
                "RESISTANCE": ("#f85149", "solid"),
                "BREAKOUT": ("#ffa657", "dash"),
            }
            for lvl in analysis.levels[:20]:
                color, dash = color_map.get(lvl.level_type.value, ("#8b949e", "dot"))
                fig.add_hline(
                    y=lvl.price,
                    line_color=color,
                    line_dash=dash,
                    opacity=0.7,
                    annotation_text=f"{lvl.level_type.value.replace('_', ' ')} ₹{lvl.price:,.0f}",
                    annotation_font_color=color,
                    annotation_font_size=10,
                )

            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117", height=500,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Level tables
            tab1, tab2, tab3 = st.tabs(["💚 Supports", "🔴 Resistances", "All Levels"])
            with tab1:
                sup_data = [{"Price": l.price, "Type": l.level_type.value,
                             "Method": l.method, "Strength": l.strength, "Notes": l.notes}
                            for l in (analysis.strong_supports + analysis.weak_supports)]
                if sup_data:
                    st.dataframe(pd.DataFrame(sup_data).sort_values("Price", ascending=False),
                                 use_container_width=True)

            with tab2:
                res_data = [{"Price": l.price, "Method": l.method,
                             "Strength": l.strength, "Notes": l.notes}
                            for l in analysis.resistances]
                if res_data:
                    st.dataframe(pd.DataFrame(res_data).sort_values("Price"),
                                 use_container_width=True)

            with tab3:
                all_data = [{"Price": l.price, "Type": l.level_type.value,
                             "Method": l.method, "Strength": l.strength}
                            for l in analysis.levels]
                if all_data:
                    st.dataframe(pd.DataFrame(all_data).sort_values("Price", ascending=False),
                                 use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE: OI DASHBOARD (Phase 2)
# ─────────────────────────────────────────────────────────────

elif page == "📉 OI Dashboard":
    from nifty_quant_lab.dashboard.pages.oi_dashboard import render_oi_dashboard
    render_oi_dashboard()


# ─────────────────────────────────────────────────────────────
# PAGE: ADMIN
# ─────────────────────────────────────────────────────────────

elif page == "⚙️ Admin":
    st.title("⚙️ Admin Panel")
    st.caption("Manual triggers and data management")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Pipeline")

        if st.button("📥 Sync Symbol Registry", use_container_width=True):
            with st.spinner("Syncing..."):
                from nifty_quant_lab.data.downloader import NSEDataDownloader
                dl = NSEDataDownloader()
                run_async(dl.sync_symbol_registry())
            st.success("✓ Symbol registry synced")

        if st.button("📊 Download EOD Data (1Y)", use_container_width=True):
            with st.spinner("Downloading... this may take a few minutes"):
                from nifty_quant_lab.data.downloader import NSEDataDownloader
                dl = NSEDataDownloader()
                results = run_async(dl.download_historical(years=1))
                ok = sum(1 for v in results.values() if v)
            st.success(f"✓ Downloaded: {ok}/{len(results)} symbols")

        if st.button("⚡ Compute All Indicators", use_container_width=True):
            with st.spinner("Computing indicators..."):
                from nifty_quant_lab.indicators.service import IndicatorService
                svc = IndicatorService()
                results = run_async(svc.compute_all_symbols())
                ok = sum(1 for v in results.values() if v)
            st.success(f"✓ Indicators computed: {ok}/{len(results)} symbols")

    with col2:
        st.subheader("Scanner & Reports")

        if st.button("🔍 Run Full Scan", use_container_width=True):
            with st.spinner("Scanning NIFTY50 universe..."):
                from nifty_quant_lab.signals.scanner import SwingScanner
                from nifty_quant_lab.signals.scanner_service import ScannerPersistenceService
                scanner = SwingScanner()
                session = run_async(scanner.scan_universe())
                persist = ScannerPersistenceService()
                run_async(persist.save_scan_session(session))
            st.success(
                f"✓ Scan complete: {len(session.buy_signals)} BUY | "
                f"{len(session.watchlist_signals)} WATCHLIST"
            )
            st.cache_data.clear()

        if st.button("📄 Generate Daily Report", use_container_width=True):
            with st.spinner("Generating report..."):
                from nifty_quant_lab.reports.generator import DailyReportGenerator
                gen = DailyReportGenerator()
                paths = run_async(gen.generate_and_send())
            if paths.get("html"):
                st.success(f"✓ Report generated: {paths}")
            else:
                st.error("Report generation failed")

        if st.button("🧹 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("✓ Cache cleared")

    st.divider()
    st.subheader("System Status")
    col_a, col_b = st.columns(2)
    with col_a:
        try:
            import asyncio
            from nifty_quant_lab.database.connection import check_connection
            db_ok = run_async(check_connection())
            st.metric("Database", "✅ Connected" if db_ok else "❌ Disconnected")
        except Exception as e:
            st.metric("Database", f"❌ {str(e)[:40]}")
    with col_b:
        from nifty_quant_lab.config.scheduler import scheduler
        st.metric("Scheduler", "✅ Running" if scheduler._running else "⏸ Stopped")
