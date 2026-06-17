"""
NIFTY Quant Lab — OI Dashboard (Phase 2)
==========================================
Streamlit page: Option chain OI heatmap, PCR trend, max pain, IV skew.

Add to dashboard/app.py sidebar navigation as "📉 OI Dashboard".
Or run standalone:  streamlit run dashboard/pages/oi_dashboard.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from nifty_quant_lab.analytics.oi_analytics import OIAnalyticsEngine
from nifty_quant_lab.analytics.futures_analytics import FuturesAnalyticsEngine
from nifty_quant_lab.data.providers.nse_scraper import NSEOptionChainScraper
from nifty_quant_lab.signals.oi_service import OIPersistenceService


# ── Async helper
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── Cached fetch
@st.cache_data(ttl=60, show_spinner=False)
def fetch_chain(symbol: str, expiry: Optional[str] = None):
    scraper = NSEOptionChainScraper()
    result = run_async(scraper.fetch_option_chain(symbol, expiry))
    return result


@st.cache_data(ttl=300, show_spinner=False)
def load_pcr_history(symbol: str, hours: int = 6):
    svc = OIPersistenceService()
    return run_async(svc.get_pcr_history(symbol, hours=hours))


# ─────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────

def oi_heatmap(df: pd.DataFrame, spot: float, atm: float) -> go.Figure:
    """
    OI heatmap — CE OI (red bars) and PE OI (green bars) at each strike.
    Sorted with ATM in centre. Spot and max pain lines overlaid.
    """
    df = df.sort_values("strike")

    fig = go.Figure()

    # CE OI — above ATM focus
    fig.add_trace(go.Bar(
        x=df["strike"], y=df["ce_oi"],
        name="CE OI", marker_color="#f85149",
        opacity=0.8,
    ))
    # PE OI
    fig.add_trace(go.Bar(
        x=df["strike"], y=df["pe_oi"],
        name="PE OI", marker_color="#3fb950",
        opacity=0.8,
    ))

    # Spot line
    fig.add_vline(
        x=spot, line_dash="solid", line_color="#e6edf3",
        line_width=2, annotation_text=f"Spot {spot:,.0f}",
        annotation_font_color="#e6edf3",
    )

    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=420,
        title="Open Interest by Strike",
        xaxis_title="Strike",
        yaxis_title="OI (contracts)",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#21262d", tickangle=-45)
    fig.update_yaxes(showgrid=True, gridcolor="#21262d")
    return fig


def oi_change_chart(df: pd.DataFrame) -> go.Figure:
    """OI Change bars — shows where fresh positions are being built."""
    df = df.sort_values("strike")
    colors_ce = ["#f85149" if v >= 0 else "#85e89d" for v in df["ce_oi_change"]]
    colors_pe = ["#3fb950" if v >= 0 else "#ffa657" for v in df["pe_oi_change"]]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["CE OI Change", "PE OI Change"],
                        shared_yaxes=True)
    fig.add_trace(
        go.Bar(x=df["strike"], y=df["ce_oi_change"],
               marker_color=colors_ce, name="CE ΔOI"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=df["strike"], y=df["pe_oi_change"],
               marker_color=colors_pe, name="PE ΔOI"),
        row=1, col=2
    )
    fig.add_hline(y=0, line_color="#8b949e", line_dash="dash", opacity=0.5)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=300, showlegend=False, margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def pcr_trend_chart(history: list) -> go.Figure:
    """PCR intraday trend line."""
    if not history:
        return go.Figure()

    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["pcr_oi"],
        mode="lines+markers", name="PCR (OI)",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
    ))
    # Threshold lines
    for level, color, label in [
        (1.3, "#3fb950", "Bullish (1.3)"),
        (0.8, "#f85149", "Bearish (0.8)"),
    ]:
        fig.add_hline(
            y=level, line_dash="dot", line_color=color,
            annotation_text=label, annotation_font_color=color,
        )

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=250, title="PCR Intraday Trend",
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_title="Time (IST)", yaxis_title="PCR",
    )
    return fig


def iv_skew_chart(df: pd.DataFrame) -> go.Figure:
    """IV smile — CE IV and PE IV across strikes."""
    df = df[(df["ce_iv"] > 0) | (df["pe_iv"] > 0)].sort_values("strike")
    fig = go.Figure()
    if "ce_iv" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["strike"], y=df["ce_iv"],
            mode="lines+markers", name="CE IV",
            line=dict(color="#f85149", width=1.5),
        ))
    if "pe_iv" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["strike"], y=df["pe_iv"],
            mode="lines+markers", name="PE IV",
            line=dict(color="#3fb950", width=1.5),
        ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=280, title="IV Smile",
        xaxis_title="Strike", yaxis_title="Implied Volatility (%)",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.02),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────────

def render_oi_dashboard():
    """Main OI dashboard render function. Call from dashboard/app.py."""

    st.title("📉 OI Dashboard")
    st.caption("Live option chain OI heatmap, PCR trend, and max pain analysis")

    # ── Controls
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        symbol = st.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"], index=0)
    with col2:
        atm_range = st.slider("ATM ± Strikes", 5, 30, 15)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button("🔄 Refresh", use_container_width=True)

    if refresh:
        st.cache_data.clear()

    # ── Fetch data
    with st.spinner(f"Fetching {symbol} option chain from NSE..."):
        result = fetch_chain(symbol)

    if not result or not result.success:
        st.error(f"NSE data unavailable: {getattr(result, 'error', 'Unknown error')}")
        st.info("NSE may be closed or blocking requests. Try again during market hours.")
        return

    snapshot = result.data
    engine = OIAnalyticsEngine()
    analysis = engine.analyze(snapshot)

    # ── Metrics row
    pcr = analysis.pcr_analysis
    spot = snapshot.spot_price

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Spot", f"₹{spot:,.2f}")
    m2.metric("ATM Strike", f"{snapshot.atm_strike:,.0f}" if snapshot.atm_strike else "N/A")

    if pcr:
        pcr_delta_color = "normal" if pcr.signal.value == "NEUTRAL" else "inverse"
        m3.metric("PCR (OI)", f"{pcr.pcr_oi:.2f}", delta=pcr.signal.value)
        m4.metric("Max Pain", f"₹{pcr.max_pain:,.0f}" if pcr.max_pain else "N/A",
                  delta=f"{pcr.max_pain_gap_pct:+.1f}%" if pcr.max_pain_gap_pct else None)

    if analysis.iv_skew is not None:
        skew_label = "Put skew" if analysis.iv_skew > 0 else "Call skew"
        m5.metric("IV Skew", f"{analysis.iv_skew:+.1f}", delta=skew_label)

    # ── PCR interpretation banner
    if pcr:
        color = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(pcr.signal.value, "⚪")
        st.info(f"{color} **PCR Signal [{pcr.signal_strength}]:** {pcr.interpretation}")

    # ── OI walls summary
    if analysis.nearest_ce_wall or analysis.nearest_pe_wall:
        wc1, wc2 = st.columns(2)
        if analysis.nearest_ce_wall:
            w = analysis.nearest_ce_wall
            wc1.metric(
                f"🔴 CE Wall (Resistance)",
                f"{w.strike:,.0f}",
                delta=f"OI: {w.oi:,} [{w.strength}]",
            )
        if analysis.nearest_pe_wall:
            w = analysis.nearest_pe_wall
            wc2.metric(
                f"🟢 PE Wall (Support)",
                f"{w.strike:,.0f}",
                delta=f"OI: {w.oi:,} [{w.strength}]",
            )

    st.divider()

    # ── Build ATM-range DataFrame
    atm_rows = snapshot.atm_rows(atm_range) if snapshot.atm_strike else snapshot.rows
    df_chain = pd.DataFrame([{
        "strike": r.strike,
        "ce_oi": r.ce_oi, "ce_oi_change": r.ce_oi_change,
        "ce_volume": r.ce_volume, "ce_iv": r.ce_iv, "ce_ltp": r.ce_ltp,
        "pe_oi": r.pe_oi, "pe_oi_change": r.pe_oi_change,
        "pe_volume": r.pe_volume, "pe_iv": r.pe_iv, "pe_ltp": r.pe_ltp,
        "pcr_oi": r.pcr_oi,
    } for r in sorted(atm_rows, key=lambda x: x.strike)])

    # ── Tabs: charts + table
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 OI Heatmap", "📈 OI Change", "🌀 IV Smile", "📋 Chain Table"]
    )

    with tab1:
        if not df_chain.empty:
            st.plotly_chart(
                oi_heatmap(df_chain, spot, snapshot.atm_strike or spot),
                use_container_width=True,
            )

    with tab2:
        if not df_chain.empty:
            st.plotly_chart(oi_change_chart(df_chain), use_container_width=True)

    with tab3:
        iv_data = df_chain[(df_chain["ce_iv"] > 0) | (df_chain["pe_iv"] > 0)]
        if not iv_data.empty:
            st.plotly_chart(iv_skew_chart(iv_data), use_container_width=True)
        else:
            st.info("IV data not available from NSE for this expiry.")

    with tab4:
        if not df_chain.empty:
            # Highlight ATM row
            def _highlight_atm(row):
                if snapshot.atm_strike and row["strike"] == snapshot.atm_strike:
                    return ["background-color: #1a2e4a"] * len(row)
                return [""] * len(row)

            display = df_chain[["strike", "ce_oi", "ce_oi_change", "ce_iv", "ce_ltp",
                                  "pe_oi", "pe_oi_change", "pe_iv", "pe_ltp", "pcr_oi"]]
            st.dataframe(
                display.style.apply(_highlight_atm, axis=1),
                use_container_width=True,
                height=500,
            )

    # ── PCR History
    st.divider()
    st.subheader("PCR Intraday Trend")
    hours = st.select_slider("Look-back", [1, 2, 3, 6, 12], value=6)
    history = load_pcr_history(symbol, hours=hours)
    if history:
        st.plotly_chart(pcr_trend_chart(history), use_container_width=True)
    else:
        st.info("No PCR history in DB yet. OI data accumulates as the scheduler runs during market hours.")

    # ── Last updated
    st.caption(f"Last fetched: {snapshot.timestamp.strftime('%H:%M:%S IST')} | "
               f"Expiry: {snapshot.expiry} | "
               f"{len(snapshot.rows)} total strikes")


# ── Standalone entry point
if __name__ == "__main__":
    render_oi_dashboard()
