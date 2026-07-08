"""
app.py
------
Streamlit UI: enter a ticker, get a Bloomberg-styled dashboard with
pre-market / current / post-market prices, 1..90 days directional forecasts,
a historical+predicted chart, and a per-prediction CSV log.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import history_store
from data_service import TickerSnapshot, fetch_snapshot
from predictor import PredictionResult, predict

# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TERMINAL // Stock / ETF / Mutual Fund Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bloomberg-ish palette
BG          = "#0A0A0A"
PANEL       = "#141414"
PANEL_HI    = "#1C1C1C"
BORDER      = "#2A2A2A"
AMBER       = "#FFA500"
AMBER_DIM   = "#7A4F00"
GREEN       = "#26FF7A"
RED         = "#FF3B3B"
TEXT        = "#E6E6E6"
MUTED       = "#7A7A7A"

CUSTOM_CSS = f"""
<style>
:root {{
  --bg: {BG};
  --panel: {PANEL};
  --panel-hi: {PANEL_HI};
  --border: {BORDER};
  --amber: {AMBER};
  --green: {GREEN};
  --red: {RED};
  --text: {TEXT};
  --muted: {MUTED};
}}

/* base */
html, body, [class*="css"], .stApp {{
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: "JetBrains Mono","SF Mono",Consolas,monospace !important;
}}
.block-container {{ padding-top: 1.2rem !important; max-width: 1400px; }}

/* sidebar */
section[data-testid="stSidebar"] {{
  background: #0E0E0E !important;
  border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] * {{ color: var(--text) !important; }}

/* hide default streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}

/* ----- BIG TICKER INPUT ----- */
.ticker-label {{
  color: var(--amber);
  font-size: 11px;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  margin-bottom: 6px;
}}
div[data-testid="stTextInput"] input {{
  background: var(--panel) !important;
  color: var(--amber) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
  font-family: "JetBrains Mono","SF Mono",Consolas,monospace !important;
  font-size: 30px !important;
  font-weight: 700 !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  height: 40px !important;
  padding: 0 24px !important;
  caret-color: var(--amber) !important;
}}
div[data-testid="stTextInput"] input:focus {{
  border-color: var(--amber) !important;
  box-shadow: 0 0 0 1px var(--amber) !important;
  outline: none !important;
}}
div[data-testid="stTextInput"] input::placeholder {{
  color: {AMBER_DIM} !important;
  font-weight: 500;
}}

/* button */
div[data-testid="stButton"] > button {{
  background: var(--amber) !important;
  color: #000 !important;
  border: none !important;
  border-radius: 2px !important;
  font-family: inherit !important;
  font-weight: 700 !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  height: 40px !important;
  font-size: 16px !important;
  width: 100% !important;
}}
div[data-testid="stButton"] > button:hover {{
  background: #FFB733 !important;
  color: #000 !important;
}}

/* panels */
.panel {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 2px;
  padding: 14px 18px;
  margin-bottom: 12px;
}}
.panel h3 {{
  margin: 0 0 10px 0;
  color: var(--amber);
  font-size: 11px;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  font-weight: 600;
  border-bottom: 1px solid var(--border);
  padding-bottom: 6px;
}}

.kv {{ display: flex; justify-content: space-between; padding: 4px 0;
       font-size: 13px; }}
.kv .k {{ color: var(--muted); }}
.kv .v {{ color: var(--text); font-weight: 600; }}

.price-big {{ font-size: 34px; font-weight: 700; letter-spacing: 0.02em; }}
.price-lbl {{ font-size: 10px; letter-spacing: 0.3em;
              text-transform: uppercase; color: var(--muted); }}

.up   {{ color: var(--green) !important; }}
.down {{ color: var(--red)   !important; }}
.flat {{ color: var(--muted) !important; }}

.fc-row {{ display: grid; grid-template-columns: 60px 1fr 1fr 1fr 1fr;
           gap: 12px; align-items: center; padding: 10px 0;
           border-bottom: 1px solid var(--border); font-size: 13px; }}
.fc-row:last-child {{ border-bottom: none; }}
.fc-row .day {{ color: var(--amber); font-weight: 700; letter-spacing: 0.1em; }}
.fc-row .price {{ font-weight: 700; }}
.fc-row .pct {{ font-weight: 700; }}
.fc-row .dir {{ font-weight: 700; letter-spacing: 0.1em; }}

.conf-bar {{ background: #222; height: 6px; border-radius: 1px;
             position: relative; overflow: hidden; }}
.conf-bar > span {{ display: block; height: 100%; background: var(--amber); }}

.header-bar {{
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 8px 0 16px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 16px;
}}
.header-bar .brand {{
  color: var(--amber);
  font-size: 12px; letter-spacing: 0.4em; font-weight: 700;
}}
.header-bar .clock {{
  color: var(--muted); font-size: 11px; letter-spacing: 0.2em;
}}

.indicator-grid {{ display: grid; grid-template-columns: repeat(4, 1fr);
                    gap: 10px; }}
.ind-cell {{ background: var(--panel-hi); padding: 10px 12px;
              border: 1px solid var(--border); }}
.ind-cell .lbl {{ font-size: 9px; letter-spacing: 0.3em;
                   color: var(--muted); text-transform: uppercase; }}
.ind-cell .val {{ font-size: 18px; font-weight: 700; margin-top: 4px; }}

.disclaimer {{
  margin-top: 28px; padding-top: 12px;
  border-top: 1px solid var(--border);
  color: var(--muted); font-size: 10px; letter-spacing: 0.05em;
  line-height: 1.6;
}}

.watch-pill {{
  display: inline-block; background: var(--panel-hi);
  border: 1px solid var(--border); padding: 6px 12px; margin: 2px;
  font-size: 12px; letter-spacing: 0.1em;
}}
.watch-pill.up   {{ border-color: var(--green); }}
.watch-pill.down {{ border-color: var(--red); }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — settings, watchlist, history
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='color:#FFA500;font-weight:700;letter-spacing:0.3em;"
        "font-size:12px;margin-bottom:14px;'>◼ TERMINAL</div>",
        unsafe_allow_html=True,
    )

    horizon = st.slider("Forecast horizon (days)", 1, 90, 90)
    period = st.selectbox(
        "History window",
        ["3mo", "6mo", "1y", "2y"],
        index=0,
    )
    save_to_csv = st.checkbox("Log prediction to CSV", value=False)

    st.markdown("---")
    st.markdown("**Watchlist**")
    watch_raw = st.text_area(
        "Comma-separated tickers",
        value=st.session_state.get("watchlist", "AMA, GRAG, ORCS, BWET"),
        height=80,
        label_visibility="collapsed",
    )
    st.session_state["watchlist"] = watch_raw

    st.markdown("---")
    st.markdown("**Prediction Log**")
    log_df = history_store.load()
    st.caption(f"{len(log_df)} rows")
    if not log_df.empty:
        st.download_button(
            "Download CSV",
            data=log_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("Clear log", use_container_width=True):
            history_store.clear()
            st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
now_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
st.markdown(
    f"""
    <div class="header-bar">
      <div class="brand">◼ STOCK / ETF / MUTUAL FUND FORECAST TERMINAL</div>
      <div class="clock">{now_str}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# BIG TICKER INPUT
# ---------------------------------------------------------------------------
st.markdown('<div class="ticker-label">▸ Enter Ticker Symbol</div>',
            unsafe_allow_html=True)

c_in, c_btn = st.columns([4, 1])
with c_in:
    symbol = st.text_input(
        "ticker",
        value=st.session_state.get("symbol", "AMA"),
        placeholder="AMA",
        label_visibility="collapsed",
        key="symbol_input",
    )
with c_btn:
    go_clicked = st.button("▶ ANALYZE", use_container_width=True)

if symbol:
    st.session_state["symbol"] = symbol.strip().upper()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_price(v, currency="USD"):
    if v is None:
        return "—"
    sym = "$" if currency == "USD" else ""
    return f"{sym}{v:,.2f}"


def fmt_pct(v):
    if v is None:
        return "—"
    return f"{v:+.2f}%"


def fmt_int(v):
    if v is None:
        return "—"
    if v >= 1_000_000_000:
        return f"{v/1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v/1_000:.1f}K"
    return f"{v:,}"


def color_class(v):
    if v is None:
        return "flat"
    if v > 0:
        return "up"
    if v < 0:
        return "down"
    return "flat"


# ---------------------------------------------------------------------------
# Cached fetch
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def cached_snapshot(sym: str, period: str) -> TickerSnapshot:
    return fetch_snapshot(sym, history_period=period)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
sym = st.session_state.get("symbol", "").strip().upper()

if sym:
    with st.spinner(f"Fetching {sym}…"):
        snap = cached_snapshot(sym, period)

    if not snap.is_valid:
        st.error(f"Could not load data for **{sym}**. "
                 f"Check the symbol and your internet connection. "
                 f"{snap.name}")
    else:
        # ---- top row: identity + 3 price panels ---------------------------
        col_id, col_pre, col_cur, col_post = st.columns([1.4, 1, 1, 1])

        with col_id:
            chg_cls = color_class(snap.day_change)
            st.markdown(
                f"""
                <div class="panel" style="height:100%;">
                  <h3>Instrument</h3>
                  <div style="font-size:24px;font-weight:700;color:var(--amber);
                              letter-spacing:0.1em;">{snap.symbol}</div>
                  <div style="color:var(--text);font-size:13px;margin-top:4px;">
                    {snap.name}
                  </div>
                  <div style="color:var(--muted);font-size:11px;
                              letter-spacing:0.15em;text-transform:uppercase;
                              margin-top:6px;">{snap.sector or '—'}</div>
                  <div style="margin-top:14px;" class="{chg_cls}">
                    <span style="font-size:20px;font-weight:700;">
                      {fmt_pct(snap.day_change_pct)}
                    </span>
                    <span style="font-size:13px;margin-left:8px;">
                      ({fmt_price(snap.day_change, snap.currency)})
                    </span>
                  </div>
                  <div style="font-size:10px;letter-spacing:0.2em;
                              color:var(--muted);margin-top:6px;">
                    MKT STATE: {snap.market_state}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        for col, label, val in [
            (col_pre, "Pre-Market", snap.pre_market),
            (col_cur, "Current",    snap.current),
            (col_post, "Post-Market", snap.post_market),
        ]:
            with col:
                st.markdown(
                    f"""
                    <div class="panel" style="height:100%;">
                      <h3>{label}</h3>
                      <div class="price-lbl">{snap.currency}</div>
                      <div class="price-big">{fmt_price(val, snap.currency)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ---- run prediction -----------------------------------------------
        result = predict(snap.history, days=horizon)

        # ---- indicators row -----------------------------------------------
        rsi_color = (
            "var(--red)" if result.rsi > 70
            else "var(--green)" if result.rsi < 30
            else "var(--text)"
        )
        sig_color = (
            "var(--green)" if result.signal_score > 0.1
            else "var(--red)" if result.signal_score < -0.1
            else "var(--muted)"
        )
        st.markdown(
            f"""
            <div class="panel">
              <h3>Technical Indicators</h3>
              <div class="indicator-grid">
                <div class="ind-cell">
                  <div class="lbl">RSI (14)</div>
                  <div class="val" style="color:{rsi_color};">{result.rsi}</div>
                </div>
                <div class="ind-cell">
                  <div class="lbl">SMA 10 / 30</div>
                  <div class="val">{result.sma_short} / {result.sma_long}</div>
                </div>
                <div class="ind-cell">
                  <div class="lbl">Momentum 10D</div>
                  <div class="val {color_class(result.momentum_pct)}">
                    {fmt_pct(result.momentum_pct)}
                  </div>
                </div>
                <div class="ind-cell">
                  <div class="lbl">Signal Score</div>
                  <div class="val" style="color:{sig_color};">
                    {result.signal_score:+.3f}
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- forecast table + chart ---------------------------------------
        col_fc, col_chart = st.columns([1, 1.6])

        with col_fc:
            rows = ""
            for fc in result.forecasts:
                cls = ("up" if fc.direction == "UP"
                       else "down" if fc.direction == "DOWN"
                       else "flat")
                arrow = "▲" if fc.direction == "UP" else "▼" if fc.direction == "DOWN" else "—"
                rows += f"""
                <div class="fc-row">
                  <div class="day">D+{fc.day}</div>
                  <div class="price">{fmt_price(fc.target_price, snap.currency)}</div>
                  <div class="pct {cls}">{fmt_pct(fc.change_pct)}</div>
                  <div class="dir {cls}">{arrow} {fc.direction}</div>
                  <div>
                    <div style="font-size:10px;color:var(--muted);
                                letter-spacing:0.1em;">CONF {fc.confidence}%</div>
                    <div class="conf-bar"><span style="width:{fc.confidence}%;"></span></div>
                  </div>
                </div>
                """
            st.markdown(
                f"""
                <div class="panel">
                  <h3>Forecast — Next {horizon} Trading Days</h3>
                  <div class="fc-row" style="color:var(--muted);font-size:10px;
                       letter-spacing:0.2em;text-transform:uppercase;">
                    <div>Day</div><div>Target</div><div>Change</div>
                    <div>Dir</div><div>Confidence</div>
                  </div>
                  {rows}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_chart:
            # build projection series
            hist = snap.history.copy()
            last_date = hist.index[-1]
            future_dates = [last_date + timedelta(days=i)
                            for i in range(1, horizon + 1)]
            future_prices = [fc.target_price for fc in result.forecasts]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist["Close"],
                mode="lines",
                line=dict(color=AMBER, width=1.6),
                name="Close",
            ))
            # bridge last actual to first projected
            fig.add_trace(go.Scatter(
                x=[last_date] + future_dates,
                y=[float(hist["Close"].iloc[-1])] + future_prices,
                mode="lines+markers",
                line=dict(color=GREEN if result.signal_score >= 0 else RED,
                          width=2, dash="dot"),
                marker=dict(size=7, symbol="diamond"),
                name=f"Forecast {horizon}D",
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=PANEL,
                plot_bgcolor=PANEL,
                font=dict(family="JetBrains Mono, monospace",
                          color=TEXT, size=11),
                margin=dict(l=10, r=10, t=30, b=10),
                height=420,
                xaxis=dict(gridcolor=BORDER, showgrid=True, zeroline=False),
                yaxis=dict(gridcolor=BORDER, showgrid=True, zeroline=False,
                           tickprefix="$"),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                ),
                title=dict(
                    text=f"  {snap.symbol}  //  PRICE + FORECAST",
                    font=dict(color=AMBER, size=12),
                    x=0,
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---- range info ---------------------------------------------------
        st.markdown(
            f"""
            <div class="panel">
              <h3>Range &amp; Volume</h3>
              <div class="indicator-grid">
                <div class="ind-cell">
                  <div class="lbl">52W High</div>
                  <div class="val up">{fmt_price(snap.week52_high, snap.currency)}</div>
                </div>
                <div class="ind-cell">
                  <div class="lbl">52W Low</div>
                  <div class="val down">{fmt_price(snap.week52_low, snap.currency)}</div>
                </div>
                <div class="ind-cell">
                  <div class="lbl">Volume</div>
                  <div class="val">{fmt_int(snap.volume)}</div>
                </div>
                <div class="ind-cell">
                  <div class="lbl">Avg Volume</div>
                  <div class="val">{fmt_int(snap.avg_volume)}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- watchlist ----------------------------------------------------
        watch_syms = [
            s.strip().upper() for s in st.session_state.get("watchlist", "").split(",")
            if s.strip()
        ]
        if watch_syms:
            pills = ""
            for ws in watch_syms[:12]:
                try:
                    w = cached_snapshot(ws, "3mo")
                    if not w.is_valid:
                        continue
                    cls = ("up" if (w.day_change_pct or 0) > 0
                           else "down" if (w.day_change_pct or 0) < 0
                           else "flat")
                    pills += (
                        f'<span class="watch-pill {cls}">'
                        f'<b>{w.symbol}</b> &nbsp; '
                        f'{fmt_price(w.current, w.currency)} &nbsp; '
                        f'<span class="{cls}">{fmt_pct(w.day_change_pct)}</span>'
                        f'</span>'
                    )
                except Exception:
                    continue
            if pills:
                st.markdown(
                    f"""
                    <div class="panel">
                      <h3>Watchlist</h3>
                      <div>{pills}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ---- save to csv --------------------------------------------------
        if go_clicked and save_to_csv:
            history_store.append(snap.symbol, result)
            st.toast(f"Logged {snap.symbol} prediction to CSV", icon="💾")

else:
    st.info("Enter a ticker above and press **ANALYZE** to begin.")

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer">
      ⓘ &nbsp; This tool combines simple technical indicators (SMA, RSI, momentum, recent return)
      into a directional signal. It is <b>not</b> a predictive model, not investment advice,
      and not a recommendation to buy or sell any security. Markets are stochastic;
      forecasts are illustrative only. Data via Yahoo Finance and may be delayed.
      Use at your own risk.
    </div>
    """,
    unsafe_allow_html=True,
)
