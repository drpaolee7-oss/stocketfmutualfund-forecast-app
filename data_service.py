"""
data_service.py
---------------
Thin wrapper around yfinance that gives us everything the UI needs in one
predictable shape. All network access goes through here so the rest of the
app stays easy to test and reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf


@dataclass
class TickerSnapshot:
    """Everything we want to show for a single ticker at a moment in time."""
    symbol: str
    name: str = ""
    sector: str = ""
    currency: str = "USD"

    pre_market: Optional[float] = None
    current: Optional[float] = None
    post_market: Optional[float] = None

    previous_close: Optional[float] = None
    day_change: Optional[float] = None
    day_change_pct: Optional[float] = None

    volume: Optional[int] = None
    avg_volume: Optional[int] = None

    week52_high: Optional[float] = None
    week52_low: Optional[float] = None

    market_state: str = "UNKNOWN"  # PRE | REGULAR | POST | CLOSED
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    history: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def is_valid(self) -> bool:
        return self.current is not None and not self.history.empty


def _safe(d: dict, *keys):
    """Walk a dict via fallback keys, returning the first non-null value."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def fetch_snapshot(symbol: str, history_period: str = "6mo") -> TickerSnapshot:
    """Pull a full snapshot for `symbol`. Returns an empty snapshot on failure."""
    symbol = symbol.strip().upper()
    snap = TickerSnapshot(symbol=symbol)
    if not symbol:
        return snap

    try:
        tk = yf.Ticker(symbol)

        # fast_info is cheaper and more reliable than .info for prices
        fast = {}
        try:
            fast = dict(tk.fast_info) if tk.fast_info else {}
        except Exception:
            fast = {}

        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        snap.name = info.get("longName") or info.get("shortName") or symbol
        snap.sector = info.get("sector") or info.get("quoteType", "") or ""
        snap.currency = _safe(fast, "currency") or info.get("currency") or "USD"

        snap.current = _safe(fast, "last_price", "lastPrice") or info.get("regularMarketPrice")
        snap.previous_close = (
            _safe(fast, "previous_close", "previousClose")
            or info.get("regularMarketPreviousClose")
        )
        snap.pre_market = info.get("preMarketPrice")
        snap.post_market = info.get("postMarketPrice")

        snap.volume = _safe(fast, "last_volume", "lastVolume") or info.get("regularMarketVolume")
        snap.avg_volume = info.get("averageVolume") or info.get("averageDailyVolume10Day")

        snap.week52_high = (
            _safe(fast, "year_high", "yearHigh") or info.get("fiftyTwoWeekHigh")
        )
        snap.week52_low = (
            _safe(fast, "year_low", "yearLow") or info.get("fiftyTwoWeekLow")
        )

        snap.market_state = info.get("marketState", "UNKNOWN")

        if snap.current is not None and snap.previous_close:
            snap.day_change = snap.current - snap.previous_close
            snap.day_change_pct = (snap.day_change / snap.previous_close) * 100.0

        # History for chart + indicators
        hist = tk.history(period=history_period, auto_adjust=False)
        if not hist.empty:
            hist = hist.dropna(subset=["Close"])
            snap.history = hist

            # If live price wasn't returned (common for ETFs after-hours),
            # fall back to the most recent close.
            if snap.current is None:
                snap.current = float(hist["Close"].iloc[-1])
            if snap.previous_close is None and len(hist) >= 2:
                snap.previous_close = float(hist["Close"].iloc[-2])
                if snap.current is not None:
                    snap.day_change = snap.current - snap.previous_close
                    snap.day_change_pct = (
                        snap.day_change / snap.previous_close
                    ) * 100.0

    except Exception as exc:
        # Surface as an empty snapshot — UI checks .is_valid
        snap.name = f"(error: {exc})"

    return snap
