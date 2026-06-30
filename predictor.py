"""
predictor.py
------------
Honest, simple technical-indicator-based directional forecasts for 1..N days.

This is NOT a real predictive model. It combines a few well-known signals
(SMA crossover, RSI, momentum, recent return) into a composite score, then
projects a price path using the recent drift +/- a damped signal term.
Confidence reflects signal agreement, not statistical certainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class DayForecast:
    day: int                 # 1..N
    target_price: float
    change_pct: float        # vs. current
    direction: str           # "UP" | "DOWN" | "FLAT"
    confidence: float        # 0..100


@dataclass
class PredictionResult:
    current_price: float
    rsi: float
    sma_short: float
    sma_long: float
    momentum_pct: float
    signal_score: float      # -1..+1
    forecasts: List[DayForecast]


# ---------- indicators ----------

def _sma(series: pd.Series, n: int) -> float:
    if len(series) < n:
        return float(series.mean())
    return float(series.tail(n).mean())


def _rsi(series: pd.Series, n: int = 14) -> float:
    if len(series) < n + 1:
        return 50.0
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.tail(n).mean()
    avg_loss = loss.tail(n).mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _momentum_pct(series: pd.Series, n: int = 10) -> float:
    if len(series) < n + 1:
        return 0.0
    return float((series.iloc[-1] / series.iloc[-n - 1] - 1) * 100)


# ---------- main API ----------

def predict(history: pd.DataFrame, days: int = 5) -> PredictionResult:
    """
    history: DataFrame with at least a 'Close' column, indexed by date.
    days:    horizon (1..5 in the UI, but any positive int works).
    """
    close = history["Close"].astype(float).dropna()
    if len(close) < 20:
        # Not enough data — return a flat forecast so the UI can still render.
        cur = float(close.iloc[-1]) if not close.empty else 0.0
        return PredictionResult(
            current_price=cur, rsi=50.0, sma_short=cur, sma_long=cur,
            momentum_pct=0.0, signal_score=0.0,
            forecasts=[
                DayForecast(d, cur, 0.0, "FLAT", 0.0) for d in range(1, days + 1)
            ],
        )

    cur = float(close.iloc[-1])
    sma_s = _sma(close, 10)
    sma_l = _sma(close, 30)
    rsi = _rsi(close, 14)
    mom = _momentum_pct(close, 10)

    # --- Composite signal in [-1, +1] -------------------------------------
    # 1) SMA crossover: short above long is bullish
    sma_signal = np.tanh((sma_s - sma_l) / max(sma_l, 1e-9) * 20)  # -1..+1

    # 2) RSI: 50 is neutral, above is bullish, below bearish; saturate at 70/30
    rsi_signal = np.clip((rsi - 50) / 20, -1, 1)

    # 3) 10-day momentum: scale ~+/-10% to +/-1
    mom_signal = np.clip(mom / 10, -1, 1)

    # 4) Last-5-day return as a short-term tape signal
    ret5 = float(close.pct_change().tail(5).sum())  # ~sum of daily returns
    tape_signal = np.clip(ret5 * 5, -1, 1)          # scale ~+/-20% -> +/-1

    weights = np.array([0.30, 0.25, 0.25, 0.20])
    signals = np.array([sma_signal, rsi_signal, mom_signal, tape_signal])
    composite = float(np.dot(weights, signals))     # -1..+1

    # --- Path projection ---------------------------------------------------
    # Daily drift = blend of historical drift and signal-implied drift,
    # capped so we never produce silly numbers.
    daily_returns = close.pct_change().dropna()
    hist_drift = float(daily_returns.tail(30).mean())          # ~recent avg
    vol = float(daily_returns.tail(30).std() or 0.01)          # ~recent stdev

    # Signal-implied daily drift: cap at +/- 1 sigma so 5d move stays sane.
    signal_drift = composite * vol

    # Blend: 40% history, 60% signal (signal-led, but anchored).
    blended_drift = 0.4 * hist_drift + 0.6 * signal_drift

    # Agreement-based confidence: how aligned the four signals are.
    # If they all point the same way -> high confidence; if they disagree -> low.
    same_sign = np.sign(signals)
    agreement = float(np.abs(same_sign.sum())) / len(signals)  # 0..1
    base_conf = 35 + 50 * agreement * abs(composite)           # 35..85ish
    # Confidence decays with horizon.
    forecasts: List[DayForecast] = []
    for d in range(1, days + 1):
        target = cur * ((1 + blended_drift) ** d)
        change_pct = (target / cur - 1) * 100
        if abs(change_pct) < 0.15:
            direction = "FLAT"
        elif change_pct > 0:
            direction = "UP"
        else:
            direction = "DOWN"

        # Decay: day 1 keeps ~95% of base, day 5 keeps ~65%.
        decay = 0.95 - (d - 1) * 0.075
        conf = max(5.0, min(95.0, base_conf * decay))

        forecasts.append(DayForecast(
            day=d,
            target_price=round(target, 2),
            change_pct=round(change_pct, 2),
            direction=direction,
            confidence=round(conf, 1),
        ))

    return PredictionResult(
        current_price=cur,
        rsi=round(rsi, 1),
        sma_short=round(sma_s, 2),
        sma_long=round(sma_l, 2),
        momentum_pct=round(mom, 2),
        signal_score=round(composite, 3),
        forecasts=forecasts,
    )
