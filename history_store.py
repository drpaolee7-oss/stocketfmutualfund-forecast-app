"""
history_store.py
----------------
Append-only CSV log of predictions for later review.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import List

import pandas as pd

from predictor import PredictionResult

CSV_PATH = "prediction_history.csv"

FIELDS = [
    "timestamp_utc",
    "symbol",
    "current_price",
    "rsi",
    "sma_short",
    "sma_long",
    "momentum_pct",
    "signal_score",
    "day",
    "target_price",
    "change_pct",
    "direction",
    "confidence",
]


def append(symbol: str, result: PredictionResult, path: str = CSV_PATH) -> None:
    """Append one row per forecast day."""
    new_file = not os.path.exists(path)
    ts = datetime.utcnow().isoformat(timespec="seconds")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for fc in result.forecasts:
            w.writerow({
                "timestamp_utc": ts,
                "symbol": symbol,
                "current_price": result.current_price,
                "rsi": result.rsi,
                "sma_short": result.sma_short,
                "sma_long": result.sma_long,
                "momentum_pct": result.momentum_pct,
                "signal_score": result.signal_score,
                "day": fc.day,
                "target_price": fc.target_price,
                "change_pct": fc.change_pct,
                "direction": fc.direction,
                "confidence": fc.confidence,
            })


def load(path: str = CSV_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=FIELDS)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=FIELDS)


def clear(path: str = CSV_PATH) -> None:
    if os.path.exists(path):
        os.remove(path)
