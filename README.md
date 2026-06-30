# Stock / ETF / Mutual Fund Forecast Terminal

A Bloomberg-styled Streamlit app that takes a stock or ETF ticker and shows:

- **Pre-market**, **current**, and **post-market** prices
- **1–90 day directional forecasts** with target price, % change, and confidence %
- Historical price chart with the predicted path overlaid
- Technical indicators (RSI, SMA crossover, momentum, composite signal)
- 52-week high/low, volume, and average volume
- Multi-ticker watchlist
- CSV log of every prediction (downloadable)

The ticker input is intentionally **large** (~56px, ~96px tall) so it reads like a trading terminal.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (typically http://localhost:8501).

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI + theme |
| `data_service.py` | yfinance wrapper → `TickerSnapshot` |
| `predictor.py` | SMA / RSI / momentum composite forecast |
| `history_store.py` | Append-only CSV log of predictions |
| `requirements.txt` | Pinned-floor dependencies |

## Disclaimer

This is **not** a predictive model and **not** investment advice. The "forecast"
is a blend of well-known technical indicators projected forward as a damped
drift. Use it for visualization and experimentation only.
