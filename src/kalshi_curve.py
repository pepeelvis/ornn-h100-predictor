"""Kalshi-implied forward curve for GPU compute prices.

Kalshi lists ladders of binary strikes ("Will H100 SXM compute be above $X
by <date>?") across weekly / monthly / yearly horizons, cash-settled against
Ornn's OCPI. Each strike's market price is the crowd's estimate of
P(price > strike) at that expiry -- i.e. one point on the implied survival
function for that date. Stitching one implied price per expiry together
gives a forward curve, the same idea Kalshi itself described when it
announced "compute forward curves" (bootstrapping a curve from ladders
across expirations).

No auth needed -- Kalshi market data (events/markets/order books) is public.
Reference: https://trading-api.kalshi.com (docs), https://api.elections.kalshi.com (live host)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests
from sklearn.isotonic import IsotonicRegression

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
_TRAPZ = getattr(np, "trapezoid", np.trapz)  # numpy>=2.0 renamed trapz -> trapezoid
REQUEST_DELAY_SECONDS = 0.2  # be polite to Kalshi's public, unauthenticated rate limit

# Confirmed live series tickers (from /trade-api/v2/series) per GPU x horizon.
GPU_SERIES = {
    "H100": {"weekly": "KXH100W", "monthly": "KXH100MON", "quarterly": "KXH100Q", "yearly": "KXH100MAX"},
    "H200": {"weekly": "KXH200W", "monthly": "KXH200MON", "quarterly": "KXH200Q", "yearly": "KXH200MAX"},
    "B200": {"weekly": "KXB200W", "monthly": "KXB200MON", "quarterly": "KXB200Q", "yearly": "KXB200MAX"},
    "A100": {"weekly": "KXA100W", "monthly": "KXA100MON", "quarterly": "KXA100Q", "yearly": "KXA100MAX"},
    "RTX5090": {"weekly": "KXRTX5090W", "monthly": "KXRTX5090MON", "quarterly": "KXRTX5090Q", "yearly": "KXRTX5090MAX"},
}


def _get(path: str, **params) -> dict:
    for attempt in range(5):
        resp = requests.get(f"{BASE_URL}/{path}", params=params, headers={"Accept": "application/json"}, timeout=30)
        if resp.status_code == 429:
            time.sleep(2 ** attempt)  # public API rate limit -- back off and retry
            continue
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY_SECONDS)
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def fetch_open_events(series_ticker: str) -> list[dict]:
    return _get("events", series_ticker=series_ticker, status="open").get("events", [])


def fetch_active_markets(event_ticker: str) -> list[dict]:
    markets = _get("markets", event_ticker=event_ticker).get("markets", [])
    return [m for m in markets if m.get("status") == "active"]


def _strike_price(m: dict) -> float | None:
    """Prefer last traded price (reflects actual consensus); fall back to
    bid/ask mid only if there's been no trade yet. Wide/stale quote spreads
    on thin strikes make raw bid/ask mid unreliable on its own."""
    last = float(m.get("last_price_dollars") or 0)
    if last > 0:
        return last
    bid, ask = float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return None


def implied_price_for_event(markets: list[dict]) -> dict | None:
    """Extract the market-implied price from a strike ladder.

    Builds the implied survival function P(price > strike) across the
    ladder, cleans it with an isotonic (decreasing) fit -- thin/stale
    strikes can otherwise violate monotonicity -- then reads off:
      - median: strike where survival crosses 0.5 (linear interpolation)
      - mean:   min_strike + trapezoidal integral of survival over the
                observed strike range (approximation: treats mass outside
                the observed range as negligible, reasonable when the
                ladder spans most of the probability mass)
    """
    rows = [(float(m["floor_strike"]), p) for m in markets if (p := _strike_price(m)) is not None]
    if len(rows) < 2:
        return {"single_point": rows[0]} if rows else None
    rows.sort()
    strikes = np.array([r[0] for r in rows])
    surv_raw = np.array([r[1] for r in rows])
    surv = IsotonicRegression(increasing=False).fit_transform(strikes, surv_raw)

    if surv.max() < 0.5 or surv.min() > 0.5:
        median = float(strikes[np.argmin(np.abs(surv - 0.5))])  # extrapolation not attempted
    else:
        median = float(np.interp(0.5, surv[::-1], strikes[::-1]))
    mean = float(strikes[0] + _TRAPZ(surv, strikes))
    return {"median": median, "mean": mean, "n_strikes": len(rows)}


def build_curve(gpu: str = "H100") -> pd.DataFrame:
    if gpu not in GPU_SERIES:
        raise ValueError(f"Unknown GPU {gpu!r}; choose from {list(GPU_SERIES)}")
    rows = []
    for horizon, ticker in GPU_SERIES[gpu].items():
        for ev in fetch_open_events(ticker):
            markets = fetch_active_markets(ev["event_ticker"])
            if not markets:
                continue
            implied = implied_price_for_event(markets)
            if implied is None:
                continue
            row = {
                "gpu": gpu,
                "horizon": horizon,
                "event_ticker": ev["event_ticker"],
                "title": ev["title"],
                "strike_date": ev["strike_date"],
            }
            row.update(implied)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["strike_date"] = pd.to_datetime(df["strike_date"])
    now = pd.Timestamp.now(tz="UTC")
    df["days_out"] = (df["strike_date"] - now).dt.days
    return df.sort_values("strike_date").reset_index(drop=True)


def build_all_curves() -> pd.DataFrame:
    return pd.concat([build_curve(g) for g in GPU_SERIES], ignore_index=True)


if __name__ == "__main__":
    import os

    df = build_all_curves()
    pd.set_option("display.width", 160)
    cols = ["gpu", "horizon", "event_ticker", "days_out", "median", "mean", "n_strikes", "single_point"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "data", "kalshi_forward_curve.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
