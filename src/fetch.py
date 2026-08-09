"""Pull daily index history from Ornn's public dashboard API.

Ornn doesn't publish a documented public API. This hits the same endpoint
the dashboard's own frontend calls (found by inspecting its JS bundle):

    GET https://dashboard.ornnai.com/api/gpu/{gpu_type}/index-history?startDate=...&endDate=...

Without an authenticated Ornn Data account, the API caps the response at the
trailing ~3 months regardless of the requested date range (it reports back
`"access": "public-3mo"`). That's the ceiling on how much history this
predictor can see.
"""
from __future__ import annotations

import datetime as dt
import urllib.parse

import pandas as pd
import requests

BASE_URL = "https://dashboard.ornnai.com/api/gpu/{gpu}/index-history"
USER_AGENT = "Mozilla/5.0 (predictor-bot; +https://github.com)"

# GPU types as accepted by the API (from its own "Invalid GPU type" error
# message). H100 is the prediction target; the others are correlated
# markets used as extra features.
TARGET_GPU = "H100 SXM"
FEATURE_GPUS = ["H200", "A100 SXM4", "RTX 5090", "B200"]
ALL_GPUS = [TARGET_GPU] + FEATURE_GPUS


def fetch_gpu_history(gpu_type: str, start: dt.date | None = None, end: dt.date | None = None) -> pd.Series:
    """Fetch one GPU's daily index history as a date-indexed Series."""
    end = end or dt.date.today()
    start = start or (end - dt.timedelta(days=3650))
    url = BASE_URL.format(gpu=urllib.parse.quote(gpu_type))
    resp = requests.get(
        url,
        params={"startDate": start.isoformat(), "endDate": end.isoformat()},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"Ornn API error for {gpu_type!r}: {payload}")
    rows = payload.get("data", [])
    if not rows:
        raise RuntimeError(f"Ornn API returned no data for {gpu_type!r}")
    s = pd.DataFrame(rows)
    s["date"] = pd.to_datetime(s["timestamp"]).dt.date
    s = s.drop_duplicates(subset="date", keep="last").set_index("date")["index_value"]
    s.name = gpu_type
    return s.sort_index()


def fetch_all(gpu_types: list[str] = ALL_GPUS) -> pd.DataFrame:
    """Fetch all series and align them into one wide DataFrame (date index)."""
    series = {g: fetch_gpu_history(g) for g in gpu_types}
    wide = pd.concat(series.values(), axis=1)
    wide.columns = [c.lower().replace(" ", "_") for c in series.keys()]
    return wide.sort_index()


if __name__ == "__main__":
    df = fetch_all()
    out = "data/history.csv"
    df.to_csv(out)
    print(f"Fetched {len(df)} days, {df.index.min()} -> {df.index.max()}")
    print(f"Saved to {out}")
    print(df.tail())
