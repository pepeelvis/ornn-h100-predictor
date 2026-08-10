"""Pull GX Hopper US (General Index / Compute Desk) history from the GX API.

GX Hopper US is General Index's daily $/GPU-hour benchmark covering H100 +
H200 rentals, built to IOSCO benchmark principles. Unlike Ornn's OCPI, there
is no free/anonymous tier -- every call needs a GX account.

Setup (one-time, ~5 minutes):
  1. Sign up for the free GX Go trial at https://app.g-x.co (no credit card
     required for the 14-day trial).
  2. In the GX Go catalog, search "GX Hopper US" and note its GX index code
     (looks like "GX0001234").
  3. Generate an API key at https://app.g-x.co/MyAccount#Settings.
  4. Set environment variables:
       export GX_API_TOKEN=<your api key>
       export GX_HOPPER_CODE=<the GX0... code from step 2>
     (or GX_USERNAME / GX_PASSWORD instead of GX_API_TOKEN -- this module
     will log in and exchange them for a bearer token itself)

Reference: https://docs.g-x.co/
"""
from __future__ import annotations

import datetime as dt
import os

import pandas as pd
import requests

BASE_URL = "https://api.g-x.co"
SETUP_HINT = (
    "GX (General Index) credentials not found.\n"
    "  1. Sign up for the free GX Go trial: https://app.g-x.co\n"
    "  2. Find 'GX Hopper US' in the index catalog to get its GX code (e.g. GX0001234)\n"
    "  3. Generate an API key: https://app.g-x.co/MyAccount#Settings\n"
    "  4. export GX_API_TOKEN=<key>   export GX_HOPPER_CODE=<code>\n"
    "     (or export GX_USERNAME=... GX_PASSWORD=... instead of GX_API_TOKEN)"
)


def _get_token() -> str:
    token = os.environ.get("GX_API_TOKEN")
    if token:
        return token
    user, pw = os.environ.get("GX_USERNAME"), os.environ.get("GX_PASSWORD")
    if not (user and pw):
        raise RuntimeError(SETUP_HINT)
    resp = requests.get(f"{BASE_URL}/auth/login", auth=(user, pw), timeout=30)
    resp.raise_for_status()
    return resp.json()["token"]


def fetch_gx_index(code: str | None = None, start: dt.date | None = None, end: dt.date | None = None) -> pd.Series:
    """Fetch one GX index's daily history as a date-indexed Series of Mid prices."""
    code = code or os.environ.get("GX_HOPPER_CODE")
    if not code:
        raise RuntimeError(SETUP_HINT)
    token = _get_token()
    end = end or dt.date.today()
    start = start or (end - dt.timedelta(days=3650))
    resp = requests.get(
        f"{BASE_URL}/index",
        params={"code": code, "from": start.isoformat(), "to": end.isoformat(), "metadata": "false"},
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    rows = payload if isinstance(payload, list) else payload.get("data", payload)
    if not rows:
        raise RuntimeError(f"GX API returned no rows for code {code!r}")
    df = pd.DataFrame(rows)
    date_col = "Date(UTC)" if "Date(UTC)" in df.columns else "Date"
    df["date"] = pd.to_datetime(df[date_col]).dt.date
    s = df.drop_duplicates(subset="date", keep="last").set_index("date")["Mid"].astype(float)
    s.name = "gx_hopper_us"
    return s.sort_index()


if __name__ == "__main__":
    s = fetch_gx_index()
    print(f"Fetched {len(s)} days, {s.index.min()} -> {s.index.max()}")
    print(s.tail())
