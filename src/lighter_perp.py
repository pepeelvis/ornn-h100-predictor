"""Fetch H100 perpetual-future data from Lighter (on-chain perp DEX).

Lighter's *trading frontend* (app.lighter.xyz) is geo-fenced and needs a VPN
from restricted regions for compliance reasons -- that block is specifically
about executing trades, not about reading public chain state. The market
data API (mainnet.zklighter.elliot.ai) is unauthenticated, unrestricted, and
answers directly with no VPN required; this module only reads from it.

A perpetual has no fixed expiry, so it isn't a point on a dated forward
curve the way a Kalshi monthly/yearly contract is. What it does give you is
a continuously-live cash price (mark_price) plus Lighter's own oracle
estimate of the OCPI H100 print (index_price) and the funding rate --
i.e. the market's real-time view of compute cost of carry, updating far
faster than Ornn's once-a-day publish.

Reference: https://apidocs.lighter.xyz
"""
from __future__ import annotations

import requests

BASE_URL = "https://mainnet.zklighter.elliot.ai/api/v1"


def find_market_id(symbol: str = "H100") -> int:
    resp = requests.get(f"{BASE_URL}/orderBooks", headers={"Accept": "application/json"}, timeout=30)
    resp.raise_for_status()
    for ob in resp.json()["order_books"]:
        if ob["symbol"] == symbol:
            return ob["market_id"]
    raise RuntimeError(f"No Lighter market found for symbol {symbol!r}")


def fetch_perp_snapshot(symbol: str = "H100") -> dict:
    market_id = find_market_id(symbol)
    resp = requests.get(
        f"{BASE_URL}/orderBookDetails",
        params={"market_id": market_id},
        headers={"Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    d = resp.json()["order_book_details"][0]
    return {
        "symbol": symbol,
        "mark_price": float(d["mark_price"]),
        "index_price": float(d["index_price"]),  # Lighter's oracle estimate of OCPI
        "last_trade_price": float(d["last_trade_price"]),
        "daily_price_low": float(d["daily_price_low"]),
        "daily_price_high": float(d["daily_price_high"]),
        "daily_price_change_pct": float(d["daily_price_change"]),
        "open_interest": float(d["open_interest"]),
        "daily_quote_volume": float(d["daily_quote_token_volume"]),
    }


def basis_vs_ocpi(perp: dict, ocpi_spot: float) -> float:
    """mark_price minus the latest Ornn OCPI H100 print -- positive means
    the perp is trading rich to the daily index (bullish cost-of-carry
    sentiment intraday), negative means cheap."""
    return perp["mark_price"] - ocpi_spot


if __name__ == "__main__":
    snap = fetch_perp_snapshot()
    print("Lighter H100 perp snapshot:")
    for k, v in snap.items():
        print(f"  {k}: {v}")
