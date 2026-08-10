"""Combined H100 term structure: Ornn OCPI spot -> Lighter perp -> Kalshi
implied forward curve.

Three different instruments, three different time horizons, one underlying
(all settle against or track Ornn's OCPI H100 print):
  - Ornn OCPI:     the actual daily print (t=0 anchor)
  - Lighter perp:   continuous mark price, no fixed expiry (near-t=0, live)
  - Kalshi ladders: dated, gives real forward points (weekly/monthly/yearly)

Run this for a same-day snapshot of the whole curve. It doesn't try to
merge these into one smooth curve (a perpetual and a dated binary ladder
aren't natively comparable) -- it lays them side by side so contango/
backwardation and cross-venue dislocation are visible at a glance.
"""
from __future__ import annotations

import pandas as pd

from fetch import fetch_gpu_history
from kalshi_curve import build_curve
from lighter_perp import basis_vs_ocpi, fetch_perp_snapshot


def build_h100_term_structure() -> pd.DataFrame:
    ocpi = fetch_gpu_history("H100 SXM")
    spot = float(ocpi.iloc[-1])
    spot_date = ocpi.index[-1]

    rows = [{
        "venue": "Ornn OCPI",
        "instrument": "spot (daily print)",
        "days_out": 0,
        "implied_price": spot,
        "vs_spot": 0.0,
        "vs_spot_pct": 0.0,
    }]

    try:
        perp = fetch_perp_snapshot("H100")
        rows.append({
            "venue": "Lighter",
            "instrument": "H100 perp (mark)",
            "days_out": 0,
            "implied_price": perp["mark_price"],
            "vs_spot": basis_vs_ocpi(perp, spot),
            "vs_spot_pct": 100 * basis_vs_ocpi(perp, spot) / spot,
        })
    except Exception as e:  # pragma: no cover - network dependent
        print(f"  (Lighter fetch failed: {e})")

    kalshi = build_curve("H100")
    for _, r in kalshi.iterrows():
        price = r["median"] if pd.notna(r.get("median")) else None
        if price is None:
            continue
        rows.append({
            "venue": "Kalshi",
            "instrument": f"{r['horizon']} ({r['event_ticker']})",
            "days_out": r["days_out"],
            "implied_price": price,
            "vs_spot": price - spot,
            "vs_spot_pct": 100 * (price - spot) / spot,
        })

    df = pd.DataFrame(rows).sort_values("days_out").reset_index(drop=True)
    df.attrs["ocpi_spot"] = spot
    df.attrs["ocpi_spot_date"] = str(spot_date)
    return df


if __name__ == "__main__":
    import os

    df = build_h100_term_structure()
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(f"Ornn OCPI H100 spot ({df.attrs['ocpi_spot_date']}): {df.attrs['ocpi_spot']:.4f}")
    print()
    print(df.to_string(index=False))

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "data", "h100_term_structure.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
