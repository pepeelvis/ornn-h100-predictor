"""OCPI (Ornn) vs GX Hopper US spread model.

Both are $/GPU-hour benchmarks but not identical in scope: Ornn's OCPI H100
series is single-SKU (H100 SXM only), while GX Hopper US blends H100 *and*
H200 rentals. The spread between them is therefore a real, tradeable basis
-- part hardware-mix effect (H200 commands a premium over H100), part
index-construction methodology (Ornn's printed-trades-only vs GX's
IOSCO-benchmark panel) -- not just noise.

Model: treat the spread as a mean-reverting AR(1) process,

    spread_t = a + b * spread_(t-1) + e_t

fit by OLS on daily history. `b` implies a half-life (days for a dislocation
to close halfway); `a`/`b` give a one-day-ahead forecast. A walk-forward
backtest scores this against a naive "spread stays flat" baseline, same
spirit as model.py's H100 backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ZSCORE_WINDOW = 20


def build_spread(ocpi_h100: pd.Series, gx_hopper: pd.Series) -> pd.DataFrame:
    df = pd.concat([ocpi_h100.rename("ocpi_h100"), gx_hopper.rename("gx_hopper")], axis=1).dropna()
    if df.empty:
        raise ValueError("No overlapping dates between OCPI and GX Hopper history")
    df["spread"] = df["ocpi_h100"] - df["gx_hopper"]
    roll = df["spread"].rolling(ZSCORE_WINDOW, min_periods=5)
    df["spread_zscore"] = (df["spread"] - roll.mean()) / roll.std()
    return df


def fit_ar1(spread: pd.Series) -> dict:
    """OLS fit of spread_t on spread_(t-1); returns intercept/slope, residual
    std, and the implied mean-reversion half-life in days."""
    if len(spread) < 5:
        raise ValueError("Need at least 5 spread observations to fit AR(1)")
    y = spread.values[1:]
    x = spread.values[:-1]
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    half_life = float(np.log(0.5) / np.log(b)) if 0 < b < 1 else float("inf")
    return {"a": float(a), "b": float(b), "resid_std": float(resid.std()), "half_life_days": half_life}


def forecast_next_spread(spread: pd.Series, params: dict | None = None) -> float:
    params = params or fit_ar1(spread)
    return float(params["a"] + params["b"] * spread.iloc[-1])


def walk_forward_backtest(spread: pd.Series, n_test: int = 20) -> dict:
    """Re-fit AR(1) at each step using only data available up to that point,
    predict the next day's spread, compare to naive (spread stays flat)."""
    n_test = min(n_test, len(spread) - 15)
    if n_test < 5:
        raise ValueError("Not enough overlapping OCPI/GX history for a spread backtest")
    naive_err, ar1_err = [], []
    for i in range(len(spread) - n_test, len(spread) - 1):
        train = spread.iloc[: i + 1]
        actual_next = spread.iloc[i + 1]
        params = fit_ar1(train)
        pred = forecast_next_spread(train, params)
        naive_err.append(abs(train.iloc[-1] - actual_next))
        ar1_err.append(abs(pred - actual_next))
    return {
        "mae_naive": float(np.mean(naive_err)),
        "mae_ar1": float(np.mean(ar1_err)),
        "n_test": n_test,
    }


def signal(zscore: float) -> str:
    """A dislocation-of-more-than-1.5-sigma flags a basis-trade idea; this is
    a read on the *index* spread, not a claim about which physical GPU
    market to actually trade against it."""
    if pd.isna(zscore):
        return "insufficient history for a signal"
    if zscore > 1.5:
        return "OCPI rich vs GX Hopper -- spread historically mean-reverts down from here"
    if zscore < -1.5:
        return "OCPI cheap vs GX Hopper -- spread historically mean-reverts up from here"
    return "within normal range -- no dislocation signal"


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from fetch import fetch_gpu_history
    from gx_fetch import fetch_gx_index

    ocpi = fetch_gpu_history("H100 SXM")
    gx = fetch_gx_index()
    df = build_spread(ocpi, gx)

    bt = walk_forward_backtest(df["spread"])
    params = fit_ar1(df["spread"])
    fc = forecast_next_spread(df["spread"], params)
    z = df["spread_zscore"].iloc[-1]

    print(f"{len(df)} overlapping days, {df.index.min()} -> {df.index.max()}")
    print(f"Latest: OCPI={df['ocpi_h100'].iloc[-1]:.3f}  GX Hopper={df['gx_hopper'].iloc[-1]:.3f}  "
          f"spread={df['spread'].iloc[-1]:.3f}  z={z:.2f}")
    print(f"AR(1): a={params['a']:.4f} b={params['b']:.4f} half_life={params['half_life_days']:.1f}d")
    print(f"Backtest MAE: ar1={bt['mae_ar1']:.4f} vs naive={bt['mae_naive']:.4f}  (n={bt['n_test']})")
    print(f"Next-day spread forecast: {fc:.3f}")
    print(f"Signal: {signal(z)}")
