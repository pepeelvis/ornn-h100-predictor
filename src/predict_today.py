"""Daily entrypoint: fetch the latest Ornn H100 index history, forecast
tomorrow's ~16:30 ET publish, log the prediction, and reconcile any past
predictions whose actual print has since landed.

Run this once a day, any time before the publish (data only updates once
daily, so running it earlier vs. later the same day makes no difference).
"""
from __future__ import annotations

import datetime as dt
import os
import time
import warnings

import pandas as pd
import requests

from fetch import fetch_all
from features import TARGET_COL
from lighter_perp import fetch_perp_snapshot, basis_vs_ocpi
from model import predict_next, walk_forward_backtest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG_PATH = os.path.join(ROOT, "predictions_log.csv")
LOG_COLUMNS = [
    "run_at", "target_date", "last_known_date", "last_known_value",
    "naive", "holt", "ridge", "meanrev", "blend", "lower_80", "upper_80",
    "backtest_mae_blend", "backtest_mae_naive", "actual", "abs_error",
]
LIGHTER_LOG_PATH = os.path.join(ROOT, "lighter_basis_log.csv")
LIGHTER_LOG_COLUMNS = [
    "run_at", "date", "ocpi_spot", "mark_price", "index_price",
    "last_trade_price", "basis", "basis_bps", "open_interest", "daily_quote_volume",
]


def load_log() -> pd.DataFrame:
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH, parse_dates=["target_date", "last_known_date"])
    return pd.DataFrame(columns=LOG_COLUMNS)


def reconcile(log: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Fill in actuals for past predictions whose target_date has now published."""
    actual_by_date = wide["h100_sxm"]
    for i, row in log.iterrows():
        if pd.notna(row.get("actual")):
            continue
        target = row["target_date"].date()
        if target in actual_by_date.index:
            actual = float(actual_by_date.loc[target])
            log.at[i, "actual"] = actual
            log.at[i, "abs_error"] = abs(actual - row["blend"])
    return log


def log_lighter_basis(wide: pd.DataFrame) -> None:
    """Append today's Lighter H100 perp-vs-OCPI basis to a standalone log.

    Purely a data-collection step for a future signal (2026-08-20 investigation
    found no historical Lighter series existed to backtest against, despite
    the perp being a continuously-live market on the same underlying -- a much
    stronger candidate lead indicator than the loosely-correlated external
    equities also checked that day). Not wired into the forecast yet; once
    there's ~30-60 days of history here, backtest whether the basis predicts
    next-day OCPI moves. Deliberately isolated from the main prediction: a
    Lighter API hiccup should never block today's forecast/publish.
    """
    date = wide.index[-1]
    already_logged = False
    log = pd.DataFrame(columns=LIGHTER_LOG_COLUMNS)
    if os.path.exists(LIGHTER_LOG_PATH):
        log = pd.read_csv(LIGHTER_LOG_PATH, parse_dates=["date"])
        already_logged = (log["date"].dt.date == date).any()
    if already_logged:
        print(f"  (already logged Lighter basis for {date}, not duplicating)")
        return

    ocpi_spot = float(wide[TARGET_COL].iloc[-1])
    snap = fetch_perp_snapshot()
    basis = basis_vs_ocpi(snap, ocpi_spot)
    new_row = {
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "date": pd.Timestamp(date),
        "ocpi_spot": ocpi_spot,
        "mark_price": snap["mark_price"],
        "index_price": snap["index_price"],
        "last_trade_price": snap["last_trade_price"],
        "basis": basis,
        "basis_bps": basis / ocpi_spot * 10000,
        "open_interest": snap["open_interest"],
        "daily_quote_volume": snap["daily_quote_volume"],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        log = pd.concat([log, pd.DataFrame([new_row])], ignore_index=True)
    log.to_csv(LIGHTER_LOG_PATH, index=False)
    print(f"  Lighter perp basis: {basis:+.4f} ({new_row['basis_bps']:+.0f} bps) -> {LIGHTER_LOG_PATH}")


def fetch_all_with_retry(retries: int = 3, delay_seconds: float = 30.0) -> pd.DataFrame:
    """fetch_all(), but guards against transient failures silently killing
    or corrupting the daily run. Two known incidents in production:
      - 2026-08-10: a NaN last-known value propagated through naive/holt/blend
        and got logged + published as blank cells with no error.
      - 2026-08-12: the LaunchAgent fired right as the machine woke from
        sleep, before networking was back up, so DNS resolution for the
        Ornn API raised a raw ConnectionError that crashed the whole script
        before anything could be logged -- no prediction, no publish, and
        (worse) no fired retry, since that failure mode wasn't covered here.
    Retries (with a longer pause, since a just-woken machine can take a bit
    to get networking back) on either a NaN target value or a
    connection/DNS-shaped requests exception; still fails loudly instead of
    logging garbage or silently skipping the day if the problem persists.
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            wide = fetch_all()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = f"network error fetching Ornn API: {exc} (attempt {attempt + 1}/{retries + 1})"
            print(f"  Warning: {last_error}, retrying in {delay_seconds:.0f}s...")
            time.sleep(delay_seconds)
            continue
        if pd.notna(wide[TARGET_COL].iloc[-1]):
            return wide
        last_error = f"{TARGET_COL} is NaN on {wide.index[-1]} (attempt {attempt + 1}/{retries + 1})"
        print(f"  Warning: {last_error}, retrying in {delay_seconds:.0f}s...")
        time.sleep(delay_seconds)
    raise RuntimeError(f"Ornn API fetch did not succeed after retrying: {last_error}")


def main():
    print("Fetching latest Ornn H100 index history (public, ~3mo window)...")
    wide = fetch_all_with_retry()
    print(f"  {len(wide)} days, {wide.index.min()} -> {wide.index.max()}")

    print("Running walk-forward backtest to score naive / holt / ridge / meanrev / blend...")
    bt = walk_forward_backtest(wide, n_test=25)
    for m, v in bt["mae"].items():
        print(f"  MAE[{m:>7}] = {v:.4f}  (index units, e.g. $/GPU-hr)")
    print(f"  Ensemble weights (holt excluded, see model.py): "
          f"ridge={bt['weights']['ridge']:.2f} meanrev={bt['weights']['meanrev']:.2f}")

    fc = predict_next(wide, weights=bt["weights"])
    resid_std = bt["resid_std"]
    lower, upper = fc["blend"] - 1.28 * resid_std, fc["blend"] + 1.28 * resid_std

    print()
    print(f"Prediction for {fc['target_date'].date()} ~16:30 ET publish:")
    print(f"  Last known ({fc['last_known_date']}): {fc['last_known_value']:.3f}")
    print(f"  Naive (carry-forward):                {fc['naive']:.3f}")
    print(f"  Holt (trend smoothing):                {fc['holt']:.3f}")
    print(f"  Ridge (lag/cross-GPU regression):      {fc['ridge']:.3f}")
    print(f"  Mean-reversion (7d, k=0.3):            {fc['meanrev']:.3f}")
    print(f"  >>> Blended forecast:                  {fc['blend']:.3f}")
    print(f"      80% interval:                      [{lower:.3f}, {upper:.3f}]")

    log = load_log()
    log = reconcile(log, wide)

    new_row = {
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "target_date": fc["target_date"],
        "last_known_date": fc["last_known_date"],
        "last_known_value": fc["last_known_value"],
        "naive": fc["naive"],
        "holt": fc["holt"],
        "ridge": fc["ridge"],
        "meanrev": fc["meanrev"],
        "blend": fc["blend"],
        "lower_80": lower,
        "upper_80": upper,
        "backtest_mae_blend": bt["mae"]["blend"],
        "backtest_mae_naive": bt["mae"]["naive"],
        "actual": pd.NA,
        "abs_error": pd.NA,
    }
    already_logged = (log["target_date"].dt.date == fc["target_date"].date()).any() if len(log) else False
    if not already_logged:
        new_df = pd.DataFrame([new_row])
        if log.empty:
            log = new_df
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                log = pd.concat([log, new_df], ignore_index=True)
    else:
        print(f"  (already have a prediction logged for {fc['target_date'].date()}, not duplicating)")

    log.to_csv(LOG_PATH, index=False)
    print(f"\nLogged to {LOG_PATH}")

    resolved = log.dropna(subset=["actual"])
    if len(resolved) >= 3:
        blend_mae = (resolved["actual"] - resolved["blend"]).abs().mean()
        naive_mae = (resolved["actual"] - resolved["naive"]).abs().mean()
        print(f"Track record so far ({len(resolved)} resolved days): "
              f"blend MAE={blend_mae:.4f} vs naive MAE={naive_mae:.4f}")

    print("\nLogging Lighter H100 perp basis (supplementary, non-critical)...")
    try:
        log_lighter_basis(wide)
    except Exception as exc:
        print(f"  Warning: Lighter basis logging failed, skipping (does not affect today's forecast): {exc}")


if __name__ == "__main__":
    main()
