"""Forecasters for tomorrow's Ornn H100 index publish, plus a walk-forward
backtest that scores each one and derives ensemble weights + an error band.

Three independent forecasters, blended:
  - naive:  tomorrow = today
  - holt:   Holt's linear-trend exponential smoothing on the H100 series
  - ridge:  regularized linear regression on lag/rolling/cross-GPU features

With ~90 days of public history, this stays deliberately simple: enough
signal to beat a naive carry-forward, not so much model as to overfit noise.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import Holt

from features import TARGET_COL, build_feature_frame, split_train_and_live

# Lower alphas are numerically unstable on tiny CV folds with collinear
# lag/rolling features (near-singular X^T X) -- 1.0 is the effective floor.
RIDGE_ALPHAS = [1.0, 3.0, 10.0, 30.0, 100.0]


def naive_forecast(history: pd.Series) -> float:
    return float(history.iloc[-1])


def holt_forecast(history: pd.Series) -> float:
    if len(history) < 10:
        return naive_forecast(history)
    fit = Holt(history.values, damped_trend=True, initialization_method="estimated").fit(optimized=True)
    return float(fit.forecast(1)[0])


def _select_ridge_alpha(X: np.ndarray, y: np.ndarray) -> float:
    n_splits = min(5, max(2, len(X) // 10))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_alpha, best_mae = RIDGE_ALPHAS[0], np.inf
    with warnings.catch_warnings():
        # Earliest TimeSeriesSplit folds have very few training rows, which
        # can make X^T X near-singular for a given fold/alpha combo. Those
        # folds just score badly and lose the alpha selection below.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for alpha in RIDGE_ALPHAS:
            errs = []
            for tr_idx, te_idx in tscv.split(X):
                scaler = StandardScaler().fit(X[tr_idx])
                model = Ridge(alpha=alpha).fit(scaler.transform(X[tr_idx]), y[tr_idx])
                pred = model.predict(scaler.transform(X[te_idx]))
                errs.append(np.mean(np.abs(pred - y[te_idx])))
            mae = float(np.mean(errs))
            if mae < best_mae:
                best_mae, best_alpha = mae, alpha
    return best_alpha


def ridge_forecast(train: pd.DataFrame, live_row: pd.DataFrame) -> float:
    feature_cols = [c for c in train.columns if c != "y"]
    X, y = train[feature_cols].values, train["y"].values
    if len(X) < 15:
        return naive_forecast(pd.Series(train[f"{TARGET_COL}_lag1"]))
    alpha = _select_ridge_alpha(X, y)
    with warnings.catch_warnings():
        # Same benign BLAS warning as in _select_ridge_alpha (numpy @ on
        # Apple's Accelerate backend), not a real numerical issue here.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        scaler = StandardScaler().fit(X)
        model = Ridge(alpha=alpha).fit(scaler.transform(X), y)
        x_live = live_row[feature_cols].values
        return float(model.predict(scaler.transform(x_live))[0])


def walk_forward_backtest(wide: pd.DataFrame, n_test: int = 25) -> dict:
    """Re-fit each forecaster at every step using only data available up to
    that point, predict the next day, and compare against what actually
    published. Returns per-method errors, inverse-MAE ensemble weights, and
    the residual std of the blended forecast (used for the +/- band).
    """
    dates = wide.index
    n_test = min(n_test, len(dates) - 20)
    if n_test < 5:
        raise ValueError("Not enough history for a meaningful backtest")

    results = {"naive": [], "holt": [], "ridge": [], "blend": [], "actual": [], "date": []}

    for i in range(len(dates) - n_test, len(dates) - 1):
        train_wide = wide.iloc[: i + 1]
        actual_next = float(wide[TARGET_COL].iloc[i + 1])
        history = train_wide[TARGET_COL]

        feat = build_feature_frame(train_wide)
        train, live = split_train_and_live(feat)
        if live.empty:
            continue

        p_naive = naive_forecast(history)
        p_holt = holt_forecast(history)
        p_ridge = ridge_forecast(train, live) if len(train) >= 15 else p_naive
        p_blend = float(np.mean([p_holt, p_ridge]))

        results["naive"].append(p_naive)
        results["holt"].append(p_holt)
        results["ridge"].append(p_ridge)
        results["blend"].append(p_blend)
        results["actual"].append(actual_next)
        results["date"].append(dates[i + 1])

    df = pd.DataFrame(results).set_index("date")
    mae = {m: float(np.mean(np.abs(df[m] - df["actual"]))) for m in ["naive", "holt", "ridge", "blend"]}
    resid_std = float((df["blend"] - df["actual"]).std())
    inv = {m: 1.0 / max(mae[m], 1e-6) for m in ["holt", "ridge"]}
    total = sum(inv.values())
    weights = {m: v / total for m, v in inv.items()}
    return {"detail": df, "mae": mae, "weights": weights, "resid_std": resid_std}


def predict_next(wide: pd.DataFrame, weights: dict | None = None) -> dict:
    """Point forecast (weighted holt/ridge blend) + naive baseline for
    tomorrow's publish, using all available history."""
    weights = weights or {"holt": 0.5, "ridge": 0.5}
    history = wide[TARGET_COL]

    feat = build_feature_frame(wide)
    train, live = split_train_and_live(feat)
    if live.empty:
        raise RuntimeError("No live row to predict — check that wide's last date has no y yet")

    p_naive = naive_forecast(history)
    p_holt = holt_forecast(history)
    p_ridge = ridge_forecast(train, live)
    p_blend = weights["holt"] * p_holt + weights["ridge"] * p_ridge

    target_date = pd.Timestamp(wide.index[-1]) + pd.Timedelta(days=1)
    return {
        "target_date": target_date,
        "naive": p_naive,
        "holt": p_holt,
        "ridge": p_ridge,
        "blend": p_blend,
        "last_known_date": wide.index[-1],
        "last_known_value": p_naive,
    }
