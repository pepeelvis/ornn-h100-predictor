"""Turn the wide daily-index DataFrame into a supervised learning table.

Every row at date T is a snapshot of everything known as of T's close, and
the label is the H100 index value that publishes at T+1. Only past/present
values are used as features (no same-day leakage across GPU series, since
all series appear to publish at the same daily timestamp).
"""
from __future__ import annotations

import pandas as pd

TARGET_COL = "h100_sxm"
TARGET_LAGS = [1, 2, 3, 5, 7]
TARGET_ROLL_WINDOWS = [3, 7, 14]
FEATURE_GPU_LAGS = [1, 2]  # keep cross-GPU features light to avoid overfitting ~90 rows


def build_feature_frame(wide: pd.DataFrame) -> pd.DataFrame:
    df = wide.copy()
    feat = pd.DataFrame(index=df.index)

    for lag in TARGET_LAGS:
        feat[f"{TARGET_COL}_lag{lag}"] = df[TARGET_COL].shift(lag - 1)  # lag1 == today's known value
    for w in TARGET_ROLL_WINDOWS:
        feat[f"{TARGET_COL}_roll_mean{w}"] = df[TARGET_COL].rolling(w).mean()
    feat[f"{TARGET_COL}_roll_std7"] = df[TARGET_COL].rolling(7).std()

    for col in df.columns:
        if col == TARGET_COL:
            continue
        for lag in FEATURE_GPU_LAGS:
            feat[f"{col}_lag{lag}"] = df[col].shift(lag - 1)

    feat[f"{TARGET_COL}_diff1"] = df[TARGET_COL].diff(1)
    feat[f"{TARGET_COL}_diff5"] = df[TARGET_COL].diff(5)

    # Assumes daily-cadence, gap-free dates (true of Ornn's public series) so
    # "target date" = current date + 1 is well-defined even for the live row.
    target_date = pd.to_datetime(df.index.to_series()) + pd.Timedelta(days=1)
    feat["target_dow"] = target_date.dt.dayofweek

    feat["y"] = df[TARGET_COL].shift(-1)
    return feat


def split_train_and_live(feat: pd.DataFrame):
    """Rows with a known y are trainable; the last row (y is NaN) is what we
    predict for tomorrow's publish."""
    usable = feat.dropna(subset=[c for c in feat.columns if c != "y"])
    train = usable.dropna(subset=["y"])
    live = usable[usable["y"].isna()]
    return train, live
