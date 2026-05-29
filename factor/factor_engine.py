from __future__ import annotations

import numpy as np
import pandas as pd


def compute_factors(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return daily_df.copy()

    required = {"code", "name", "date", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(daily_df.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")

    df = daily_df.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    if "turnover" in df.columns:
        numeric_columns.append("turnover")
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    grouped = df.groupby("code", group_keys=False)
    close = grouped["close"]
    volume = grouped["volume"]

    for window in (5, 10, 20, 60):
        df[f"ma{window}"] = close.transform(
            lambda series, w=window: series.rolling(w, min_periods=1).mean()
        )

    df["return_1d"] = close.pct_change(fill_method=None)
    for window in (5, 10, 20):
        df[f"return_{window}d"] = close.pct_change(window, fill_method=None)

    df["volume_ma20"] = volume.transform(lambda series: series.rolling(20, min_periods=1).mean())
    df["volume_ratio"] = np.where(
        df["volume_ma20"].gt(0),
        df["volume"] / df["volume_ma20"],
        np.nan,
    )

    return_1d = df.groupby("code", group_keys=False)["return_1d"]
    df["volatility_20d"] = return_1d.transform(
        lambda series: series.rolling(20, min_periods=2).std()
    )
    rolling_high_20 = close.transform(lambda series: series.rolling(20, min_periods=1).max())
    rolling_high_60 = close.transform(lambda series: series.rolling(60, min_periods=1).max())
    df["breakout_20d"] = df["close"].ge(rolling_high_20)
    df["breakout_60d"] = df["close"].ge(rolling_high_60)
    df["ma_bullish"] = df["close"].gt(df["ma20"]) & df["ma20"].gt(df["ma60"])
    df["distance_to_ma20"] = np.where(df["ma20"].gt(0), df["close"] / df["ma20"] - 1, np.nan)
    df["max_drawdown_20d"] = close.transform(
        lambda series: series.rolling(20, min_periods=1).apply(_window_max_drawdown, raw=True)
    )
    return df


def _window_max_drawdown(values: np.ndarray) -> float:
    if len(values) == 0 or np.isnan(values).all():
        return np.nan
    clean = values.astype(float)
    peaks = np.maximum.accumulate(clean)
    drawdowns = np.where(peaks > 0, clean / peaks - 1, np.nan)
    return float(np.nanmin(drawdowns))
