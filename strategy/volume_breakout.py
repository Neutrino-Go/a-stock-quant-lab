from __future__ import annotations

import pandas as pd

from strategy.base import build_signal_frame, latest_signal_rows, non_st_mask, require_columns


def volume_breakout_strategy(
    factor_df: pd.DataFrame,
    signal_date: str | pd.Timestamp | None = None,
    min_volume_ratio: float = 1.8,
) -> pd.DataFrame:
    require_columns(
        factor_df,
        ["code", "name", "date", "close", "breakout_60d", "volume_ratio"],
    )
    prepared = factor_df.copy()
    if "return_1d" not in prepared.columns:
        prepared["date"] = pd.to_datetime(prepared["date"])
        prepared = prepared.sort_values(["code", "date"])
        prepared["return_1d"] = prepared.groupby("code")["close"].pct_change(fill_method=None)

    rows = latest_signal_rows(prepared, signal_date)
    rows = rows[non_st_mask(rows)].copy()
    mask = (
        rows["breakout_60d"].fillna(False).astype(bool)
        & rows["volume_ratio"].gt(min_volume_ratio)
        & rows["return_1d"].between(0.03, 0.08, inclusive="both")
    )
    selected = rows[mask].copy()
    if selected.empty:
        return build_signal_frame(selected, "volume_breakout_strategy", "")

    selected["score"] = (
        70
        + ((selected["volume_ratio"] - min_volume_ratio).clip(0, 2) / 2 * 15)
        + ((selected["return_1d"] - 0.03).clip(0, 0.05) / 0.05 * 10)
    )
    return build_signal_frame(
        selected,
        "volume_breakout_strategy",
        "60d breakout with volume expansion and controlled daily gain",
        stop_loss_pct=0.07,
        take_profit_pct=0.16,
    )
