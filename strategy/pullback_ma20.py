from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.base import build_signal_frame, latest_signal_rows, non_st_mask, require_columns


def pullback_ma20_strategy(
    factor_df: pd.DataFrame,
    signal_date: str | pd.Timestamp | None = None,
    max_distance: float = 0.03,
) -> pd.DataFrame:
    require_columns(
        factor_df,
        [
            "code",
            "name",
            "date",
            "close",
            "ma20",
            "ma60",
            "return_5d",
            "volume_ratio",
            "distance_to_ma20",
        ],
    )
    rows = latest_signal_rows(factor_df, signal_date)
    rows = rows[non_st_mask(rows)].copy()
    mask = (
        rows["ma20"].gt(rows["ma60"])
        & rows["distance_to_ma20"].abs().le(max_distance)
        & rows["return_5d"].between(-0.08, 0.03, inclusive="both")
        & rows["volume_ratio"].lt(1.2)
    )
    selected = rows[mask].copy()
    if selected.empty:
        return build_signal_frame(selected, "pullback_ma20_strategy", "")

    distance_score = (1 - selected["distance_to_ma20"].abs() / max_distance).clip(0, 1) * 20
    volume_score = (1.2 - selected["volume_ratio"]).clip(0, 1.2) / 1.2 * 10
    selected["score"] = 65 + distance_score + volume_score
    selected["risk_level"] = np.where(selected["return_5d"].lt(-0.05), "medium", "low")
    return build_signal_frame(
        selected,
        "pullback_ma20_strategy",
        "MA20 above MA60, price near MA20, mild 5d pullback, volume not overheated",
        stop_loss_pct=0.06,
        take_profit_pct=0.12,
    )
