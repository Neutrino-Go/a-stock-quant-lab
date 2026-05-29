from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.base import build_signal_frame, latest_signal_rows, non_st_mask, require_columns


def trend_strength_strategy(
    factor_df: pd.DataFrame,
    signal_date: str | pd.Timestamp | None = None,
    min_amount: float = 200_000_000,
    min_return_20d: float = 0.10,
) -> pd.DataFrame:
    require_columns(
        factor_df,
        ["code", "name", "date", "close", "ma20", "ma60", "return_20d", "amount"],
    )
    rows = latest_signal_rows(factor_df, signal_date)
    rows = rows[non_st_mask(rows)].copy()
    mask = (
        rows["close"].gt(rows["ma20"])
        & rows["ma20"].gt(rows["ma60"])
        & rows["return_20d"].gt(min_return_20d)
        & rows["amount"].gt(min_amount)
    )
    selected = rows[mask].copy()
    if selected.empty:
        return build_signal_frame(selected, "trend_strength_strategy", "")

    amount_score = np.log10(selected["amount"].clip(lower=1)) - 8
    selected["score"] = (
        65
        + (selected["return_20d"].clip(0, 0.30) / 0.30 * 20)
        + amount_score.clip(0, 2) * 5
    )
    return build_signal_frame(
        selected,
        "trend_strength_strategy",
        "MA20 above MA60, 20d return above 10%, amount active, non-ST",
        stop_loss_pct=0.08,
        take_profit_pct=0.18,
    )
