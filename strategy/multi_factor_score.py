from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.base import build_signal_frame, latest_signal_rows, non_st_mask, require_columns, risk_level


def multi_factor_score_strategy(
    factor_df: pd.DataFrame,
    signal_date: str | pd.Timestamp | None = None,
    top_n: int = 30,
    min_score: float = 60,
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
            "return_20d",
            "amount",
            "breakout_60d",
            "distance_to_ma20",
            "max_drawdown_20d",
        ],
    )
    rows = latest_signal_rows(factor_df, signal_date)
    rows = rows[non_st_mask(rows)].copy()
    if rows.empty:
        return build_signal_frame(rows, "multi_factor_score_strategy", "")

    trend_score = np.where(rows["close"].gt(rows["ma20"]) & rows["ma20"].gt(rows["ma60"]), 20, 0)
    momentum_score = ((rows["return_20d"].fillna(0) + 0.05).clip(0, 0.25) / 0.25) * 20
    amount_score = np.select(
        [rows["amount"].gt(500_000_000), rows["amount"].gt(200_000_000), rows["amount"].gt(100_000_000)],
        [15, 10, 5],
        default=0,
    )
    breakout_score = np.where(rows["breakout_60d"].fillna(False).astype(bool), 15, 0)
    distance_abs = rows["distance_to_ma20"].abs().fillna(1)
    distance_score = np.select(
        [distance_abs.le(0.03), distance_abs.le(0.08), distance_abs.le(0.15)],
        [15, 10, 5],
        default=0,
    )
    drawdown = rows["max_drawdown_20d"].fillna(0)
    drawdown_score = np.select(
        [drawdown.ge(-0.05), drawdown.ge(-0.10), drawdown.ge(-0.20)],
        [15, 10, 5],
        default=0,
    )

    scored = rows.copy()
    scored["score"] = (
        trend_score
        + momentum_score
        + amount_score
        + breakout_score
        + distance_score
        + drawdown_score
    )
    scored["risk_level"] = risk_level(scored["max_drawdown_20d"], scored["distance_to_ma20"])
    selected = scored[scored["score"].ge(min_score)].sort_values("score", ascending=False).head(top_n)
    if selected.empty:
        return build_signal_frame(selected, "multi_factor_score_strategy", "")

    return build_signal_frame(
        selected,
        "multi_factor_score_strategy",
        "Composite score: trend, momentum, liquidity, breakout, MA20 distance, drawdown risk",
        stop_loss_pct=0.08,
        take_profit_pct=0.18,
    )
