from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from strategy.base import SIGNAL_COLUMNS, empty_signal_frame
from strategy.multi_factor_score import multi_factor_score_strategy
from strategy.pullback_ma20 import pullback_ma20_strategy
from strategy.trend_strength import trend_strength_strategy
from strategy.volume_breakout import volume_breakout_strategy


StrategyFunc = Callable[..., pd.DataFrame]

STRATEGIES: dict[str, StrategyFunc] = {
    "trend_strength_strategy": trend_strength_strategy,
    "pullback_ma20_strategy": pullback_ma20_strategy,
    "volume_breakout_strategy": volume_breakout_strategy,
    "multi_factor_score_strategy": multi_factor_score_strategy,
}


def run_strategy(
    strategy_name: str,
    factor_df: pd.DataFrame,
    signal_date: str | pd.Timestamp | None = None,
    **params,
) -> pd.DataFrame:
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return STRATEGIES[strategy_name](factor_df, signal_date=signal_date, **params)


def run_all_strategies(
    factor_df: pd.DataFrame,
    signal_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    outputs = [strategy(factor_df, signal_date=signal_date) for strategy in STRATEGIES.values()]
    non_empty = [frame for frame in outputs if not frame.empty]
    if not non_empty:
        return empty_signal_frame()
    return pd.concat(non_empty, ignore_index=True).loc[:, SIGNAL_COLUMNS]


def rank_recommendations(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()

    grouped = (
        signals.groupby(["code", "name"], as_index=False)
        .agg(
            signal_date=("signal_date", "max"),
            strategy_name=("strategy_name", lambda values: ", ".join(sorted(set(values)))),
            score=("score", "sum"),
            signal_reason=("signal_reason", lambda values: " | ".join(sorted(set(values)))),
            suggested_buy_price=("suggested_buy_price", "mean"),
            suggested_stop_loss=("suggested_stop_loss", "min"),
            suggested_take_profit=("suggested_take_profit", "max"),
            risk_level=("risk_level", _combine_risk),
        )
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def _combine_risk(values: pd.Series) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    reverse = {value: key for key, value in order.items()}
    highest = max(order.get(str(value), 1) for value in values)
    return reverse[highest]
