from __future__ import annotations

from itertools import product
from typing import Any

import pandas as pd

from backtest.backtest_engine import run_strategy_backtest
from ranking.ranking_engine import STRATEGIES


DEFAULT_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "trend_strength_strategy": {
        "min_amount": [100_000_000, 200_000_000, 300_000_000, 500_000_000],
        "min_return_20d": [0.05, 0.10, 0.15, 0.20],
    },
    "pullback_ma20_strategy": {
        "max_distance": [0.01, 0.02, 0.03, 0.05],
    },
    "volume_breakout_strategy": {
        "min_volume_ratio": [1.5, 1.8, 2.0, 2.5],
    },
}


def optimize_strategy(
    factor_df: pd.DataFrame,
    strategy_name: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    param_grid: dict[str, list[Any]] | None = None,
    holding_days: int = 20,
    take_profit_pct: float | None = 0.10,
    stop_loss_pct: float | None = 0.08,
    sell_on_ma20_break: bool = True,
    top_n: int = 20,
) -> pd.DataFrame:
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    grid = param_grid or DEFAULT_PARAM_GRIDS.get(strategy_name, {})
    if not grid:
        raise ValueError(f"No parameter grid configured for {strategy_name}")

    rows = []
    for params in _iter_param_grid(grid):
        metrics = run_strategy_backtest(
            factor_df=factor_df,
            strategy_func=STRATEGIES[strategy_name],
            start_date=start_date,
            end_date=end_date,
            strategy_params=params,
            holding_days=holding_days,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            sell_on_ma20_break=sell_on_ma20_break,
        )
        rows.append(
            {
                "strategy_name": strategy_name,
                "parameters": params,
                "total_trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "annual_return": metrics["CAGR"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "avg_return": metrics["avg_return"],
                "profit_loss_ratio": metrics["profit_loss_ratio"],
            }
        )

    result = pd.DataFrame(rows)
    return (
        result.sort_values(
            ["sharpe_ratio", "annual_return", "win_rate", "max_drawdown"],
            ascending=[False, False, False, False],
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def _iter_param_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(param_grid)
    return [dict(zip(keys, values)) for values in product(*(param_grid[key] for key in keys))]
