from __future__ import annotations

import pandas as pd

from backtest.backtest_engine import run_strategy_backtest
from ranking.ranking_engine import STRATEGIES
from strategy.metadata import display_strategy_name


DEFAULT_PERIODS = {
    "1个月": 21,
    "3个月": 63,
    "6个月": 126,
    "1年": 252,
}


def rate_strategy(cagr: float, sharpe_ratio: float, max_drawdown: float) -> str:
    if sharpe_ratio > 1.5 and cagr > 0.15 and abs(max_drawdown) < 0.20:
        return "A"
    if sharpe_ratio > 1.0 and cagr > 0.08:
        return "B"
    if cagr > 0:
        return "C"
    return "D"


def build_leaderboard_from_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=["周期", "策略", "评级", "胜率", "平均收益", "年化", "回撤", "夏普", "盈亏比"])

    rows = []
    for item in metrics_df.to_dict("records"):
        rows.append(
            {
                "周期": item.get("period", ""),
                "策略": display_strategy_name(str(item["strategy_name"])),
                "评级": rate_strategy(
                    float(item.get("CAGR", 0)),
                    float(item.get("sharpe_ratio", 0)),
                    float(item.get("max_drawdown", 0)),
                ),
                "胜率": float(item.get("win_rate", 0)),
                "平均收益": float(item.get("avg_return", 0)),
                "年化": float(item.get("CAGR", 0)),
                "回撤": float(item.get("max_drawdown", 0)),
                "夏普": float(item.get("sharpe_ratio", 0)),
                "盈亏比": float(item.get("profit_loss_ratio", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["周期", "评级", "夏普", "年化"], ascending=[True, True, False, False]).reset_index(drop=True)


def build_strategy_leaderboard(
    factor_df: pd.DataFrame,
    periods: dict[str, int] | None = None,
    as_of: str | pd.Timestamp | None = None,
    holding_days: int = 20,
    take_profit_pct: float | None = 0.10,
    stop_loss_pct: float | None = 0.08,
) -> pd.DataFrame:
    if factor_df.empty:
        return build_leaderboard_from_metrics(pd.DataFrame())

    periods = periods or DEFAULT_PERIODS
    data = factor_df.copy()
    data["date"] = pd.to_datetime(data["date"])
    end_date = pd.to_datetime(as_of) if as_of is not None else data["date"].max()

    metrics_rows = []
    for period_name, trading_days in periods.items():
        available_dates = sorted(data.loc[data["date"].le(end_date), "date"].drop_duplicates())
        if not available_dates:
            continue
        start_idx = max(0, len(available_dates) - trading_days)
        start_date = available_dates[start_idx]
        for strategy_name, strategy_func in STRATEGIES.items():
            metrics = run_strategy_backtest(
                factor_df=data,
                strategy_func=strategy_func,
                start_date=start_date,
                end_date=end_date,
                holding_days=holding_days,
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
                sell_on_ma20_break=True,
            )
            metrics_rows.append(
                {
                    "period": period_name,
                    "strategy_name": strategy_name,
                    "win_rate": metrics["win_rate"],
                    "avg_return": metrics["avg_return"],
                    "CAGR": metrics["CAGR"],
                    "max_drawdown": metrics["max_drawdown"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "profit_loss_ratio": metrics["profit_loss_ratio"],
                    "total_trades": metrics["total_trades"],
                }
            )
    return build_leaderboard_from_metrics(pd.DataFrame(metrics_rows))
