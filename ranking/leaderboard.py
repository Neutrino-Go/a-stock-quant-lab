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


def rate_strategy(
    net_cagr: float | None = None,
    sharpe_ratio: float = 0.0,
    max_drawdown: float = 0.0,
    cagr: float | None = None,
) -> str:
    score_cagr = float(net_cagr if net_cagr is not None else cagr if cagr is not None else 0.0)
    if score_cagr > 0.15 and sharpe_ratio > 1.5 and max_drawdown > -0.20:
        return "A"
    if score_cagr > 0.08 and sharpe_ratio > 1.0 and max_drawdown > -0.30:
        return "B"
    if score_cagr > 0:
        return "C"
    return "D"


def build_leaderboard_from_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "period",
        "strategy",
        "rating",
        "win_rate",
        "avg_return",
        "gross_annual_return",
        "net_annual_return",
        "cost_impact",
        "max_drawdown",
        "sharpe_ratio",
        "calmar_ratio",
        "profit_loss_ratio",
    ]
    if metrics_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for item in metrics_df.to_dict("records"):
        net_cagr = _first_number(item, "net_CAGR", "net_annual_return", "CAGR", "annual_return")
        gross_annual = _first_number(item, "gross_annual_return", "gross_CAGR", "CAGR", "annual_return")
        max_drawdown = float(item.get("max_drawdown", 0) or 0)
        sharpe_ratio = float(item.get("sharpe_ratio", 0) or 0)
        rows.append(
            {
                "period": item.get("period", ""),
                "strategy": display_strategy_name(str(item["strategy_name"])),
                "rating": rate_strategy(
                    net_cagr=net_cagr,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_drawdown,
                ),
                "win_rate": float(item.get("win_rate", 0) or 0),
                "avg_return": float(item.get("avg_return", 0) or 0),
                "gross_annual_return": gross_annual,
                "net_annual_return": net_cagr,
                "cost_impact": float(item.get("cost_impact", max(0.0, gross_annual - net_cagr)) or 0),
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "calmar_ratio": _calmar(net_cagr, max_drawdown),
                "profit_loss_ratio": float(item.get("profit_loss_ratio", 0) or 0),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["period", "rating", "sharpe_ratio", "net_annual_return"], ascending=[True, True, False, False])
        .reset_index(drop=True)
    )


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
                    "gross_annual_return": metrics["CAGR"],
                    "net_annual_return": metrics["CAGR"],
                    "cost_impact": 0.0,
                    "max_drawdown": metrics["max_drawdown"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "profit_loss_ratio": metrics["profit_loss_ratio"],
                    "total_trades": metrics["total_trades"],
                }
            )
    return build_leaderboard_from_metrics(pd.DataFrame(metrics_rows))


def _first_number(item: dict[str, object], *keys: str) -> float:
    for key in keys:
        value = item.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _calmar(cagr: float, max_drawdown: float) -> float:
    if max_drawdown == 0:
        return 0.0
    return float(cagr / abs(max_drawdown))
