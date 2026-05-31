from __future__ import annotations

import numpy as np
import pandas as pd


def empty_portfolio_metrics(initial_cash: float) -> dict[str, float | int]:
    return {
        "total_return": 0.0,
        "gross_total_return": 0.0,
        "net_total_return": 0.0,
        "annual_return": 0.0,
        "CAGR": 0.0,
        "annual_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "calmar_ratio": 0.0,
        "win_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_loss_ratio": 0.0,
        "expectancy": 0.0,
        "max_consecutive_losses": 0,
        "turnover": 0.0,
        "exposure_ratio": 0.0,
        "avg_holding_days": 0.0,
        "total_trades": 0,
        "final_equity": float(initial_cash),
        "cash_drag": 0.0,
        "cost_impact": 0.0,
        "total_commission": 0.0,
        "total_stamp_tax": 0.0,
        "total_regulatory_fee": 0.0,
        "total_transfer_fee": 0.0,
        "total_slippage_cost": 0.0,
        "total_cost": 0.0,
    }


def calculate_portfolio_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
) -> dict[str, float | int]:
    if equity_curve.empty:
        return empty_portfolio_metrics(initial_cash)

    equity = equity_curve.copy()
    equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
    equity["total_equity"] = pd.to_numeric(equity["total_equity"], errors="coerce")
    if "gross_total_equity" not in equity.columns:
        equity["gross_total_equity"] = equity["total_equity"]
    equity["gross_total_equity"] = pd.to_numeric(equity["gross_total_equity"], errors="coerce")
    equity = equity.dropna(subset=["total_equity"]).sort_values("date").reset_index(drop=True)
    if equity.empty:
        return empty_portfolio_metrics(initial_cash)

    final_equity = float(equity["total_equity"].iloc[-1])
    gross_final_equity = float(equity["gross_total_equity"].iloc[-1])
    net_total_return = final_equity / initial_cash - 1
    gross_total_return = gross_final_equity / initial_cash - 1
    returns = equity["total_equity"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    drawdown = _drawdown(equity["total_equity"])
    cagr = _cagr(equity["date"], final_equity, initial_cash)
    annual_volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    sharpe_ratio = _ratio(returns.mean() * 252, annual_volatility)
    downside = returns[returns.lt(0)]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(252)) if len(downside) > 1 else 0.0
    sortino_ratio = _ratio(returns.mean() * 252, downside_vol)
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    trade_metrics = _trade_metrics(trades)
    turnover = _turnover(trades, equity["total_equity"].mean())
    exposure_ratio = (
        float(pd.to_numeric(equity.get("market_value", 0), errors="coerce").fillna(0).gt(0).mean())
        if "market_value" in equity.columns
        else 0.0
    )
    cash_drag = (
        float((pd.to_numeric(equity.get("cash", 0), errors="coerce").fillna(0) / equity["total_equity"]).mean())
        if "cash" in equity.columns
        else 0.0
    )
    costs = _cost_totals(trades)

    return {
        "total_return": float(net_total_return),
        "gross_total_return": float(gross_total_return),
        "net_total_return": float(net_total_return),
        "annual_return": float(cagr),
        "CAGR": float(cagr),
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": _ratio(cagr, abs(max_drawdown)),
        "turnover": turnover,
        "exposure_ratio": exposure_ratio,
        "final_equity": final_equity,
        "cash_drag": cash_drag,
        "cost_impact": float(gross_total_return - net_total_return),
        **trade_metrics,
        **costs,
    }


def _drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1


def _cagr(dates: pd.Series, final_equity: float, initial_cash: float) -> float:
    if initial_cash <= 0 or final_equity <= 0:
        return 0.0
    clean_dates = dates.dropna()
    if len(clean_dates) < 2:
        return final_equity / initial_cash - 1
    days = max((clean_dates.iloc[-1] - clean_dates.iloc[0]).days, 1)
    return float((final_equity / initial_cash) ** (365.25 / days) - 1)


def _trade_metrics(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty or "net_return" not in trades.columns:
        return {
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_loss_ratio": 0.0,
            "expectancy": 0.0,
            "max_consecutive_losses": 0,
            "avg_holding_days": 0.0,
            "total_trades": 0,
        }
    returns = pd.to_numeric(trades["net_return"], errors="coerce").dropna()
    wins = returns[returns.gt(0)]
    losses = returns[returns.lt(0)]
    win_rate = float(returns.gt(0).mean()) if not returns.empty else 0.0
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(abs(losses.mean())) if not losses.empty else 0.0
    loss_rate = float(returns.lt(0).mean()) if not returns.empty else 0.0
    if avg_loss == 0:
        profit_loss_ratio = float("inf") if avg_win > 0 else 0.0
    else:
        profit_loss_ratio = avg_win / avg_loss
    holding_days = (
        float(pd.to_numeric(trades["holding_days"], errors="coerce").mean())
        if "holding_days" in trades.columns and not trades.empty
        else 0.0
    )
    return {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_loss_ratio": float(profit_loss_ratio),
        "expectancy": float(win_rate * avg_win - loss_rate * avg_loss),
        "max_consecutive_losses": int(_max_consecutive_losses(returns)),
        "avg_holding_days": holding_days if np.isfinite(holding_days) else 0.0,
        "total_trades": int(len(returns)),
    }


def _max_consecutive_losses(returns: pd.Series) -> int:
    current = 0
    maximum = 0
    for value in returns:
        if value < 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _turnover(trades: pd.DataFrame, average_equity: float) -> float:
    if trades.empty or average_equity <= 0:
        return 0.0
    buy = pd.to_numeric(trades.get("gross_buy_amount", 0), errors="coerce").fillna(0).sum()
    sell = pd.to_numeric(trades.get("gross_sell_amount", 0), errors="coerce").fillna(0).sum()
    return float((buy + sell) / average_equity)


def _cost_totals(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "total_commission": 0.0,
            "total_stamp_tax": 0.0,
            "total_regulatory_fee": 0.0,
            "total_transfer_fee": 0.0,
            "total_slippage_cost": 0.0,
            "total_cost": 0.0,
        }
    mapping = {
        "total_commission": "commission",
        "total_stamp_tax": "stamp_tax",
        "total_regulatory_fee": "regulatory_fee",
        "total_transfer_fee": "transfer_fee",
        "total_slippage_cost": "slippage_cost",
        "total_cost": "total_cost",
    }
    return {output: _column_sum(trades, column) for output, column in mapping.items()}


def _column_sum(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator == 0:
        return 0.0
    return float(numerator / denominator)
