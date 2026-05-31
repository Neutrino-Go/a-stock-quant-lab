from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from execution.cost_model import TradingCostConfig, calculate_buy_cost, calculate_sell_cost
from execution.slippage_model import buy_fill_price, sell_fill_price, slippage_cost


def _empty_result() -> dict[str, object]:
    return {
        "total_trades": 0,
        "win_rate": 0,
        "avg_return": 0,
        "median_return": 0,
        "CAGR": 0,
        "cumulative_return": 0,
        "sharpe_ratio": 0,
        "max_return": 0,
        "min_return": 0,
        "max_drawdown": 0,
        "profit_loss_ratio": 0,
        "max_consecutive_losses": 0,
        "expectancy": 0,
        "trade_details": pd.DataFrame(),
        "equity_curve": pd.DataFrame(columns=["period", "date", "equity", "drawdown"]),
        "drawdown_curve": pd.DataFrame(columns=["period", "date", "drawdown"]),
    }


@dataclass
class BacktestEngine:
    holding_days: int = 20
    take_profit_pct: float | None = 0.10
    stop_loss_pct: float | None = 0.08
    sell_on_ma20_break: bool = True
    allow_overlapping_positions: bool = False

    def run(self, price_df: pd.DataFrame, signal_df: pd.DataFrame) -> dict[str, object]:
        if price_df.empty or signal_df.empty:
            return _empty_result()

        prices = price_df.copy()
        prices["code"] = prices["code"].astype(str).str.zfill(6)
        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.sort_values(["code", "date"]).reset_index(drop=True)

        signals = signal_df.copy()
        signals["code"] = signals["code"].astype(str).str.zfill(6)
        signals["signal_date"] = pd.to_datetime(signals["signal_date"])
        signals = signals.sort_values(["signal_date", "code"])

        trades: list[dict[str, object]] = []
        active_until: dict[tuple[str, str], pd.Timestamp] = {}
        for signal in signals.to_dict("records"):
            position_key = (str(signal["code"]), str(signal.get("strategy_name", "")))
            signal_date = pd.to_datetime(signal["signal_date"])
            if (
                not self.allow_overlapping_positions
                and position_key in active_until
                and signal_date < active_until[position_key]
            ):
                continue
            trade = self._simulate_one_signal(prices[prices["code"].eq(signal["code"])], signal)
            if trade:
                trades.append(trade)
                active_until[position_key] = pd.to_datetime(trade["exit_date"])

        if not trades:
            return _empty_result()

        trade_details = pd.DataFrame(trades).sort_values(["buy_date", "code"]).reset_index(drop=True)
        metrics = self._metrics(trade_details)
        metrics["trade_details"] = trade_details
        return metrics

    def _simulate_one_signal(
        self,
        code_prices: pd.DataFrame,
        signal: dict[str, object],
    ) -> dict[str, object] | None:
        if code_prices.empty:
            return None

        code_prices = code_prices.reset_index(drop=True)
        signal_date = pd.to_datetime(signal["signal_date"])
        signal_positions = code_prices.index[code_prices["date"].eq(signal_date)].to_list()
        if not signal_positions:
            return None

        signal_idx = signal_positions[0]
        buy_idx = signal_idx + 1
        if buy_idx >= len(code_prices):
            return None

        buy_row = code_prices.iloc[buy_idx]
        buy_price = float(buy_row["open"])
        if not np.isfinite(buy_price) or buy_price <= 0:
            return None

        max_exit_idx = min(buy_idx + max(self.holding_days, 1) - 1, len(code_prices) - 1)
        exit_idx = max_exit_idx
        exit_price = float(code_prices.iloc[max_exit_idx]["close"])
        exit_date = code_prices.iloc[max_exit_idx]["date"]
        exit_reason = "holding_period"

        stop_price = buy_price * (1 - self.stop_loss_pct) if self.stop_loss_pct is not None else None
        take_price = buy_price * (1 + self.take_profit_pct) if self.take_profit_pct is not None else None

        for idx in range(buy_idx, max_exit_idx + 1):
            row = code_prices.iloc[idx]
            if stop_price is not None and float(row["low"]) <= stop_price:
                exit_idx = idx
                exit_price = stop_price
                exit_date = row["date"]
                exit_reason = "stop_loss"
                break
            if take_price is not None and float(row["high"]) >= take_price:
                exit_idx = idx
                exit_price = take_price
                exit_date = row["date"]
                exit_reason = "take_profit"
                break
            if self.sell_on_ma20_break and "ma20" in row.index and pd.notna(row["ma20"]):
                if float(row["close"]) < float(row["ma20"]):
                    exit_idx = idx
                    exit_price = float(row["close"])
                    exit_date = row["date"]
                    exit_reason = "ma20_break"
                    break

        trade_return = exit_price / buy_price - 1
        return {
            "code": signal["code"],
            "name": signal.get("name", ""),
            "strategy_name": signal.get("strategy_name", ""),
            "signal_date": signal_date,
            "buy_date": buy_row["date"],
            "buy_price": buy_price,
            "exit_date": exit_date,
            "exit_price": float(exit_price),
            "exit_reason": exit_reason,
            "holding_days": int(exit_idx - buy_idx + 1),
            "return": float(trade_return),
            "score": signal.get("score", np.nan),
        }

    @staticmethod
    def _metrics(trades: pd.DataFrame) -> dict[str, object]:
        return calculate_trade_metrics(trades)


def calculate_trade_metrics(trades: pd.DataFrame) -> dict[str, object]:
    if trades.empty or "return" not in trades.columns:
        return _empty_result()

    clean = trades.copy()
    clean["return"] = pd.to_numeric(clean["return"], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=["return"])
    if clean.empty:
        return _empty_result()

    returns = clean["return"].astype(float).reset_index(drop=True)
    wins = returns[returns.gt(0)]
    losses = returns[returns.lt(0)]
    equity_values = pd.concat(
        [pd.Series([1.0]), (1 + returns).cumprod()],
        ignore_index=True,
    )
    running_max = equity_values.cummax()
    drawdown_values = equity_values / running_max - 1
    curve_dates = _curve_dates(clean, len(equity_values))
    equity_curve = pd.DataFrame(
        {
            "period": range(len(equity_values)),
            "date": curve_dates,
            "equity": equity_values,
            "drawdown": drawdown_values,
        }
    )
    drawdown_curve = equity_curve.loc[:, ["period", "date", "drawdown"]].copy()
    cumulative_return = float(equity_values.iloc[-1] - 1)
    cagr = _cagr(clean, float(equity_values.iloc[-1]))
    sharpe_ratio = _sharpe_ratio(clean, returns)
    win_rate = float(returns.gt(0).mean())
    loss_rate = float(returns.lt(0).mean())
    avg_win = float(wins.mean()) if not wins.empty else 0
    avg_loss = float(abs(losses.mean())) if not losses.empty else 0

    if losses.empty:
        profit_loss_ratio = float("inf") if not wins.empty else 0
    elif wins.empty:
        profit_loss_ratio = 0
    else:
        profit_loss_ratio = float(avg_win / avg_loss) if avg_loss > 0 else float("inf")

    return {
        "total_trades": int(len(clean)),
        "win_rate": win_rate,
        "avg_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "CAGR": cagr,
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe_ratio,
        "max_return": float(returns.max()),
        "min_return": float(returns.min()),
        "max_drawdown": float(drawdown_values.min()),
        "profit_loss_ratio": profit_loss_ratio,
        "max_consecutive_losses": int(_max_consecutive_losses(returns)),
        "expectancy": float(win_rate * avg_win - loss_rate * avg_loss),
        "trade_details": clean,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
    }


def _curve_dates(trades: pd.DataFrame, length: int) -> pd.Series:
    if "exit_date" not in trades.columns:
        return pd.Series(range(length))
    exit_dates = pd.to_datetime(trades["exit_date"], errors="coerce").reset_index(drop=True)
    first_date = exit_dates.dropna().iloc[0] if not exit_dates.dropna().empty else pd.NaT
    return pd.concat([pd.Series([first_date]), exit_dates], ignore_index=True)


def _cagr(trades: pd.DataFrame, final_equity: float) -> float:
    if final_equity <= 0:
        return -1.0
    if {"buy_date", "exit_date"}.issubset(trades.columns):
        start = pd.to_datetime(trades["buy_date"], errors="coerce").min()
        end = pd.to_datetime(trades["exit_date"], errors="coerce").max()
        if pd.notna(start) and pd.notna(end):
            days = max((end - start).days, 1)
            return float(final_equity ** (365.25 / days) - 1)
    return float(final_equity - 1)


def _sharpe_ratio(trades: pd.DataFrame, returns: pd.Series) -> float:
    std = returns.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return 0
    if {"buy_date", "exit_date"}.issubset(trades.columns):
        start = pd.to_datetime(trades["buy_date"], errors="coerce").min()
        end = pd.to_datetime(trades["exit_date"], errors="coerce").max()
        if pd.notna(start) and pd.notna(end):
            years = max((end - start).days / 365.25, 1 / 365.25)
            trades_per_year = len(returns) / years
            return float(returns.mean() / std * np.sqrt(trades_per_year))
    return float(returns.mean() / std * np.sqrt(len(returns)))


def _max_consecutive_losses(returns: pd.Series) -> int:
    max_count = 0
    current = 0
    for value in returns:
        if value < 0:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count


def backtest_signals(
    price_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    holding_days: int = 20,
    take_profit_pct: float | None = 0.10,
    stop_loss_pct: float | None = 0.08,
    sell_on_ma20_break: bool = True,
    allow_overlapping_positions: bool = False,
) -> dict[str, object]:
    return BacktestEngine(
        holding_days=holding_days,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        sell_on_ma20_break=sell_on_ma20_break,
        allow_overlapping_positions=allow_overlapping_positions,
    ).run(price_df, signal_df)


def run_strategy_backtest(
    factor_df: pd.DataFrame,
    strategy_func: Callable[..., pd.DataFrame],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    strategy_params: dict[str, object] | None = None,
    holding_days: int = 20,
    take_profit_pct: float | None = 0.10,
    stop_loss_pct: float | None = 0.08,
    sell_on_ma20_break: bool = True,
    allow_overlapping_positions: bool = False,
) -> dict[str, object]:
    if factor_df.empty:
        return _empty_result()

    strategy_params = strategy_params or {}
    prices = factor_df.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    signal_dates = sorted(prices.loc[prices["date"].between(start_ts, end_ts), "date"].unique())

    signals = []
    for signal_date in signal_dates:
        daily_signals = strategy_func(prices, signal_date=signal_date, **strategy_params)
        if not daily_signals.empty:
            signals.append(daily_signals)

    if not signals:
        return _empty_result()

    signal_df = pd.concat(signals, ignore_index=True).drop_duplicates(
        ["code", "signal_date", "strategy_name"]
    )
    return backtest_signals(
        prices,
        signal_df,
        holding_days=holding_days,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        sell_on_ma20_break=sell_on_ma20_break,
        allow_overlapping_positions=allow_overlapping_positions,
    )


def example_backtest(factor_df: pd.DataFrame) -> dict[str, object]:
    from strategy.trend_strength import trend_strength_strategy

    dates = pd.to_datetime(factor_df["date"])
    return run_strategy_backtest(
        factor_df=factor_df,
        strategy_func=trend_strength_strategy,
        start_date=dates.min(),
        end_date=dates.max(),
        holding_days=20,
        take_profit_pct=0.15,
        stop_loss_pct=0.08,
        sell_on_ma20_break=True,
    )


def apply_realistic_costs_to_trades(
    trade_details: pd.DataFrame,
    cost_config: TradingCostConfig | None = None,
    shares: int = 100,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if trade_details.empty:
        return trade_details.copy(), calculate_trade_metrics(trade_details)

    cost_config = cost_config or TradingCostConfig()
    rows = []
    for trade in trade_details.to_dict("records"):
        buy_price = float(trade["buy_price"])
        sell_price = float(trade["exit_price"])
        buy_fill = buy_fill_price(buy_price, cost_config)
        sell_fill = sell_fill_price(sell_price, cost_config)
        buy_cost = calculate_buy_cost(buy_fill, shares, cost_config)
        sell_cost = calculate_sell_cost(sell_fill, shares, cost_config)
        gross_buy = buy_price * shares
        gross_sell = sell_price * shares
        net_buy = buy_fill * shares + buy_cost["total_cost"]
        net_sell = sell_fill * shares - sell_cost["total_cost"]
        total_cost = buy_cost["total_cost"] + sell_cost["total_cost"]
        total_slippage = slippage_cost(buy_price, buy_fill, shares) + slippage_cost(
            sell_price,
            sell_fill,
            shares,
        )
        enriched = dict(trade)
        enriched.update(
            {
                "gross_return": gross_sell / gross_buy - 1 if gross_buy > 0 else 0.0,
                "net_return": net_sell / net_buy - 1 if net_buy > 0 else 0.0,
                "return": net_sell / net_buy - 1 if net_buy > 0 else 0.0,
                "buy_fill_price": buy_fill,
                "sell_fill_price": sell_fill,
                "gross_buy_amount": gross_buy,
                "gross_sell_amount": gross_sell,
                "net_buy_cash": net_buy,
                "net_sell_cash": net_sell,
                "commission": buy_cost["commission"] + sell_cost["commission"],
                "stamp_tax": sell_cost["stamp_tax"],
                "regulatory_fee": buy_cost["regulatory_fee"] + sell_cost["regulatory_fee"],
                "transfer_fee": buy_cost["transfer_fee"] + sell_cost["transfer_fee"],
                "total_cost": total_cost,
                "slippage_cost": total_slippage,
            }
        )
        rows.append(enriched)

    adjusted = pd.DataFrame(rows)
    metrics = calculate_trade_metrics(adjusted)
    metrics["gross_return"] = float((1 + trade_details["return"]).prod() - 1)
    metrics["net_return"] = metrics["cumulative_return"]
    metrics["cost_impact"] = float(metrics["gross_return"] - metrics["net_return"])
    metrics["total_commission"] = float(adjusted["commission"].sum())
    metrics["total_stamp_tax"] = float(adjusted["stamp_tax"].sum())
    metrics["total_slippage_cost"] = float(adjusted["slippage_cost"].sum())
    metrics["trade_details"] = adjusted
    return adjusted, metrics
