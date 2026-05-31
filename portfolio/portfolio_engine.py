from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from execution.cost_model import TradingCostConfig
from execution.order_model import (
    Order,
    OrderStatus,
    create_buy_order,
    create_sell_order,
    is_limit_down,
    is_limit_up,
    skipped_order,
)
from portfolio.portfolio_metrics import calculate_portfolio_metrics
from portfolio.position_sizer import PositionSizingConfig, calculate_position_size


@dataclass(frozen=True)
class ExitConfig:
    holding_days: int = 20
    take_profit_pct: float | None = 0.10
    stop_loss_pct: float | None = 0.08
    sell_on_ma20_break: bool = True
    allow_add_position: bool = False
    enable_limit_check: bool = True
    limit_pct: float = 0.10
    daily_top_n: int | None = None
    deduplicate_signals: bool = True


class PortfolioBacktestEngine:
    def __init__(
        self,
        cost_config: TradingCostConfig | None = None,
        position_config: PositionSizingConfig | None = None,
        exit_config: ExitConfig | None = None,
    ) -> None:
        self.cost_config = cost_config or TradingCostConfig()
        self.position_config = position_config or PositionSizingConfig()
        self.exit_config = exit_config or ExitConfig()
        self._reject_add_position_mode()

    def run(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> dict[str, object]:
        self._reject_add_position_mode()
        prices = self._prepare_prices(price_data)
        if prices.empty:
            return self._empty_result(pd.DataFrame())

        start_ts = pd.to_datetime(start_date) if start_date is not None else prices["date"].min()
        end_ts = pd.to_datetime(end_date) if end_date is not None else prices["date"].max()
        prices = prices[prices["date"].between(start_ts, end_ts)].copy()
        calendar = list(prices["date"].drop_duplicates().sort_values())
        if not calendar:
            return self._empty_result(pd.DataFrame())

        signals_by_order_date = self._schedule_signals(signals, calendar)
        price_lookup = {(row.code, row.date): row for row in prices.itertuples(index=False)}
        code_prices = {code: frame.reset_index(drop=True) for code, frame in prices.groupby("code")}

        cash = float(self.position_config.initial_cash)
        cumulative_costs = 0.0
        positions: dict[str, dict[str, Any]] = {}
        orders: list[dict[str, object]] = []
        trades: list[dict[str, object]] = []
        equity_rows: list[dict[str, object]] = []
        position_rows: list[dict[str, object]] = []

        for date_idx, current_date in enumerate(calendar):
            cash, cumulative_costs = self._process_exits(
                current_date=current_date,
                date_idx=date_idx,
                calendar=calendar,
                positions=positions,
                code_prices=code_prices,
                orders=orders,
                trades=trades,
                cash=cash,
                cumulative_costs=cumulative_costs,
            )
            cash, cumulative_costs = self._process_buys(
                current_date=current_date,
                date_idx=date_idx,
                signals=signals_by_order_date.get(current_date, pd.DataFrame()),
                price_lookup=price_lookup,
                orders=orders,
                positions=positions,
                cash=cash,
                cumulative_costs=cumulative_costs,
            )
            market_value = self._market_value(positions, price_lookup, current_date)
            total_equity = cash + market_value
            equity_rows.append(
                {
                    "date": current_date,
                    "cash": cash,
                    "market_value": market_value,
                    "total_equity": total_equity,
                    "gross_total_equity": total_equity + cumulative_costs,
                    "positions_count": len(positions),
                }
            )
            position_rows.extend(self._snapshot_positions(positions, price_lookup, current_date))

        equity_curve = pd.DataFrame(equity_rows)
        if not equity_curve.empty:
            running_max = equity_curve["total_equity"].cummax()
            equity_curve["drawdown"] = equity_curve["total_equity"] / running_max - 1
        trades_df = pd.DataFrame(trades)
        orders_df = pd.DataFrame(orders)
        positions_df = pd.DataFrame(position_rows)
        metrics = calculate_portfolio_metrics(
            equity_curve,
            trades_df,
            initial_cash=self.position_config.initial_cash,
        )
        return {
            "portfolio_equity_curve": equity_curve,
            "portfolio_drawdown_curve": equity_curve.loc[:, ["date", "drawdown"]].copy()
            if "drawdown" in equity_curve.columns
            else pd.DataFrame(columns=["date", "drawdown"]),
            "trades": trades_df,
            "orders": orders_df,
            "positions_snapshot": positions_df,
            "metrics": metrics,
        }

    def _reject_add_position_mode(self) -> None:
        if self.exit_config.allow_add_position:
            raise NotImplementedError(
                "allow_add_position=True is not supported in V3; "
                "lot-level position accounting is planned for V4."
            )

    def _process_exits(
        self,
        current_date: pd.Timestamp,
        date_idx: int,
        calendar: list[pd.Timestamp],
        positions: dict[str, dict[str, Any]],
        code_prices: dict[str, pd.DataFrame],
        orders: list[dict[str, object]],
        trades: list[dict[str, object]],
        cash: float,
        cumulative_costs: float,
    ) -> tuple[float, float]:
        for code, position in list(positions.items()):
            if current_date <= position["buy_date"]:
                continue
            row = self._price_row_for_code_date(code_prices.get(code), current_date)
            if row is None:
                continue
            exit_decision = self._exit_decision(position, row, date_idx, code_prices.get(code))
            if exit_decision is None:
                continue

            prev_close = self._previous_close(code_prices.get(code), current_date)
            if (
                self.exit_config.enable_limit_check
                and prev_close is not None
                and is_limit_down(float(row["open"]), prev_close, self.exit_config.limit_pct)
            ):
                order = skipped_order(
                    code=code,
                    name=position["name"],
                    side="SELL",
                    signal_date=position["signal_date"],
                    order_date=current_date,
                    requested_price=float(row["open"]),
                    status=OrderStatus.SKIPPED_LIMIT_DOWN,
                    reason="limit-down approximation prevented sell",
                )
                orders.append(order.to_dict())
                continue

            sell_order = create_sell_order(
                code=code,
                name=position["name"],
                signal_date=position["signal_date"],
                order_date=current_date,
                requested_price=exit_decision["price"],
                shares=position["shares"],
                config=self.cost_config,
                reason=exit_decision["reason"],
            )
            cash += sell_order.net_amount
            cumulative_costs += sell_order.cost["total_cost"] + sell_order.slippage_cost
            orders.append(sell_order.to_dict())
            trades.append(self._trade_row(position, sell_order, date_idx, exit_decision["reason"]))
            del positions[code]
        return cash, cumulative_costs

    def _process_buys(
        self,
        current_date: pd.Timestamp,
        date_idx: int,
        signals: pd.DataFrame,
        price_lookup: dict[tuple[str, pd.Timestamp], Any],
        orders: list[dict[str, object]],
        positions: dict[str, dict[str, Any]],
        cash: float,
        cumulative_costs: float,
    ) -> tuple[float, float]:
        if signals.empty:
            return cash, cumulative_costs
        for signal in signals.to_dict("records"):
            code = str(signal["code"]).zfill(6)
            if len(positions) >= self.position_config.max_positions:
                orders.append(
                    skipped_order(
                        code,
                        str(signal.get("name", "")),
                        "BUY",
                        pd.to_datetime(signal["signal_date"]),
                        current_date,
                        0.0,
                        OrderStatus.SKIPPED_NO_CASH,
                        "maximum positions reached",
                    ).to_dict()
                )
                continue
            if not self.exit_config.allow_add_position and code in positions:
                orders.append(
                    skipped_order(
                        code,
                        str(signal.get("name", "")),
                        "BUY",
                        pd.to_datetime(signal["signal_date"]),
                        current_date,
                        0.0,
                        OrderStatus.SKIPPED_DUPLICATE_POSITION,
                        "position already exists for this code",
                    ).to_dict()
                )
                continue
            price_row = price_lookup.get((code, current_date))
            if price_row is None or not np.isfinite(float(price_row.open)):
                orders.append(
                    skipped_order(
                        code,
                        str(signal.get("name", "")),
                        "BUY",
                        pd.to_datetime(signal["signal_date"]),
                        current_date,
                        0.0,
                        OrderStatus.SKIPPED_NO_PRICE_DATA,
                        "no open price data on order date",
                    ).to_dict()
                )
                continue
            prev_close = self._previous_close_from_lookup(price_lookup, code, current_date)
            if (
                self.exit_config.enable_limit_check
                and prev_close is not None
                and is_limit_up(float(price_row.open), prev_close, self.exit_config.limit_pct)
            ):
                orders.append(
                    skipped_order(
                        code,
                        str(signal.get("name", "")),
                        "BUY",
                        pd.to_datetime(signal["signal_date"]),
                        current_date,
                        float(price_row.open),
                        OrderStatus.SKIPPED_LIMIT_UP,
                        "limit-up approximation prevented buy",
                    ).to_dict()
                )
                continue
            stop_loss_price = float(signal.get("suggested_stop_loss", price_row.open * 0.92))
            size = calculate_position_size(
                cash=cash,
                price=float(price_row.open),
                stop_loss_price=stop_loss_price,
                config=self.position_config,
                cost_config=self.cost_config,
                current_positions=len(positions),
            )
            if size.shares < 100:
                orders.append(
                    skipped_order(
                        code,
                        str(signal.get("name", "")),
                        "BUY",
                        pd.to_datetime(signal["signal_date"]),
                        current_date,
                        float(price_row.open),
                        OrderStatus.SKIPPED_NO_CASH,
                        size.reason,
                    ).to_dict()
                )
                continue
            buy_order = create_buy_order(
                code=code,
                name=str(signal.get("name", "")),
                signal_date=pd.to_datetime(signal["signal_date"]),
                order_date=current_date,
                requested_price=float(price_row.open),
                requested_shares=size.shares,
                cash_available=cash,
                config=self.cost_config,
                reason=str(signal.get("signal_reason", "")),
            )
            orders.append(buy_order.to_dict())
            if buy_order.status != OrderStatus.FILLED:
                continue
            cash -= buy_order.net_amount
            cumulative_costs += buy_order.cost["total_cost"] + buy_order.slippage_cost
            positions[code] = {
                "code": code,
                "name": str(signal.get("name", "")),
                "strategy_name": str(signal.get("strategy_name", "")),
                "triggered_strategies": str(signal.get("triggered_strategies", "")),
                "signal_date": pd.to_datetime(signal["signal_date"]),
                "buy_date": current_date,
                "buy_index": date_idx,
                "shares": buy_order.shares,
                "buy_order": buy_order,
                "score": float(signal.get("score", np.nan)),
                "suggested_stop_loss": stop_loss_price,
                "suggested_take_profit": float(signal.get("suggested_take_profit", 0.0)),
            }
        return cash, cumulative_costs

    def _exit_decision(
        self,
        position: dict[str, Any],
        row: pd.Series,
        date_idx: int,
        code_price_frame: pd.DataFrame | None,
    ) -> dict[str, Any] | None:
        buy_order: Order = position["buy_order"]
        if self.exit_config.stop_loss_pct is not None:
            stop_price = buy_order.fill_price * (1 - self.exit_config.stop_loss_pct)
            if float(row["low"]) <= stop_price:
                return {"reason": "stop_loss", "price": stop_price}
        if self.exit_config.take_profit_pct is not None:
            take_price = buy_order.fill_price * (1 + self.exit_config.take_profit_pct)
            if float(row["high"]) >= take_price:
                return {"reason": "take_profit", "price": take_price}
        if self.exit_config.sell_on_ma20_break and self._previous_close_below_ma20(code_price_frame, row["date"]):
            return {"reason": "ma20_break", "price": float(row["open"])}
        holding_days = date_idx - int(position["buy_index"]) + 1
        if holding_days >= max(1, self.exit_config.holding_days):
            return {"reason": "holding_period", "price": float(row["open"])}
        return None

    def _trade_row(
        self,
        position: dict[str, Any],
        sell_order: Order,
        sell_index: int,
        exit_reason: str,
    ) -> dict[str, object]:
        buy_order: Order = position["buy_order"]
        gross_return = sell_order.requested_price / buy_order.requested_price - 1
        net_return = sell_order.net_amount / buy_order.net_amount - 1
        total_cost = buy_order.cost["total_cost"] + sell_order.cost["total_cost"]
        total_slippage = buy_order.slippage_cost + sell_order.slippage_cost
        return {
            "code": position["code"],
            "name": position["name"],
            "strategy_name": position["strategy_name"],
            "triggered_strategies": position.get("triggered_strategies", ""),
            "signal_date": position["signal_date"],
            "buy_date": buy_order.order_date,
            "sell_date": sell_order.order_date,
            "buy_price": buy_order.fill_price,
            "sell_price": sell_order.fill_price,
            "gross_buy_amount": buy_order.gross_amount,
            "gross_sell_amount": sell_order.gross_amount,
            "net_buy_cash": buy_order.net_amount,
            "net_sell_cash": sell_order.net_amount,
            "gross_return": float(gross_return),
            "net_return": float(net_return),
            "return": float(net_return),
            "exit_reason": exit_reason,
            "holding_days": int(sell_index - int(position["buy_index"]) + 1),
            "shares": buy_order.shares,
            "score": position.get("score", np.nan),
            "commission": buy_order.cost["commission"] + sell_order.cost["commission"],
            "stamp_tax": sell_order.cost["stamp_tax"],
            "regulatory_fee": buy_order.cost["regulatory_fee"] + sell_order.cost["regulatory_fee"],
            "transfer_fee": buy_order.cost["transfer_fee"] + sell_order.cost["transfer_fee"],
            "total_cost": total_cost,
            "slippage_cost": total_slippage,
        }

    def _schedule_signals(
        self,
        signals: pd.DataFrame,
        calendar: list[pd.Timestamp],
    ) -> dict[pd.Timestamp, pd.DataFrame]:
        if signals.empty:
            return {}
        prepared = signals.copy()
        prepared["code"] = prepared["code"].astype(str).str.zfill(6)
        prepared["signal_date"] = pd.to_datetime(prepared["signal_date"])
        if self.exit_config.deduplicate_signals:
            prepared = prepared.drop_duplicates(["code", "signal_date", "strategy_name"])
        rows_by_date: dict[pd.Timestamp, list[dict[str, object]]] = {}
        calendar_series = pd.Series(calendar)
        for signal in prepared.sort_values(["signal_date", "score"], ascending=[True, False]).to_dict("records"):
            next_dates = calendar_series[calendar_series.gt(pd.to_datetime(signal["signal_date"]))]
            if next_dates.empty:
                continue
            order_date = next_dates.iloc[0]
            rows_by_date.setdefault(order_date, []).append(signal)

        result = {}
        for order_date, rows in rows_by_date.items():
            frame = pd.DataFrame(rows).sort_values("score", ascending=False)
            if self.exit_config.daily_top_n is not None and self.exit_config.daily_top_n > 0:
                frame = frame.head(int(self.exit_config.daily_top_n))
            result[order_date] = frame
        return result

    @staticmethod
    def _prepare_prices(price_data: pd.DataFrame) -> pd.DataFrame:
        if price_data.empty:
            return price_data.copy()
        prices = price_data.copy()
        prices["code"] = prices["code"].astype(str).str.zfill(6)
        prices["date"] = pd.to_datetime(prices["date"])
        for column in ["open", "high", "low", "close", "ma20"]:
            if column in prices.columns:
                prices[column] = pd.to_numeric(prices[column], errors="coerce")
        return prices.dropna(subset=["date"]).sort_values(["date", "code"]).reset_index(drop=True)

    @staticmethod
    def _price_row_for_code_date(frame: pd.DataFrame | None, current_date: pd.Timestamp) -> pd.Series | None:
        if frame is None or frame.empty:
            return None
        rows = frame.loc[frame["date"].eq(current_date)]
        if rows.empty:
            return None
        return rows.iloc[0]

    @staticmethod
    def _previous_close(frame: pd.DataFrame | None, current_date: pd.Timestamp) -> float | None:
        if frame is None or frame.empty:
            return None
        previous = frame.loc[frame["date"].lt(current_date)].tail(1)
        if previous.empty:
            return None
        return float(previous.iloc[0]["close"])

    @staticmethod
    def _previous_close_from_lookup(
        price_lookup: dict[tuple[str, pd.Timestamp], Any],
        code: str,
        current_date: pd.Timestamp,
    ) -> float | None:
        dates = sorted(date for row_code, date in price_lookup if row_code == code and date < current_date)
        if not dates:
            return None
        return float(price_lookup[(code, dates[-1])].close)

    @staticmethod
    def _previous_close_below_ma20(frame: pd.DataFrame | None, current_date: pd.Timestamp) -> bool:
        if frame is None or frame.empty or "ma20" not in frame.columns:
            return False
        previous = frame.loc[frame["date"].lt(current_date)].tail(1)
        if previous.empty or pd.isna(previous.iloc[0]["ma20"]):
            return False
        return float(previous.iloc[0]["close"]) < float(previous.iloc[0]["ma20"])

    @staticmethod
    def _market_value(
        positions: dict[str, dict[str, Any]],
        price_lookup: dict[tuple[str, pd.Timestamp], Any],
        current_date: pd.Timestamp,
    ) -> float:
        value = 0.0
        for code, position in positions.items():
            row = price_lookup.get((code, current_date))
            if row is not None and np.isfinite(float(row.close)):
                value += float(row.close) * int(position["shares"])
            else:
                value += float(position["buy_order"].fill_price) * int(position["shares"])
        return float(value)

    @staticmethod
    def _snapshot_positions(
        positions: dict[str, dict[str, Any]],
        price_lookup: dict[tuple[str, pd.Timestamp], Any],
        current_date: pd.Timestamp,
    ) -> list[dict[str, object]]:
        rows = []
        for code, position in positions.items():
            row = price_lookup.get((code, current_date))
            close = float(row.close) if row is not None else float(position["buy_order"].fill_price)
            market_value = close * int(position["shares"])
            rows.append(
                {
                    "date": current_date,
                    "code": code,
                    "name": position["name"],
                    "strategy_name": position["strategy_name"],
                    "buy_date": position["buy_date"],
                    "shares": position["shares"],
                    "close": close,
                    "market_value": market_value,
                    "unrealized_return": close / float(position["buy_order"].fill_price) - 1,
                }
            )
        return rows

    def _empty_result(self, equity_curve: pd.DataFrame) -> dict[str, object]:
        metrics = calculate_portfolio_metrics(
            equity_curve,
            pd.DataFrame(),
            initial_cash=self.position_config.initial_cash,
        )
        return {
            "portfolio_equity_curve": equity_curve,
            "portfolio_drawdown_curve": pd.DataFrame(columns=["date", "drawdown"]),
            "trades": pd.DataFrame(),
            "orders": pd.DataFrame(),
            "positions_snapshot": pd.DataFrame(),
            "metrics": metrics,
        }
