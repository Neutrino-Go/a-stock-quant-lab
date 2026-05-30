from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from execution.cost_model import (
    TradingCostConfig,
    calculate_buy_cost,
    calculate_sell_cost,
)
from execution.slippage_model import buy_fill_price, sell_fill_price, slippage_cost


class OrderStatus(StrEnum):
    FILLED = "FILLED"
    SKIPPED_NO_CASH = "SKIPPED_NO_CASH"
    SKIPPED_LOT_SIZE = "SKIPPED_LOT_SIZE"
    SKIPPED_LIMIT_UP = "SKIPPED_LIMIT_UP"
    SKIPPED_LIMIT_DOWN = "SKIPPED_LIMIT_DOWN"
    SKIPPED_DUPLICATE_POSITION = "SKIPPED_DUPLICATE_POSITION"
    SKIPPED_NO_PRICE_DATA = "SKIPPED_NO_PRICE_DATA"


@dataclass
class Order:
    code: str
    name: str
    side: str
    signal_date: pd.Timestamp
    order_date: pd.Timestamp
    requested_price: float
    fill_price: float
    shares: int
    gross_amount: float
    cost: dict[str, float] = field(default_factory=dict)
    net_amount: float = 0.0
    status: OrderStatus = OrderStatus.FILLED
    reason: str = ""
    slippage_cost: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "name": self.name,
            "side": self.side,
            "signal_date": self.signal_date,
            "order_date": self.order_date,
            "requested_price": self.requested_price,
            "fill_price": self.fill_price,
            "shares": self.shares,
            "gross_amount": self.gross_amount,
            "commission": self.cost.get("commission", 0.0),
            "stamp_tax": self.cost.get("stamp_tax", 0.0),
            "regulatory_fee": self.cost.get("regulatory_fee", 0.0),
            "transfer_fee": self.cost.get("transfer_fee", 0.0),
            "total_cost": self.cost.get("total_cost", 0.0),
            "slippage_cost": self.slippage_cost,
            "net_amount": self.net_amount,
            "status": self.status.value,
            "reason": self.reason,
        }


@dataclass
class TradeFill:
    code: str
    name: str
    buy_order: Order
    sell_order: Order
    strategy_name: str
    signal_date: pd.Timestamp
    exit_reason: str


def round_down_to_lot(shares: float | int, lot_size: int = 100) -> int:
    if shares <= 0:
        return 0
    return int(shares) // lot_size * lot_size


def is_limit_up(open_price: float, prev_close: float, limit_pct: float = 0.10) -> bool:
    if open_price <= 0 or prev_close <= 0:
        return False
    return float(open_price) >= float(prev_close) * (1 + limit_pct * 0.995)


def is_limit_down(open_price: float, prev_close: float, limit_pct: float = 0.10) -> bool:
    if open_price <= 0 or prev_close <= 0:
        return False
    return float(open_price) <= float(prev_close) * (1 - limit_pct * 0.995)


def create_buy_order(
    code: str,
    name: str,
    signal_date: pd.Timestamp,
    order_date: pd.Timestamp,
    requested_price: float,
    requested_shares: int,
    cash_available: float,
    config: TradingCostConfig | None = None,
    reason: str = "",
) -> Order:
    config = config or TradingCostConfig()
    shares = round_down_to_lot(requested_shares)
    if shares < 100:
        return _skipped_order(
            code,
            name,
            "BUY",
            signal_date,
            order_date,
            requested_price,
            OrderStatus.SKIPPED_LOT_SIZE,
            "buy size is below one A-share lot",
        )

    fill_price = buy_fill_price(requested_price, config)
    cost = calculate_buy_cost(fill_price, shares, config)
    gross = fill_price * shares
    net = gross + cost["total_cost"]
    if net > cash_available:
        return _skipped_order(
            code,
            name,
            "BUY",
            signal_date,
            order_date,
            requested_price,
            OrderStatus.SKIPPED_NO_CASH,
            "cash is not enough for requested shares",
        )
    return Order(
        code=str(code).zfill(6),
        name=name,
        side="BUY",
        signal_date=pd.to_datetime(signal_date),
        order_date=pd.to_datetime(order_date),
        requested_price=float(requested_price),
        fill_price=float(fill_price),
        shares=shares,
        gross_amount=float(gross),
        cost=cost,
        net_amount=float(net),
        status=OrderStatus.FILLED,
        reason=reason,
        slippage_cost=slippage_cost(requested_price, fill_price, shares),
    )


def create_sell_order(
    code: str,
    name: str,
    signal_date: pd.Timestamp,
    order_date: pd.Timestamp,
    requested_price: float,
    shares: int,
    config: TradingCostConfig | None = None,
    reason: str = "",
) -> Order:
    config = config or TradingCostConfig()
    fill_price = sell_fill_price(requested_price, config)
    cost = calculate_sell_cost(fill_price, shares, config)
    gross = fill_price * shares
    net = gross - cost["total_cost"]
    return Order(
        code=str(code).zfill(6),
        name=name,
        side="SELL",
        signal_date=pd.to_datetime(signal_date),
        order_date=pd.to_datetime(order_date),
        requested_price=float(requested_price),
        fill_price=float(fill_price),
        shares=int(shares),
        gross_amount=float(gross),
        cost=cost,
        net_amount=float(net),
        status=OrderStatus.FILLED,
        reason=reason,
        slippage_cost=slippage_cost(requested_price, fill_price, shares),
    )


def skipped_order(
    code: str,
    name: str,
    side: str,
    signal_date: pd.Timestamp,
    order_date: pd.Timestamp,
    requested_price: float,
    status: OrderStatus,
    reason: str,
) -> Order:
    return _skipped_order(code, name, side, signal_date, order_date, requested_price, status, reason)


def _skipped_order(
    code: str,
    name: str,
    side: str,
    signal_date: pd.Timestamp,
    order_date: pd.Timestamp,
    requested_price: float,
    status: OrderStatus,
    reason: str,
) -> Order:
    return Order(
        code=str(code).zfill(6),
        name=name,
        side=side,
        signal_date=pd.to_datetime(signal_date),
        order_date=pd.to_datetime(order_date),
        requested_price=float(requested_price) if requested_price else 0.0,
        fill_price=0.0,
        shares=0,
        gross_amount=0.0,
        cost={
            "commission": 0.0,
            "stamp_tax": 0.0,
            "regulatory_fee": 0.0,
            "transfer_fee": 0.0,
            "total_cost": 0.0,
        },
        net_amount=0.0,
        status=status,
        reason=reason,
    )
