from __future__ import annotations

from dataclasses import dataclass

from execution.cost_model import TradingCostConfig, calculate_net_buy_cash_required
from execution.order_model import round_down_to_lot


@dataclass(frozen=True)
class PositionSizingConfig:
    initial_cash: float = 1_000_000
    max_positions: int = 5
    position_size_method: str = "equal_cash"
    fixed_cash_per_trade: float = 100_000
    risk_per_trade: float = 0.01
    max_position_pct: float = 0.20
    cash_reserve_pct: float = 0.05


@dataclass(frozen=True)
class PositionSizeResult:
    shares: int
    target_cash: float
    cash_required: float
    status: str
    reason: str = ""


def calculate_position_size(
    cash: float,
    price: float,
    stop_loss_price: float,
    config: PositionSizingConfig,
    cost_config: TradingCostConfig | None = None,
    current_positions: int = 0,
) -> PositionSizeResult:
    cost_config = cost_config or TradingCostConfig()
    cash = float(cash)
    price = float(price)
    if cash <= 0 or price <= 0:
        return PositionSizeResult(0, 0.0, 0.0, "SKIPPED_NO_CASH", "cash or price is invalid")

    available_cash = cash * max(0.0, 1 - config.cash_reserve_pct)
    remaining_slots = max(int(config.max_positions) - int(current_positions), 1)
    target_cash = _target_cash(available_cash, price, stop_loss_price, config, remaining_slots)
    max_cash = min(available_cash, config.initial_cash * config.max_position_pct)
    target_cash = max(0.0, min(target_cash, max_cash))
    shares = round_down_to_lot(target_cash / price)
    shares, cash_required = _fit_to_cash(shares, price, available_cash, cost_config)

    if shares < 100:
        return PositionSizeResult(
            0,
            target_cash,
            0.0,
            "SKIPPED_NO_CASH",
            "cash is not enough to buy one lot after costs",
        )
    return PositionSizeResult(shares, target_cash, cash_required, "OK")


def _target_cash(
    available_cash: float,
    price: float,
    stop_loss_price: float,
    config: PositionSizingConfig,
    remaining_slots: int,
) -> float:
    if config.position_size_method == "fixed_cash":
        return min(config.fixed_cash_per_trade, available_cash)
    if config.position_size_method == "risk_percent":
        risk_per_share = max(price - float(stop_loss_price), 0.0)
        if risk_per_share <= 0:
            return 0.0
        risk_cash = config.initial_cash * config.risk_per_trade
        return min((risk_cash / risk_per_share) * price, available_cash)
    return available_cash / remaining_slots


def _fit_to_cash(
    shares: int,
    price: float,
    available_cash: float,
    cost_config: TradingCostConfig,
) -> tuple[int, float]:
    while shares >= 100:
        cash_required = calculate_net_buy_cash_required(price, shares, cost_config)
        if cash_required <= available_cash:
            return shares, cash_required
        shares -= 100
    return 0, 0.0
