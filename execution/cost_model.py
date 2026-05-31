from __future__ import annotations

from dataclasses import dataclass

from execution.slippage_model import buy_fill_price, sell_fill_price


@dataclass(frozen=True)
class TradingCostConfig:
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_tax_rate_sell: float = 0.0005
    regulatory_fee_rate: float = 0.00002
    transfer_fee_rate: float = 0.00001
    slippage_bps: float = 5.0


def calculate_buy_cost(
    price: float,
    shares: int,
    config: TradingCostConfig | None = None,
) -> dict[str, float]:
    config = config or TradingCostConfig()
    gross = _gross_amount(price, shares)
    commission = _commission(gross, config)
    regulatory_fee = gross * config.regulatory_fee_rate
    transfer_fee = gross * config.transfer_fee_rate
    return _breakdown(
        commission=commission,
        stamp_tax=0.0,
        regulatory_fee=regulatory_fee,
        transfer_fee=transfer_fee,
    )


def calculate_sell_cost(
    price: float,
    shares: int,
    config: TradingCostConfig | None = None,
) -> dict[str, float]:
    config = config or TradingCostConfig()
    gross = _gross_amount(price, shares)
    commission = _commission(gross, config)
    stamp_tax = gross * config.stamp_tax_rate_sell
    regulatory_fee = gross * config.regulatory_fee_rate
    transfer_fee = gross * config.transfer_fee_rate
    return _breakdown(
        commission=commission,
        stamp_tax=stamp_tax,
        regulatory_fee=regulatory_fee,
        transfer_fee=transfer_fee,
    )


def calculate_net_buy_cash_required(
    price: float,
    shares: int,
    config: TradingCostConfig | None = None,
) -> float:
    config = config or TradingCostConfig()
    fill_price = buy_fill_price(price, config)
    gross = _gross_amount(fill_price, shares)
    return gross + calculate_buy_cost(fill_price, shares, config)["total_cost"]


def calculate_net_sell_cash_received(
    price: float,
    shares: int,
    config: TradingCostConfig | None = None,
) -> float:
    config = config or TradingCostConfig()
    fill_price = sell_fill_price(price, config)
    gross = _gross_amount(fill_price, shares)
    return gross - calculate_sell_cost(fill_price, shares, config)["total_cost"]


def _gross_amount(price: float, shares: int) -> float:
    if shares <= 0 or price <= 0:
        return 0.0
    return float(price) * int(shares)


def _commission(gross_amount: float, config: TradingCostConfig) -> float:
    if gross_amount <= 0:
        return 0.0
    return max(gross_amount * config.commission_rate, config.min_commission)


def _breakdown(
    commission: float,
    stamp_tax: float,
    regulatory_fee: float,
    transfer_fee: float,
) -> dict[str, float]:
    total_cost = commission + stamp_tax + regulatory_fee + transfer_fee
    return {
        "commission": float(commission),
        "stamp_tax": float(stamp_tax),
        "regulatory_fee": float(regulatory_fee),
        "transfer_fee": float(transfer_fee),
        "total_cost": float(total_cost),
    }
