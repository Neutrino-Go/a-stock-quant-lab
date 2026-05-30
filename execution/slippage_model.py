from __future__ import annotations


def slippage_multiplier(slippage_bps: float) -> float:
    return float(slippage_bps) / 10_000


def buy_fill_price(price: float, config: object) -> float:
    return round(float(price) * (1 + slippage_multiplier(getattr(config, "slippage_bps", 0))), 6)


def sell_fill_price(price: float, config: object) -> float:
    return round(float(price) * (1 - slippage_multiplier(getattr(config, "slippage_bps", 0))), 6)


def slippage_cost(reference_price: float, fill_price: float, shares: int) -> float:
    return abs(float(fill_price) - float(reference_price)) * int(shares)
