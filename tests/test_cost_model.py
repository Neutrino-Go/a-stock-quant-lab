from execution.cost_model import (
    TradingCostConfig,
    calculate_buy_cost,
    calculate_net_buy_cash_required,
    calculate_sell_cost,
)


def test_buy_cost_applies_minimum_commission_and_no_stamp_tax():
    config = TradingCostConfig()

    cost = calculate_buy_cost(price=10.0, shares=100, config=config)

    assert cost["commission"] == 5.0
    assert cost["stamp_tax"] == 0.0
    assert round(cost["regulatory_fee"], 2) == 0.02
    assert round(cost["transfer_fee"], 2) == 0.01
    assert round(cost["total_cost"], 2) == 5.03


def test_sell_cost_charges_stamp_tax_only_on_sell_side():
    config = TradingCostConfig()

    buy_cost = calculate_buy_cost(price=10.0, shares=1000, config=config)
    sell_cost = calculate_sell_cost(price=10.0, shares=1000, config=config)

    assert buy_cost["stamp_tax"] == 0.0
    assert sell_cost["stamp_tax"] == 5.0
    assert sell_cost["commission"] == 5.0
    assert sell_cost["total_cost"] > buy_cost["total_cost"]


def test_net_buy_cash_required_includes_slippage_and_fees():
    config = TradingCostConfig(slippage_bps=10)

    cash_required = calculate_net_buy_cash_required(price=10.0, shares=100, config=config)

    assert cash_required > 1000.0
    assert round(cash_required, 2) == 1006.03
