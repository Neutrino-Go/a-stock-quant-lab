from execution.cost_model import TradingCostConfig
from execution.slippage_model import buy_fill_price, sell_fill_price


def test_buy_slippage_increases_fill_price():
    config = TradingCostConfig(slippage_bps=5)

    assert buy_fill_price(10.0, config) == 10.005


def test_sell_slippage_decreases_fill_price():
    config = TradingCostConfig(slippage_bps=5)

    assert sell_fill_price(10.0, config) == 9.995
