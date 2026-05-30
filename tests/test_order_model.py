import pandas as pd

from execution.cost_model import TradingCostConfig
from execution.order_model import (
    OrderStatus,
    create_buy_order,
    create_sell_order,
    is_limit_down,
    is_limit_up,
    round_down_to_lot,
)


def test_buy_shares_are_rounded_down_to_100_share_lots():
    assert round_down_to_lot(99) == 0
    assert round_down_to_lot(199) == 100
    assert round_down_to_lot(250) == 200


def test_buy_order_skips_when_cash_is_not_enough_for_one_lot():
    order = create_buy_order(
        code="000001",
        name="Ping An",
        signal_date=pd.Timestamp("2024-01-01"),
        order_date=pd.Timestamp("2024-01-02"),
        requested_price=10.0,
        requested_shares=100,
        cash_available=900.0,
        config=TradingCostConfig(),
    )

    assert order.status == OrderStatus.SKIPPED_NO_CASH
    assert order.shares == 0
    assert order.net_amount == 0.0


def test_sell_order_fills_all_requested_shares_and_is_net_of_costs():
    order = create_sell_order(
        code="000001",
        name="Ping An",
        signal_date=pd.Timestamp("2024-01-01"),
        order_date=pd.Timestamp("2024-01-05"),
        requested_price=11.0,
        shares=100,
        config=TradingCostConfig(slippage_bps=0),
    )

    assert order.status == OrderStatus.FILLED
    assert order.shares == 100
    assert order.gross_amount == 1100.0
    assert order.net_amount < order.gross_amount
    assert order.cost["stamp_tax"] > 0


def test_limit_up_and_limit_down_checks_use_configurable_threshold():
    assert is_limit_up(open_price=11.0, prev_close=10.0, limit_pct=0.10)
    assert is_limit_down(open_price=9.0, prev_close=10.0, limit_pct=0.10)
    assert not is_limit_up(open_price=10.5, prev_close=10.0, limit_pct=0.10)
