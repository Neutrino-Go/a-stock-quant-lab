from execution.cost_model import TradingCostConfig
from portfolio.position_sizer import PositionSizingConfig, calculate_position_size


def test_fixed_cash_position_size_rounds_down_to_lot_size():
    config = PositionSizingConfig(position_size_method="fixed_cash", fixed_cash_per_trade=1250)

    result = calculate_position_size(
        cash=10_000,
        price=10.0,
        stop_loss_price=9.0,
        config=config,
        cost_config=TradingCostConfig(slippage_bps=0),
    )

    assert result.shares == 100
    assert result.status == "OK"


def test_position_size_skips_when_cash_cannot_buy_one_lot_with_costs():
    config = PositionSizingConfig(position_size_method="fixed_cash", fixed_cash_per_trade=1000)

    result = calculate_position_size(
        cash=900,
        price=10.0,
        stop_loss_price=9.0,
        config=config,
        cost_config=TradingCostConfig(slippage_bps=0),
    )

    assert result.shares == 0
    assert result.status == "SKIPPED_NO_CASH"


def test_equal_cash_uses_available_cash_divided_by_remaining_slots():
    config = PositionSizingConfig(
        initial_cash=100_000,
        max_positions=5,
        position_size_method="equal_cash",
        cash_reserve_pct=0.0,
    )

    result = calculate_position_size(
        cash=50_000,
        price=20.0,
        stop_loss_price=18.0,
        config=config,
        cost_config=TradingCostConfig(slippage_bps=0),
        current_positions=3,
    )

    assert result.target_cash <= 25_000
    assert result.shares % 100 == 0
    assert result.cash_required <= 25_000


def test_risk_percent_position_size_uses_stop_loss_distance():
    config = PositionSizingConfig(
        initial_cash=100_000,
        position_size_method="risk_percent",
        risk_per_trade=0.01,
        max_position_pct=1.0,
        cash_reserve_pct=0.0,
    )

    result = calculate_position_size(
        cash=100_000,
        price=10.0,
        stop_loss_price=9.0,
        config=config,
        cost_config=TradingCostConfig(slippage_bps=0),
    )

    assert result.shares == 1000
