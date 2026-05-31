import pandas as pd
import pytest

from execution.cost_model import TradingCostConfig
from portfolio.portfolio_engine import ExitConfig, PortfolioBacktestEngine
from portfolio.position_sizer import PositionSizingConfig


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=6)
    rows = []
    for idx, current_date in enumerate(dates):
        close = 10.0 + idx
        rows.append(
            {
                "code": "000001",
                "name": "Sample",
                "date": current_date,
                "open": close,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": 1_000_000,
                "amount": 100_000_000,
                "turnover": 1.0,
                "ma20": close * 0.95,
                "ma60": close * 0.90,
            }
        )
    return pd.DataFrame(rows)


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["000001"],
            "name": ["Sample"],
            "signal_date": [pd.Timestamp("2024-01-01")],
            "strategy_name": ["unit_strategy"],
            "score": [90.0],
            "signal_reason": ["unit"],
            "suggested_buy_price": [10.0],
            "suggested_stop_loss": [9.0],
            "suggested_take_profit": [12.0],
            "risk_level": ["medium"],
        }
    )


def _engine(initial_cash: float = 100_000) -> PortfolioBacktestEngine:
    return PortfolioBacktestEngine(
        cost_config=TradingCostConfig(slippage_bps=0),
        position_config=PositionSizingConfig(
            initial_cash=initial_cash,
            max_positions=5,
            position_size_method="fixed_cash",
            fixed_cash_per_trade=10_000,
            cash_reserve_pct=0.0,
            max_position_pct=1.0,
        ),
        exit_config=ExitConfig(
            holding_days=3,
            take_profit_pct=None,
            stop_loss_pct=None,
            sell_on_ma20_break=False,
            enable_limit_check=False,
        ),
    )


def test_portfolio_buys_signal_on_next_trading_day_open():
    result = _engine().run(_signals(), _prices(), start_date="2024-01-01", end_date="2024-01-08")

    trades = result["trades"]
    orders = result["orders"]

    assert result["portfolio_equity_curve"].iloc[0]["total_equity"] == 100_000
    assert orders.loc[orders["side"].eq("BUY")].iloc[0]["order_date"] == pd.Timestamp("2024-01-02")
    assert trades.iloc[0]["buy_date"] == pd.Timestamp("2024-01-02")


def test_portfolio_skips_duplicate_position_by_default():
    signals = pd.concat([_signals(), _signals()], ignore_index=True)
    signals.loc[1, "signal_date"] = pd.Timestamp("2024-01-02")

    result = _engine().run(signals, _prices(), start_date="2024-01-01", end_date="2024-01-08")
    buy_orders = result["orders"].loc[result["orders"]["side"].eq("BUY")]

    assert len(buy_orders.loc[buy_orders["status"].eq("FILLED")]) == 1
    assert "SKIPPED_DUPLICATE_POSITION" in set(buy_orders["status"])


def test_portfolio_rejects_add_position_mode_in_v3():
    with pytest.raises(
        NotImplementedError,
        match="allow_add_position=True is not supported in V3",
    ):
        PortfolioBacktestEngine(exit_config=ExitConfig(allow_add_position=True))


def test_portfolio_rejects_add_position_mode_at_runtime_if_mutated():
    engine = _engine()
    object.__setattr__(engine.exit_config, "allow_add_position", True)

    with pytest.raises(
        NotImplementedError,
        match="allow_add_position=True is not supported in V3",
    ):
        engine.run(_signals(), _prices())


def test_portfolio_respects_t_plus_one_before_selling():
    prices = _prices()
    prices.loc[1, "low"] = 1.0
    prices.loc[2, "low"] = 1.0
    engine = PortfolioBacktestEngine(
        cost_config=TradingCostConfig(slippage_bps=0),
        position_config=PositionSizingConfig(
            initial_cash=100_000,
            max_positions=5,
            position_size_method="fixed_cash",
            fixed_cash_per_trade=10_000,
            cash_reserve_pct=0.0,
            max_position_pct=1.0,
        ),
        exit_config=ExitConfig(
            holding_days=5,
            take_profit_pct=None,
            stop_loss_pct=0.05,
            sell_on_ma20_break=False,
            enable_limit_check=False,
        ),
    )

    result = engine.run(_signals(), prices, start_date="2024-01-01", end_date="2024-01-08")
    trade = result["trades"].iloc[0]

    assert trade["sell_date"] > trade["buy_date"]
    assert trade["exit_reason"] == "stop_loss"


def test_portfolio_empty_results_do_not_error():
    result = _engine().run(pd.DataFrame(), _prices(), start_date="2024-01-01", end_date="2024-01-08")

    assert result["metrics"]["total_trades"] == 0
    assert result["orders"].empty
