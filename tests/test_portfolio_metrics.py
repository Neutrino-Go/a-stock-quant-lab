import pandas as pd

from portfolio.portfolio_metrics import calculate_portfolio_metrics


def test_portfolio_metrics_calculate_drawdown_and_cost_impact():
    equity = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "cash": [100_000, 20_000, 104_500],
            "market_value": [0, 90_000, 0],
            "total_equity": [100_000, 110_000, 104_500],
            "gross_total_equity": [100_000, 111_000, 105_500],
            "positions_count": [0, 1, 0],
        }
    )
    trades = pd.DataFrame(
        {
            "net_return": [0.045],
            "gross_return": [0.055],
            "holding_days": [1],
            "gross_buy_amount": [100_000],
            "gross_sell_amount": [105_500],
            "total_cost": [1000],
            "commission": [600],
            "stamp_tax": [250],
            "slippage_cost": [150],
        }
    )

    metrics = calculate_portfolio_metrics(equity, trades, initial_cash=100_000)

    assert round(metrics["net_total_return"], 3) == 0.045
    assert round(metrics["gross_total_return"], 3) == 0.055
    assert round(metrics["cost_impact"], 3) == 0.010
    assert round(metrics["max_drawdown"], 3) == -0.05
    assert metrics["total_trades"] == 1
    assert metrics["gross_total_return"] >= metrics["net_total_return"]


def test_portfolio_metrics_empty_inputs_do_not_error():
    metrics = calculate_portfolio_metrics(pd.DataFrame(), pd.DataFrame(), initial_cash=100_000)

    assert metrics["final_equity"] == 100_000
    assert metrics["total_trades"] == 0
    assert metrics["max_drawdown"] == 0
