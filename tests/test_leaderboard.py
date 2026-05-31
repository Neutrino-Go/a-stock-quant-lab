import pandas as pd

import ranking.leaderboard as leaderboard_module
from backtest.backtest_engine import calculate_trade_metrics
from execution.cost_model import TradingCostConfig
from ranking.leaderboard import build_leaderboard_from_metrics, build_strategy_leaderboard, rate_strategy


def test_rate_strategy_rules_use_net_cagr_and_negative_drawdown_comparison():
    assert rate_strategy(net_cagr=0.20, sharpe_ratio=1.6, max_drawdown=-0.10) == "A"
    assert rate_strategy(net_cagr=0.10, sharpe_ratio=1.1, max_drawdown=-0.20) == "B"
    assert rate_strategy(net_cagr=0.01, sharpe_ratio=0.2, max_drawdown=-0.30) == "C"
    assert rate_strategy(net_cagr=-0.01, sharpe_ratio=0.2, max_drawdown=-0.30) == "D"


def test_build_leaderboard_from_metrics_uses_display_names_and_net_columns():
    metrics = pd.DataFrame(
        [
            {
                "strategy_name": "trend_strength_strategy",
                "period": "1个月",
                "win_rate": 0.63,
                "avg_return": 0.04,
                "CAGR": 0.22,
                "gross_annual_return": 0.23,
                "net_annual_return": 0.22,
                "cost_impact": 0.01,
                "max_drawdown": -0.12,
                "sharpe_ratio": 1.8,
                "profit_loss_ratio": 1.5,
            },
            {
                "strategy_name": "volume_breakout_strategy",
                "period": "1个月",
                "win_rate": 0.38,
                "avg_return": -0.01,
                "CAGR": -0.05,
                "gross_annual_return": -0.04,
                "net_annual_return": -0.05,
                "cost_impact": 0.01,
                "max_drawdown": -0.28,
                "sharpe_ratio": -0.4,
                "profit_loss_ratio": 0.7,
            },
        ]
    )

    leaderboard = build_leaderboard_from_metrics(metrics)

    assert list(leaderboard["strategy"]) == ["趋势强势股策略", "放量突破策略"]
    assert list(leaderboard["rating"]) == ["A", "D"]
    assert {
        "strategy",
        "rating",
        "win_rate",
        "gross_annual_return",
        "net_annual_return",
        "cost_impact",
        "max_drawdown",
        "sharpe_ratio",
    }.issubset(leaderboard.columns)


def test_build_leaderboard_from_metrics_rates_by_net_result():
    metrics = pd.DataFrame(
        [
            {
                "strategy_name": "trend_strength_strategy",
                "period": "1个月",
                "win_rate": 0.8,
                "avg_return": 0.04,
                "gross_annual_return": 0.30,
                "net_annual_return": -0.01,
                "cost_impact": 0.31,
                "max_drawdown": -0.10,
                "sharpe_ratio": 2.0,
                "profit_loss_ratio": 1.5,
            }
        ]
    )

    leaderboard = build_leaderboard_from_metrics(metrics)

    assert leaderboard.iloc[0]["rating"] == "D"


def test_strategy_leaderboard_applies_default_trading_costs(monkeypatch):
    def fake_strategy(_factor_df, **_params):
        return pd.DataFrame()

    def fake_backtest(**_kwargs):
        trades = pd.DataFrame(
            {
                "code": ["000001"],
                "name": ["Sample"],
                "buy_date": [pd.Timestamp("2023-01-01")],
                "exit_date": [pd.Timestamp("2024-01-01")],
                "buy_price": [10.0],
                "exit_price": [11.0],
                "return": [0.10],
            }
        )
        return calculate_trade_metrics(trades)

    monkeypatch.setattr(leaderboard_module, "STRATEGIES", {"trend_strength_strategy": fake_strategy})
    monkeypatch.setattr(leaderboard_module, "run_strategy_backtest", fake_backtest)
    factor_df = pd.DataFrame(
        {
            "code": ["000001", "000001", "000001"],
            "date": pd.bdate_range("2024-01-01", periods=3),
            "close": [10.0, 10.5, 11.0],
        }
    )

    leaderboard = build_strategy_leaderboard(
        factor_df,
        periods={"test": 2},
        cost_config=TradingCostConfig(),
    )
    row = leaderboard.iloc[0]

    assert row["cost_impact"] >= 0
    assert row["net_annual_return"] <= row["gross_annual_return"]
    assert row["total_commission"] > 0
    assert row["total_stamp_tax"] > 0
    assert row["total_slippage_cost"] > 0
