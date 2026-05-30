import pandas as pd

from ranking.leaderboard import build_leaderboard_from_metrics, rate_strategy


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
