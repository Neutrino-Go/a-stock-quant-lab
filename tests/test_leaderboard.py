import pandas as pd

from ranking.leaderboard import build_leaderboard_from_metrics, rate_strategy


def test_rate_strategy_rules():
    assert rate_strategy(cagr=0.20, sharpe_ratio=1.6, max_drawdown=-0.10) == "A"
    assert rate_strategy(cagr=0.10, sharpe_ratio=1.1, max_drawdown=-0.30) == "B"
    assert rate_strategy(cagr=0.01, sharpe_ratio=0.2, max_drawdown=-0.30) == "C"
    assert rate_strategy(cagr=-0.01, sharpe_ratio=0.2, max_drawdown=-0.30) == "D"


def test_build_leaderboard_from_metrics_uses_display_names_and_required_columns():
    metrics = pd.DataFrame(
        [
            {
                "strategy_name": "trend_strength_strategy",
                "period": "1个月",
                "win_rate": 0.63,
                "avg_return": 0.04,
                "CAGR": 0.22,
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
                "max_drawdown": -0.28,
                "sharpe_ratio": -0.4,
                "profit_loss_ratio": 0.7,
            },
        ]
    )

    leaderboard = build_leaderboard_from_metrics(metrics)

    assert list(leaderboard["策略"]) == ["趋势强势股策略", "放量突破策略"]
    assert list(leaderboard["评级"]) == ["A", "D"]
    assert {"策略", "评级", "胜率", "年化", "回撤", "夏普"}.issubset(leaderboard.columns)
