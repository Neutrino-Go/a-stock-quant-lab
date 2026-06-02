from __future__ import annotations

from workflow.research_scorecard import build_strategy_scorecard


def _stability_result(
    net_cagr: float = 0.18,
    max_drawdown: float = -0.12,
    sharpe: float = 1.6,
    stability_score: float = 85,
    overfit_risk: str = "LOW",
    total_trades: int = 80,
    positive_ratio: float = 0.7,
    sensitivity_score: float = 82,
):
    return {
        "analysis_id": "exp-best",
        "best_params": {"return_20d": 0.1},
        "out_of_sample_result": {
            "net_CAGR": net_cagr,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "calmar_ratio": abs(net_cagr / max_drawdown),
            "expectancy": 0.01,
            "total_trades": total_trades,
            "cost_impact": 0.01,
            "turnover": 1.0,
        },
        "walk_forward_result": {
            "summary": {
                "positive_test_fold_ratio": positive_ratio,
                "avg_test_net_CAGR": net_cagr * 0.8,
                "fold_count": 6,
            }
        },
        "sensitivity_result": {"sensitivity_score": sensitivity_score},
        "stability_score": {"stability_score": stability_score, "final_rating": "A"},
        "overfit_result": {
            "overfit_risk": overfit_risk,
            "overfit_score": 15 if overfit_risk == "LOW" else 85,
            "risk_flags": [],
        },
        "market_regime_breakdown": {
            "BULL": {"trade_count": 30, "net_CAGR": 0.2},
            "BEAR": {"trade_count": 20, "net_CAGR": 0.03},
            "SIDEWAYS": {"trade_count": 20, "net_CAGR": 0.08},
            "HIGH_VOLATILITY": {"trade_count": 10, "net_CAGR": 0.01},
        },
    }


def test_scorecard_adds_strong_result_to_candidates():
    scorecard = build_strategy_scorecard(
        optimization_result={"top_params": [{"params": {"return_20d": 0.1}}]},
        stability_results=[_stability_result()],
        overfit_results=[],
        sensitivity_results=[],
        regime_results=[],
        thresholds={},
    )

    assert scorecard["decision"] == "add_to_candidates"
    assert scorecard["final_rating"] == "A"
    assert scorecard["best_params"] == {"return_20d": 0.1}
    assert scorecard["best_experiment_id"] == "exp-best"
    assert 0 <= scorecard["overall_score"] <= 100
    assert "buy" not in scorecard["decision_reason"].lower()
    assert "guarantee" not in scorecard["decision_reason"].lower()


def test_scorecard_rejects_high_overfit_or_negative_oos_result():
    scorecard = build_strategy_scorecard(
        optimization_result={},
        stability_results=[
            _stability_result(
                net_cagr=-0.02,
                overfit_risk="HIGH",
                total_trades=20,
                positive_ratio=0.3,
                sensitivity_score=25,
            )
        ],
        overfit_results=[],
        sensitivity_results=[],
        regime_results=[],
        thresholds={},
    )

    assert scorecard["decision"] == "reject"
    assert scorecard["final_rating"] == "D"
    assert "high_overfit_risk" in scorecard["flags"]
    assert "out_of_sample_return_non_positive" in scorecard["flags"]


def test_scorecard_handles_missing_data_with_flags_and_clamped_scores():
    scorecard = build_strategy_scorecard(
        optimization_result={},
        stability_results=[],
        overfit_results=[],
        sensitivity_results=[],
        regime_results=[],
        thresholds={},
    )

    assert scorecard["decision"] == "reject"
    assert "missing_stability_results" in scorecard["flags"]
    assert all(0 <= value <= 100 for value in scorecard["sub_scores"].values())
