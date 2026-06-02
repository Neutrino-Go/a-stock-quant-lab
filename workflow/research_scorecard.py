from __future__ import annotations

from typing import Any


DEFAULT_THRESHOLDS = {
    "candidate_score": 80,
    "continue_score": 65,
    "watch_score": 50,
    "candidate_min_positive_ratio": 0.6,
    "candidate_min_trades": 50,
}


def build_strategy_scorecard(
    optimization_result: dict[str, Any] | None,
    stability_results: list[dict[str, Any]] | None,
    overfit_results: list[dict[str, Any]] | None,
    sensitivity_results: list[dict[str, Any]] | None,
    regime_results: list[dict[str, Any]] | None,
    thresholds: dict[str, Any] | None,
) -> dict[str, Any]:
    del overfit_results, sensitivity_results, regime_results
    settings = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    stability_results = stability_results or []
    flags: list[str] = []
    if not stability_results:
        flags.append("missing_stability_results")
        return _scorecard(
            overall_score=0,
            final_rating="D",
            decision="reject",
            decision_reason="No completed stability result is available for this research run.",
            sub_scores=_empty_sub_scores(),
            flags=flags,
            best_params={},
            best_experiment_id=None,
        )

    ranked = sorted(stability_results, key=_candidate_quality_score, reverse=True)
    best = ranked[0]
    metrics = dict(best.get("out_of_sample_result", {}) or {})
    walk_summary = dict(dict(best.get("walk_forward_result", {}) or {}).get("summary", {}) or {})
    sensitivity = dict(best.get("sensitivity_result", {}) or {})
    stability = dict(best.get("stability_score", {}) or {})
    overfit = dict(best.get("overfit_result", {}) or {})
    regimes = dict(best.get("market_regime_breakdown", {}) or {})

    net_cagr = _float(metrics.get("net_CAGR"))
    max_drawdown = _float(metrics.get("max_drawdown"))
    sharpe = _float(metrics.get("sharpe_ratio"))
    calmar = _float(metrics.get("calmar_ratio"))
    expectancy = _float(metrics.get("expectancy"))
    total_trades = int(_float(metrics.get("total_trades")))
    cost_impact = max(_float(metrics.get("cost_impact")), 0.0)
    turnover = max(_float(metrics.get("turnover")), 0.0)
    positive_ratio = _float(walk_summary.get("positive_test_fold_ratio"))
    avg_test_net_cagr = _float(walk_summary.get("avg_test_net_CAGR"))
    fold_count = int(_float(walk_summary.get("fold_count")))
    sensitivity_score = _float(sensitivity.get("sensitivity_score"))
    stability_value = _float(stability.get("stability_score"))
    overfit_risk = str(overfit.get("overfit_risk", "HIGH") or "HIGH")
    overfit_score_raw = _float(overfit.get("overfit_score"))

    sub_scores = {
        "return_score": _clamp(50 + net_cagr * 250 + avg_test_net_cagr * 100 + expectancy * 500),
        "risk_score": _clamp(100 + max_drawdown * 180 + calmar * 6 + sharpe * 3 - max(0.0, -max_drawdown - 0.3) * 100),
        "stability_score": _clamp(stability_value * 0.55 + positive_ratio * 100 * 0.25 + sensitivity_score * 0.20),
        "overfit_score": _clamp(100 - overfit_score_raw),
        "sample_size_score": _clamp(min(total_trades / 80, 1) * 80 + min(fold_count / 5, 1) * 20),
        "cost_score": _clamp(100 - cost_impact * 1000 - turnover * 5),
        "regime_score": _regime_score(regimes),
    }
    overall = _clamp(
        sub_scores["return_score"] * 0.22
        + sub_scores["risk_score"] * 0.16
        + sub_scores["stability_score"] * 0.24
        + sub_scores["overfit_score"] * 0.16
        + sub_scores["sample_size_score"] * 0.10
        + sub_scores["cost_score"] * 0.06
        + sub_scores["regime_score"] * 0.06
    )

    if overfit_risk == "HIGH":
        flags.append("high_overfit_risk")
    if net_cagr <= 0:
        flags.append("out_of_sample_return_non_positive")
    if total_trades < settings["candidate_min_trades"]:
        flags.append("sample_size_below_candidate_threshold")
    if positive_ratio < 0.5:
        flags.append("walk_forward_positive_ratio_low")

    if (
        overall >= settings["candidate_score"]
        and overfit_risk == "LOW"
        and net_cagr > 0
        and positive_ratio >= settings["candidate_min_positive_ratio"]
        and total_trades >= settings["candidate_min_trades"]
    ):
        decision = "add_to_candidates"
        reason = "Historical validation is strong enough to track this parameter set in the candidate library."
    elif overall >= settings["continue_score"] and overfit_risk != "HIGH" and net_cagr >= 0:
        decision = "continue_research"
        reason = "Historical validation is acceptable, but more research is required before candidate promotion."
    elif overall >= settings["watch_score"]:
        decision = "watch"
        reason = "Some historical evidence is useful, but risk, sample size, or stability limits require observation."
    else:
        decision = "reject"
        reason = "Historical validation is weak or overfitting risk is too high for continued tracking."
    if overfit_risk == "HIGH" or (net_cagr <= 0 and positive_ratio < 0.5):
        decision = "reject"
        reason = "Historical validation indicates high overfitting risk or weak out-of-sample behavior."

    rating = _rating(overall, decision, overfit_risk)
    best_params = dict(best.get("best_params") or _best_params_from_optimization(optimization_result))
    return _scorecard(
        overall_score=overall,
        final_rating=rating,
        decision=decision,
        decision_reason=reason,
        sub_scores=sub_scores,
        flags=sorted(set(flags + list(overfit.get("risk_flags", []) or []))),
        best_params=best_params,
        best_experiment_id=best.get("analysis_id"),
    )


def _candidate_quality_score(result: dict[str, Any]) -> float:
    metrics = dict(result.get("out_of_sample_result", {}) or {})
    stability = dict(result.get("stability_score", {}) or {})
    overfit = dict(result.get("overfit_result", {}) or {})
    return (
        _float(metrics.get("net_CAGR")) * 100
        + _float(metrics.get("sharpe_ratio")) * 8
        + _float(stability.get("stability_score"))
        - _float(overfit.get("overfit_score"))
        + _float(metrics.get("max_drawdown")) * 50
    )


def _best_params_from_optimization(optimization_result: dict[str, Any] | None) -> dict[str, Any]:
    rows = list((optimization_result or {}).get("top_params", []) or [])
    if not rows:
        return {}
    return dict(rows[0].get("params", {}) or rows[0].get("parameters", {}) or {})


def _regime_score(regimes: dict[str, Any]) -> float:
    usable = [
        value
        for key, value in regimes.items()
        if not str(key).startswith("_") and isinstance(value, dict) and int(_float(value.get("trade_count"))) > 0
    ]
    if not usable:
        return 35.0
    positive = sum(1 for item in usable if _float(item.get("net_CAGR")) > 0)
    coverage = min(len(usable) / 4, 1)
    return _clamp(40 + positive / len(usable) * 45 + coverage * 15)


def _rating(overall: float, decision: str, overfit_risk: str) -> str:
    if decision == "reject" or overfit_risk == "HIGH":
        return "D"
    if overall >= 80:
        return "A"
    if overall >= 65:
        return "B"
    if overall >= 50:
        return "C"
    return "D"


def _scorecard(
    overall_score: float,
    final_rating: str,
    decision: str,
    decision_reason: str,
    sub_scores: dict[str, float],
    flags: list[str],
    best_params: dict[str, Any],
    best_experiment_id: str | None,
) -> dict[str, Any]:
    return {
        "overall_score": _clamp(overall_score),
        "final_rating": final_rating,
        "decision": decision,
        "decision_reason": decision_reason,
        "sub_scores": {key: _clamp(value) for key, value in sub_scores.items()},
        "flags": flags,
        "best_params": best_params,
        "best_experiment_id": best_experiment_id,
    }


def _empty_sub_scores() -> dict[str, float]:
    return {
        "return_score": 0,
        "risk_score": 0,
        "stability_score": 0,
        "overfit_score": 0,
        "sample_size_score": 0,
        "cost_score": 0,
        "regime_score": 0,
    }


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, float(value)))
