from __future__ import annotations

from typing import Any

import numpy as np


def calculate_stability_score(
    in_sample_metrics: dict[str, Any],
    out_of_sample_metrics: dict[str, Any],
    walk_forward_summary: dict[str, Any],
    sensitivity_result: dict[str, Any],
    market_regime_breakdown: dict[str, dict[str, Any]],
    overfit_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overfit_result = overfit_result or {}
    generalization = _generalization_score(in_sample_metrics, out_of_sample_metrics)
    walk_forward = _bounded(_metric(walk_forward_summary, "walk_forward_score", 50))
    if walk_forward == 0 and "positive_test_fold_ratio" in walk_forward_summary:
        walk_forward = _bounded(_metric(walk_forward_summary, "positive_test_fold_ratio") * 100)
    parameter = _bounded(_metric(sensitivity_result, "sensitivity_score", 50))
    regime = _regime_score(market_regime_breakdown)
    trade_sample = _trade_sample_score(out_of_sample_metrics, in_sample_metrics)
    stability = (
        generalization * 0.30
        + walk_forward * 0.25
        + parameter * 0.20
        + regime * 0.15
        + trade_sample * 0.10
    )

    overfit_risk = str(overfit_result.get("overfit_risk", "MEDIUM"))
    out_cagr = _metric(out_of_sample_metrics, "net_CAGR")
    rating = _rating(stability, overfit_risk, out_cagr)
    return {
        "stability_score": float(_bounded(stability)),
        "generalization_score": float(generalization),
        "walk_forward_score": float(walk_forward),
        "parameter_stability_score": float(parameter),
        "regime_stability_score": float(regime),
        "trade_sample_score": float(trade_sample),
        "final_rating": rating,
    }


def _generalization_score(in_sample: dict[str, Any], out_sample: dict[str, Any]) -> float:
    in_cagr = _metric(in_sample, "net_CAGR")
    out_cagr = _metric(out_sample, "net_CAGR")
    if in_cagr <= 0 and out_cagr <= 0:
        cagr_score = 30.0
    elif in_cagr <= 0 < out_cagr:
        cagr_score = 80.0
    else:
        cagr_score = _bounded((out_cagr / max(abs(in_cagr), 1e-9)) * 100)

    in_drawdown = _metric(in_sample, "max_drawdown")
    out_drawdown = _metric(out_sample, "max_drawdown")
    if out_drawdown >= in_drawdown:
        drawdown_score = 100.0
    else:
        drawdown_score = _bounded(100.0 - abs(out_drawdown - in_drawdown) * 200.0)
    return _bounded(cagr_score * 0.7 + drawdown_score * 0.3)


def _regime_score(regimes: dict[str, dict[str, Any]]) -> float:
    if not regimes:
        return 50.0
    usable = [
        _metric(metrics, "net_CAGR")
        for metrics in regimes.values()
        if _metric(metrics, "trade_count") >= 5
    ]
    if not usable:
        return 50.0
    positive_ratio = float(np.mean([value > 0 for value in usable]))
    dispersion = float(np.std(usable)) if len(usable) > 1 else 0.0
    return _bounded(positive_ratio * 80.0 + max(0.0, 20.0 - dispersion * 100.0))


def _trade_sample_score(out_sample: dict[str, Any], in_sample: dict[str, Any]) -> float:
    trades = max(_metric(out_sample, "total_trades"), _metric(in_sample, "total_trades"))
    if trades >= 80:
        return 100.0
    if trades >= 50:
        return 85.0
    if trades >= 30:
        return 70.0
    if trades >= 10:
        return 45.0
    return 20.0


def _rating(stability_score: float, overfit_risk: str, out_cagr: float) -> str:
    if overfit_risk == "HIGH" or stability_score < 50:
        return "D"
    if stability_score >= 80 and overfit_risk == "LOW" and out_cagr > 0:
        return "A"
    if stability_score >= 65 and overfit_risk != "HIGH":
        return "B"
    if stability_score >= 50:
        return "C"
    return "D"


def _metric(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key)
    if value is None and key == "net_CAGR":
        value = metrics.get("CAGR", metrics.get("annual_return", default))
    if value is None:
        value = default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _bounded(value: float) -> float:
    return max(0.0, min(100.0, float(value)))
