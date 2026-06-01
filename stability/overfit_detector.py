from __future__ import annotations

from typing import Any


class OverfitDetector:
    def detect(
        self,
        in_sample_metrics: dict[str, Any],
        out_of_sample_metrics: dict[str, Any],
        walk_forward_summary: dict[str, Any],
        sensitivity_result: dict[str, Any],
        trade_count: int,
        parameter_count: int,
    ) -> dict[str, Any]:
        flags: list[str] = []
        score = 0

        in_cagr = _metric(in_sample_metrics, "net_CAGR")
        out_cagr = _metric(out_of_sample_metrics, "net_CAGR")
        in_sharpe = _metric(in_sample_metrics, "sharpe_ratio")
        out_sharpe = _metric(out_of_sample_metrics, "sharpe_ratio")
        in_drawdown = _metric(in_sample_metrics, "max_drawdown")
        out_drawdown = _metric(out_of_sample_metrics, "max_drawdown")
        walk_positive_ratio = _metric(walk_forward_summary, "positive_test_fold_ratio")
        sensitivity_score = _metric(sensitivity_result, "sensitivity_score")
        gross_oos = _metric(out_of_sample_metrics, "gross_CAGR", fallback="CAGR")
        cost_impact = _metric(out_of_sample_metrics, "cost_impact")

        if in_cagr > 0.15 and out_cagr < in_cagr * 0.5:
            flags.append("sample_performance_collapse")
            score += 18
        if in_sharpe > 1.5 and out_sharpe < 0.5:
            flags.append("sharpe_collapse")
            score += 16
        if out_cagr <= 0:
            flags.append("out_of_sample_non_positive")
            score += 18
        if walk_positive_ratio < 0.5:
            flags.append("walk_forward_low_positive_ratio")
            score += 14
        if sensitivity_score < 40:
            flags.append("parameter_sensitivity_high")
            score += 12
        if trade_count < 30:
            flags.append("low_trade_count")
            score += 10
        if parameter_count > 0 and trade_count < parameter_count * 3:
            flags.append("too_many_parameters_for_sample")
            score += 8
        if out_drawdown < in_drawdown - 0.10:
            flags.append("out_of_sample_drawdown_expansion")
            score += 8
        if gross_oos > 0 and out_cagr <= 0 and cost_impact > 0:
            flags.append("cost_impact_turns_oos_negative")
            score += 10

        overfit_score = int(max(0, min(100, score)))
        if overfit_score >= 70:
            risk = "HIGH"
            recommendation = "reject_for_now"
        elif overfit_score >= 35:
            risk = "MEDIUM"
            recommendation = "watch_carefully"
        else:
            risk = "LOW"
            recommendation = "continue_research"

        return {
            "overfit_risk": risk,
            "overfit_score": overfit_score,
            "risk_flags": flags,
            "explanation": _explanation(flags),
            "recommendation": recommendation,
        }


def _metric(metrics: dict[str, Any], key: str, fallback: str | None = None) -> float:
    value = metrics.get(key)
    if value is None and fallback is not None:
        value = metrics.get(fallback)
    if value is None and key == "net_CAGR":
        value = metrics.get("CAGR", metrics.get("annual_return", 0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _explanation(flags: list[str]) -> str:
    if not flags:
        return "No major overfitting flags were detected."
    return "Detected overfitting risk flags: " + ", ".join(flags)
