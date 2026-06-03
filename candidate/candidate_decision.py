from __future__ import annotations

from typing import Any


DEFAULT_DECISION_THRESHOLDS = {
    "min_trades": 30,
    "max_drawdown_limit": -0.30,
    "stability_drop_limit": -15,
    "sharpe_drop_limit": -0.50,
    "near_zero_cagr": 0.02,
    "min_walk_forward_positive_ratio": 0.40,
}


def evaluate_candidate_lifecycle(
    baseline_metrics: dict[str, Any],
    latest_metrics: dict[str, Any],
    comparison: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = {**DEFAULT_DECISION_THRESHOLDS, **dict(thresholds or {})}
    latest_net = _number(latest_metrics.get("net_CAGR"))
    latest_drawdown = _number(latest_metrics.get("max_drawdown"))
    latest_trades = int(_number(latest_metrics.get("total_trades"), 0) or 0)
    latest_overfit = str(latest_metrics.get("overfit_risk") or "").upper()
    latest_wf = _number(latest_metrics.get("walk_forward_positive_ratio"))
    stability_delta = _number(comparison.get("stability_score_delta"))
    sharpe_delta = _number(comparison.get("sharpe_delta"))
    flags: list[str] = []

    min_trades = int(config["min_trades"])
    drawdown_limit = float(config["max_drawdown_limit"])
    stability_drop_limit = float(config["stability_drop_limit"])
    sharpe_drop_limit = float(config["sharpe_drop_limit"])
    near_zero = float(config["near_zero_cagr"])
    wf_limit = float(config["min_walk_forward_positive_ratio"])

    if latest_trades < min_trades:
        flags.append("insufficient_trades")
        return _decision("needs_more_data", "MEDIUM", flags, 45, "最新复检样本数量不足，需要更多数据后再判断生命周期状态。")

    if latest_overfit == "HIGH":
        flags.append("overfit_high")
    if latest_net is not None and latest_net <= 0:
        flags.append("latest_return_non_positive")
    if latest_drawdown is not None and latest_drawdown < drawdown_limit:
        flags.append("drawdown_breached")
    if latest_wf is not None and latest_wf < wf_limit:
        flags.append("walk_forward_ratio_low")

    if (
        latest_overfit == "HIGH"
        or (latest_net is not None and latest_net <= 0 and latest_drawdown is not None and latest_drawdown < drawdown_limit)
        or (latest_wf is not None and latest_wf < wf_limit)
    ):
        return _decision("reject", "HIGH", flags, 20, "最新复检显示收益、回撤、过拟合或滚动验证指标显著恶化，建议从研究候选中淘汰。")

    if stability_delta is not None and stability_delta < stability_drop_limit:
        flags.append("stability_deteriorated")
    if sharpe_delta is not None and sharpe_delta < sharpe_drop_limit:
        flags.append("sharpe_deteriorated")
    if latest_net is not None and 0 < latest_net <= near_zero:
        flags.append("latest_return_near_zero")

    if flags:
        return _decision("downgrade_to_watching", "MEDIUM", flags, 55, "最新复检没有完全失效，但稳定性、收益或夏普指标下降，建议降级为观察。")

    if (
        latest_net is not None
        and latest_net > 0
        and (latest_drawdown is None or latest_drawdown > drawdown_limit)
        and latest_overfit != "HIGH"
        and (stability_delta is None or stability_delta >= stability_drop_limit)
    ):
        return _decision("keep_active", "LOW", flags, 82, "最新复检表现仍满足研究候选要求，建议保持 active 并继续定期复检。")

    return _decision("continue_monitoring", "MEDIUM", flags, 62, "最新复检变化不明显，建议继续监控并等待更多样本。")


def _decision(action: str, risk: str, flags: list[str], score: float, reason: str) -> dict[str, Any]:
    return {
        "recommended_action": action,
        "action_reason": reason,
        "risk_level": risk,
        "flags": flags,
        "decision_score": max(0.0, min(100.0, float(score))),
    }


def _number(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
