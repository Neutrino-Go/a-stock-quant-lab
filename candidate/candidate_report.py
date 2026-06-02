from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from candidate.candidate_store import CandidateStore


DEFAULT_CANDIDATE_REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "candidates"


def export_revalidation_markdown_report(
    store: CandidateStore,
    revalidation_run_id: str,
    output_dir: str | Path = DEFAULT_CANDIDATE_REPORT_DIR,
) -> Path:
    run = store.get_revalidation_run(revalidation_run_id)
    if run is None:
        raise KeyError(f"Candidate revalidation not found: {revalidation_run_id}")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = directory / f"candidate_revalidation_{run['candidate_id']}_{timestamp}.md"
    path.write_text(_render_markdown(run), encoding="utf-8")
    store.update_revalidation_run(revalidation_run_id, report_path=path)
    return path


def export_revalidation_history_csv(revalidations: pd.DataFrame, output_path: str | Path) -> Path:
    columns = [
        "candidate_name",
        "strategy_name",
        "mode",
        "status",
        "review_start_date",
        "review_end_date",
        "recommended_action",
        "latest_net_CAGR",
        "latest_max_drawdown",
        "latest_overfit_risk",
        "created_at",
    ]
    return _export_csv(revalidations, columns, output_path)


def export_lifecycle_events_csv(events: pd.DataFrame, output_path: str | Path) -> Path:
    columns = ["candidate_name", "event_type", "old_status", "new_status", "message", "created_at"]
    return _export_csv(events, columns, output_path)


def _export_csv(frame: pd.DataFrame, columns: list[str], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = frame.copy() if isinstance(frame, pd.DataFrame) and not frame.empty else pd.DataFrame(columns=columns)
    if "code" in result.columns:
        result["code"] = result["code"].map(lambda value: str(value).strip().zfill(6))
        result["code_text"] = result["code"].map(lambda value: f'="{value}"')
    result.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _render_markdown(run: dict[str, object]) -> str:
    baseline = dict(run.get("baseline_metrics_json", {}) or {})
    latest = dict(run.get("latest_metrics_json", {}) or {})
    comparison = dict(run.get("comparison_json", {}) or {})
    decision = dict(run.get("decision_json", {}) or {})
    flags = decision.get("flags", []) or []
    flag_lines = "\n".join(f"- {flag}" for flag in flags) or "- -"
    return f"""# 候选策略复检报告

## 基本信息
- candidate_id: {run.get("candidate_id")}
- candidate_name: {run.get("candidate_name")}
- strategy_name: {run.get("strategy_name")}
- 当前状态: {run.get("old_candidate_status") or "-"}
- 复检模式: {run.get("mode")}
- 复检区间: {run.get("review_start_date") or "-"} 至 {run.get("review_end_date") or "-"}
- 创建时间: {run.get("created_at")}
- Git Commit: {run.get("git_commit") or "-"}

## Baseline 表现
- net_CAGR: {baseline.get("net_CAGR", "-")}
- max_drawdown: {baseline.get("max_drawdown", "-")}
- sharpe_ratio: {baseline.get("sharpe_ratio", "-")}
- stability_score: {baseline.get("stability_score", "-")}
- overfit_risk: {baseline.get("overfit_risk", "-")}

## Latest 表现
- net_CAGR: {latest.get("net_CAGR", "-")}
- max_drawdown: {latest.get("max_drawdown", "-")}
- sharpe_ratio: {latest.get("sharpe_ratio", "-")}
- stability_score: {latest.get("stability_score", "-")}
- overfit_risk: {latest.get("overfit_risk", "-")}
- total_trades: {latest.get("total_trades", "-")}

## 差异比较
- net_CAGR_delta: {comparison.get("net_CAGR_delta", "-")}
- max_drawdown_delta: {comparison.get("max_drawdown_delta", "-")}
- sharpe_delta: {comparison.get("sharpe_delta", "-")}
- stability_score_delta: {comparison.get("stability_score_delta", "-")}
- overfit_risk 变化: {comparison.get("baseline_overfit_risk", "-")} -> {comparison.get("latest_overfit_risk", "-")}
- trade_count_delta: {comparison.get("trade_count_delta", "-")}

## 生命周期建议
- recommended_action: {run.get("recommended_action")}
- action_reason: {run.get("action_reason") or decision.get("action_reason", "-")}
- risk_level: {decision.get("risk_level", "-")}
- flags:
{flag_lines}

## 是否应用建议
- applied_action: {bool(run.get("applied_action"))}
- old_status: {run.get("old_candidate_status") or "-"}
- new_status: {run.get("new_candidate_status") or "-"}

## 风险提示
- 历史表现不代表未来收益
- 候选策略不是买入建议
- 本工具不构成投资建议
- 本工具不支持自动交易
"""
