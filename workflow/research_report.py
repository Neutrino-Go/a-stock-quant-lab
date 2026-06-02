from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from workflow.research_store import ResearchStore


DEFAULT_RESEARCH_REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "research"


def export_research_markdown_report(
    store: ResearchStore,
    research_run_id: str,
    output_dir: str | Path = DEFAULT_RESEARCH_REPORT_DIR,
) -> Path:
    detail = store.get_research_run_detail(research_run_id)
    if detail["run"] is None:
        raise KeyError(f"Research run not found: {research_run_id}")
    path = _output_path(output_dir, research_run_id, "md")
    path.write_text(_render_markdown(detail), encoding="utf-8")
    store.save_research_artifact(research_run_id, "markdown_report", path)
    return path


def export_research_json_summary(
    store: ResearchStore,
    research_run_id: str,
    output_dir: str | Path = DEFAULT_RESEARCH_REPORT_DIR,
) -> Path:
    detail = store.get_research_run_detail(research_run_id)
    if detail["run"] is None:
        raise KeyError(f"Research run not found: {research_run_id}")
    payload = {
        "run": detail["run"],
        "steps": detail["steps"].to_dict("records"),
        "artifacts": detail["artifacts"].to_dict("records"),
    }
    path = _output_path(output_dir, research_run_id, "summary.json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    store.save_research_artifact(research_run_id, "json_summary", path)
    return path


def export_research_candidates_csv(
    store: ResearchStore,
    research_run_id: str | None,
    candidates: pd.DataFrame,
    output_dir: str | Path = DEFAULT_RESEARCH_REPORT_DIR,
) -> Path:
    columns = [
        "id",
        "strategy_name",
        "candidate_name",
        "status",
        "overall_score",
        "final_rating",
        "stability_score",
        "overfit_risk",
        "net_CAGR",
        "max_drawdown",
        "sharpe_ratio",
        "walk_forward_positive_ratio",
        "parameter_stability_score",
        "created_at",
    ]
    frame = candidates.copy() if not candidates.empty else pd.DataFrame(columns=columns)
    path = _output_path(output_dir, research_run_id or "candidate_library", "candidates.csv")
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    if research_run_id:
        store.save_research_artifact(research_run_id, "csv_candidates", path)
    return path


def _output_path(output_dir: str | Path, research_run_id: str, suffix: str) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return directory / f"research_{research_run_id}_{timestamp}.{suffix}"


def _render_markdown(detail: dict[str, object]) -> str:
    run = detail["run"]
    steps = detail["steps"]
    summary = dict(run.get("summary_json", {}) or {})
    scorecard = dict(summary.get("scorecard", {}) or {})
    core = dict(summary.get("core_metrics", {}) or {})
    best_params = scorecard.get("best_params", {})
    sub_scores = dict(scorecard.get("sub_scores", {}) or {})
    flags = list(scorecard.get("flags", []) or [])
    candidate_id = summary.get("candidate_id") or run.get("best_candidate_id") or "-"
    step_lines = _step_lines(steps)
    flag_lines = "\n".join(f"- {flag}" for flag in flags) or "- -"
    return f"""# 策略研究报告

## 基本信息
- 研究任务名称: {run.get("name")}
- 策略: {run.get("strategy_name")}
- 研究区间: {run.get("start_date") or "-"} 至 {run.get("end_date") or "-"}
- 模板: {run.get("research_template")}
- Git Commit: {run.get("git_commit") or "-"}
- 创建时间: {run.get("created_at")}

## 研究流程摘要
{step_lines}

## 最终评分卡
- overall_score: {scorecard.get("overall_score", "-")}
- final_rating: {scorecard.get("final_rating", "-")}
- decision: {scorecard.get("decision", run.get("decision", "-"))}
- decision_reason: {scorecard.get("decision_reason", run.get("decision_reason", "-"))}

## 最佳参数组合
```json
{json.dumps(best_params, ensure_ascii=False, indent=2)}
```
- best_experiment_id: {scorecard.get("best_experiment_id", "-")}

## 核心指标
- 样本外 net_CAGR: {core.get("out_of_sample_net_CAGR", "-")}
- max_drawdown: {core.get("max_drawdown", "-")}
- sharpe_ratio: {core.get("sharpe_ratio", "-")}
- walk_forward_positive_ratio: {core.get("walk_forward_positive_ratio", "-")}
- parameter_stability_score: {core.get("parameter_stability_score", "-")}
- overfit_risk: {core.get("overfit_risk", "-")}
- cost_impact: {core.get("cost_impact", "-")}

## 子评分
- return_score: {sub_scores.get("return_score", "-")}
- risk_score: {sub_scores.get("risk_score", "-")}
- stability_score: {sub_scores.get("stability_score", "-")}
- overfit_score: {sub_scores.get("overfit_score", "-")}
- sample_size_score: {sub_scores.get("sample_size_score", "-")}
- cost_score: {sub_scores.get("cost_score", "-")}
- regime_score: {sub_scores.get("regime_score", "-")}

## 风险 flags
{flag_lines}

## 候选策略状态
- 是否加入候选策略库: {"yes" if candidate_id != "-" else "no"}
- candidate_id: {candidate_id}

## 研究结论
- decision: {run.get("decision")}
- 原因: {run.get("decision_reason") or scorecard.get("decision_reason", "-")}

## 风险提示
- 历史回测不代表未来收益
- 样本外表现也不能保证未来有效
- 本工具不构成投资建议
- 本工具不支持自动交易
"""


def _step_lines(steps: pd.DataFrame) -> str:
    if steps.empty:
        return "- -"
    return "\n".join(
        f"- {row.get('step_name')}: {row.get('status')} {row.get('message') or ''}".rstrip()
        for _, row in steps.iterrows()
    )
