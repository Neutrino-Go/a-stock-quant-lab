from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_quality.audit import DataQualityAuditor
from data_quality.models import now_iso


DEFAULT_REPORT_DIR = Path("reports/data_quality")


def export_data_quality_markdown_report(
    auditor: DataQualityAuditor,
    run_id: str,
    output_dir: str | Path = DEFAULT_REPORT_DIR,
) -> Path:
    run = auditor.get_run(run_id) or {}
    issues = auditor.list_issues(run_id=run_id)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"data_quality_{run_id}_{now_iso().replace(':', '').replace('+', '_')}.md"
    summary = dict(run.get("summary_json", {}) or {})
    critical = issues.loc[issues["severity"].eq("CRITICAL")] if not issues.empty and "severity" in issues else pd.DataFrame()
    lines = [
        "# 数据质量报告",
        "",
        "## 基本信息",
        f"- run_id: {run_id}",
        f"- scope: {run.get('scope', '')}",
        f"- start_date: {run.get('start_date', '')}",
        f"- end_date: {run.get('end_date', '')}",
        f"- adjust_type: {run.get('adjust_type', '')}",
        f"- created_at: {run.get('created_at', '')}",
        "",
        "## 摘要",
        f"- total_issues: {summary.get('total_issues', 0)}",
        f"- critical: {summary.get('critical_issues', 0)}",
        f"- error: {summary.get('error_issues', 0)}",
        f"- warning: {summary.get('warning_issues', 0)}",
        f"- info: {summary.get('info_issues', 0)}",
        "",
        "## 数据覆盖",
        f"- 股票数: {summary.get('codes_checked', 0)}",
        f"- 行情记录数: {summary.get('records_checked', 0)}",
        f"- 数据开始日期: {run.get('start_date', '')}",
        f"- 数据结束日期: {run.get('end_date', '')}",
        "",
        "## 主要问题类型",
    ]
    if issues.empty:
        lines.append("- 暂无问题记录")
    else:
        for issue_type, count in issues["issue_type"].value_counts().items():
            lines.append(f"- {issue_type}: {count}")
    lines.extend(["", "## 严重问题明细"])
    if critical.empty:
        lines.append("- 暂无 CRITICAL 问题")
    else:
        for row in critical.to_dict("records"):
            lines.append(f"- {row.get('code', '')} {row.get('date', '')}: {row.get('message', '')}")
    lines.extend(
        [
            "",
            "## 股票池偏差提示",
            "- 当前股票列表 fallback 可能引入幸存者偏差",
            "",
            "## 复权模式提示",
            "- raw / qfq / hfq 数据不可混用比较",
            "",
            "## 风险说明",
            "- 数据质量问题会影响回测结果",
            "- 当前股票列表 fallback 可能引入幸存者偏差",
            "- 不同复权模式结果不可直接比较",
            "- 本工具不构成投资建议",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_data_quality_issues_csv(issues: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = issues.copy() if isinstance(issues, pd.DataFrame) and not issues.empty else pd.DataFrame(
        columns=["id", "run_id", "severity", "issue_type", "code", "name", "date", "field", "message", "details_json", "created_at"]
    )
    if "code" in frame.columns:
        frame["code"] = frame["code"].map(lambda value: "" if pd.isna(value) else str(value).strip().zfill(6))
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path
