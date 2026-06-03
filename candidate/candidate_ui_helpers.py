from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from candidate.candidate_models import CandidateRevalidationConfig


ACTION_STATUS_MAP = {
    "keep_active": "active",
    "downgrade_to_watching": "watching",
    "reject": "rejected",
    "archive": "archived",
}


def action_to_candidate_status(action: str, current_status: str) -> str:
    return ACTION_STATUS_MAP.get(action, current_status)


def status_change_confirmation_matches(candidate_name: str | None, confirmation: str | None) -> bool:
    expected = str(candidate_name or "").strip()
    actual = str(confirmation or "").strip()
    return bool(expected) and expected == actual


def candidate_option_label(candidate: dict[str, object] | pd.Series) -> str:
    data = candidate.to_dict() if isinstance(candidate, pd.Series) else dict(candidate or {})
    candidate_id = str(data.get("id") or "")
    return (
        f"{data.get('candidate_name', '')} | {data.get('strategy_name', '')} | "
        f"{data.get('status', '')} | {candidate_id[:8]}"
    )


def format_revalidation_mode(mode: str) -> str:
    return {
        "forward_only": "forward_only（新增样本验证）",
        "recent_window": "recent_window（最近窗口复检）",
        "full_retest": "full_retest（当前完整数据重新测试）",
    }.get(mode, mode)


def revalidation_summary_metrics(revalidations: pd.DataFrame) -> dict[str, object]:
    if not isinstance(revalidations, pd.DataFrame) or revalidations.empty:
        return {"total_revalidations": 0, "last_revalidation_at": "-", "latest_recommended_action": "-"}
    sorted_frame = revalidations.sort_values("created_at", ascending=False)
    latest = sorted_frame.iloc[0]
    return {
        "total_revalidations": int(len(sorted_frame)),
        "last_revalidation_at": latest.get("created_at", "-"),
        "latest_recommended_action": latest.get("recommended_action", "-"),
    }


def run_batch_revalidation(revalidator, configs: Iterable[CandidateRevalidationConfig]) -> list[dict[str, object]]:
    rows = []
    for config in configs:
        try:
            result = revalidator.run_revalidation(config)
            payload = result.to_dict()
            payload["error_message"] = "" if result.status != "failed" else result.action_reason
        except Exception as exc:
            payload = {
                "candidate_id": config.candidate_id,
                "status": "failed",
                "recommended_action": "continue_monitoring",
                "latest_metrics": {},
                "error_message": f"{type(exc).__name__}: {exc}",
            }
        rows.append(payload)
    return rows


def write_download_bytes(path: Path) -> bytes:
    return path.read_bytes()
