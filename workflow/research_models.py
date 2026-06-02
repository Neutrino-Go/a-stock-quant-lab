from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ResearchConfig:
    strategy_name: str
    start_date: str | pd.Timestamp
    end_date: str | pd.Timestamp
    research_template: str = "standard"
    parameter_grid: dict[str, list[Any]] = field(default_factory=dict)
    top_n_params: int = 5
    train_ratio: float = 0.7
    walk_forward_config: dict[str, int] = field(default_factory=lambda: {"train_months": 24, "test_months": 6, "step_months": 6})
    cost_config: dict[str, Any] = field(default_factory=dict)
    position_config: dict[str, Any] = field(default_factory=dict)
    exit_config: dict[str, Any] = field(default_factory=dict)
    market_config: dict[str, Any] = field(default_factory=dict)
    decision_thresholds: dict[str, Any] = field(default_factory=dict)
    save_intermediate_experiments: bool = True
    name: str = ""
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchConfig":
        return cls(**dict(payload))


@dataclass
class ResearchStepResult:
    step_name: str
    status: str
    message: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    experiment_id: str | None = None
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchStepResult":
        return cls(**dict(payload))


@dataclass
class ResearchRunResult:
    research_run_id: str
    strategy_name: str
    status: str
    summary: dict[str, Any] = field(default_factory=dict)
    steps: list[ResearchStepResult] = field(default_factory=list)
    scorecard: dict[str, Any] = field(default_factory=dict)
    decision: str = "failed"
    candidate_id: str | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchRunResult":
        data = dict(payload)
        data["steps"] = [ResearchStepResult.from_dict(step) for step in data.get("steps", [])]
        return cls(**data)


@dataclass
class StrategyCandidate:
    id: str
    strategy_name: str
    candidate_name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    scorecard: dict[str, Any] = field(default_factory=dict)
    status: str = "watching"
    source_research_run_id: str | None = None
    source_experiment_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrategyCandidate":
        return cls(**dict(payload))


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [_json_safe(item) for item in value.to_dict("records")]
    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
