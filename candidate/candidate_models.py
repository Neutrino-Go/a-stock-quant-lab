from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CandidateRevalidationConfig:
    candidate_id: str
    mode: str = "recent_window"
    review_start_date: str | None = None
    review_end_date: str | None = None
    recent_months: int = 6
    use_stability_analysis: bool = True
    use_market_regime: bool = True
    cost_config: dict[str, Any] = field(default_factory=dict)
    position_config: dict[str, Any] = field(default_factory=dict)
    exit_config: dict[str, Any] = field(default_factory=dict)
    walk_forward_config: dict[str, Any] = field(default_factory=dict)
    decision_thresholds: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    save_as_experiment: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateRevalidationConfig":
        payload = dict(data or {})
        if "candidate_id" not in payload:
            payload["candidate_id"] = ""
        values: dict[str, Any] = {}
        for key, field_ in cls.__dataclass_fields__.items():
            if key in payload:
                values[key] = payload[key]
            elif field_.default is not MISSING:
                values[key] = field_.default
            elif field_.default_factory is not MISSING:  # type: ignore[attr-defined]
                values[key] = field_.default_factory()  # type: ignore[misc]
        return cls(**values)


@dataclass(frozen=True)
class CandidateComparison:
    baseline_net_CAGR: float | None = None
    latest_net_CAGR: float | None = None
    net_CAGR_delta: float | None = None
    baseline_max_drawdown: float | None = None
    latest_max_drawdown: float | None = None
    max_drawdown_delta: float | None = None
    baseline_sharpe: float | None = None
    latest_sharpe: float | None = None
    sharpe_delta: float | None = None
    baseline_stability_score: float | None = None
    latest_stability_score: float | None = None
    stability_score_delta: float | None = None
    baseline_overfit_risk: str | None = None
    latest_overfit_risk: str | None = None
    trade_count_delta: float | None = None
    cost_impact_delta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateComparison":
        return cls(**{key: (data or {}).get(key) for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class CandidateRevalidationResult:
    revalidation_id: str
    candidate_id: str
    status: str
    mode: str
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    latest_metrics: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    recommended_action: str = "continue_monitoring"
    action_reason: str = ""
    generated_experiment_id: str | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateRevalidationResult":
        payload = dict(data or {})
        payload.setdefault("revalidation_id", "")
        payload.setdefault("candidate_id", "")
        payload.setdefault("status", "failed")
        payload.setdefault("mode", "recent_window")
        return cls(**{key: payload.get(key) for key in cls.__dataclass_fields__})
