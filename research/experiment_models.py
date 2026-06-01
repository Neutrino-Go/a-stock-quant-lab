from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ExperimentType = Literal[
    "single_strategy_backtest",
    "portfolio_backtest",
    "parameter_optimization",
    "strategy_leaderboard",
    "market_regime_analysis",
    "strategy_stability_analysis",
]

ExperimentStatus = Literal["completed", "failed", "draft"]
ExperimentRating = Literal["A", "B", "C", "D", "unrated"]

EXPERIMENT_TYPES = [
    "single_strategy_backtest",
    "portfolio_backtest",
    "parameter_optimization",
    "strategy_leaderboard",
    "market_regime_analysis",
    "strategy_stability_analysis",
]

EXPERIMENT_STATUSES = ["completed", "failed", "draft"]
EXPERIMENT_RATINGS = ["A", "B", "C", "D", "unrated"]


@dataclass(frozen=True)
class ExperimentRun:
    id: str
    name: str
    experiment_type: str
    status: str = "completed"
    created_at: str = ""
    updated_at: str = ""
    start_date: str | None = None
    end_date: str | None = None
    strategy_name: str | None = None
    description: str = ""
    notes: str = ""
    conclusion: str = ""
    tags: list[str] = field(default_factory=list)
    rating: str = "unrated"
    git_commit: str = ""
    data_snapshot_info: dict[str, Any] = field(default_factory=dict)
    config_json: dict[str, Any] = field(default_factory=dict)
    metrics_json: dict[str, Any] = field(default_factory=dict)
