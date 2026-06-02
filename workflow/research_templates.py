from __future__ import annotations

from copy import deepcopy
from typing import Any


TEMPLATES: dict[str, dict[str, Any]] = {
    "quick": {
        "research_template": "quick",
        "top_n_params": 3,
        "train_ratio": 0.7,
        "walk_forward_config": {"train_months": 18, "test_months": 6, "step_months": 6},
        "description": "Fast first-pass strategy research.",
    },
    "standard": {
        "research_template": "standard",
        "top_n_params": 5,
        "train_ratio": 0.7,
        "walk_forward_config": {"train_months": 24, "test_months": 6, "step_months": 6},
        "description": "Default research workflow.",
    },
    "deep": {
        "research_template": "deep",
        "top_n_params": 10,
        "train_ratio": 0.7,
        "walk_forward_config": {"train_months": 36, "test_months": 6, "step_months": 3},
        "description": "Deeper parameter and walk-forward research.",
    },
    "custom": {
        "research_template": "custom",
        "top_n_params": 5,
        "train_ratio": 0.7,
        "walk_forward_config": {"train_months": 24, "test_months": 6, "step_months": 6},
        "description": "User-defined workflow settings.",
    },
}


def get_research_template(name: str) -> dict[str, Any]:
    if name not in TEMPLATES:
        raise ValueError(f"Unknown research template: {name}")
    return deepcopy(TEMPLATES[name])
