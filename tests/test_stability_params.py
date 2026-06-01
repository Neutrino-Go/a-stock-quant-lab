from __future__ import annotations

import inspect

import pandas as pd
import pytest

from ranking.ranking_engine import STRATEGIES
from research.experiment_service import ExperimentService
from research.experiment_store import ExperimentStore
from stability.parameter_sensitivity import ParameterSensitivityAnalyzer
from stability.stability_service import (
    DEFAULT_STABILITY_PARAMETER_GRIDS,
    StrategyStabilityService,
    normalize_strategy_params,
    validate_parameter_grid_for_strategy,
)


def test_default_stability_grids_only_include_supported_or_mapped_params():
    for strategy_name, grid in DEFAULT_STABILITY_PARAMETER_GRIDS.items():
        validate_parameter_grid_for_strategy(strategy_name, grid)
        for param_name in grid:
            normalized = normalize_strategy_params(strategy_name, {param_name: grid[param_name][0]})
            signature = inspect.signature(STRATEGIES[strategy_name])
            allowed = set(signature.parameters) - {"factor_df", "signal_date"}
            assert set(normalized).issubset(allowed)


def test_unknown_stability_parameter_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Unsupported stability parameter"):
        validate_parameter_grid_for_strategy("trend_strength_strategy", {"not_real": [1]})

    with pytest.raises(ValueError, match="Unsupported stability parameter"):
        normalize_strategy_params("pullback_ma20_strategy", {"volume_ratio_max": 1.2})

    with pytest.raises(ValueError, match="Unsupported stability parameter"):
        StrategyStabilityService(
            experiment_service=ExperimentService(ExperimentStore(tmp_path / "experiments.db")),
            factor_data_loader=lambda **kwargs: pd.DataFrame(),
        ).run_stability_analysis(
            strategy_name="volume_breakout_strategy",
            parameter_grid={"return_min": [0.03]},
            start_date="2021-01-01",
            end_date="2021-12-31",
        )


def test_normalize_strategy_params_maps_aliases_to_strategy_signature():
    assert normalize_strategy_params(
        "trend_strength_strategy",
        {"min_amount": 100_000_000, "return_20d": 0.1},
    ) == {"min_amount": 100_000_000, "min_return_20d": 0.1}
    assert normalize_strategy_params("pullback_ma20_strategy", {"distance_to_ma20": 0.03}) == {
        "max_distance": 0.03
    }
    assert normalize_strategy_params("volume_breakout_strategy", {"volume_ratio": 1.8}) == {
        "min_volume_ratio": 1.8
    }
    assert normalize_strategy_params("multi_factor_score_strategy", {"top_n": 10, "score_threshold": 70}) == {
        "top_n": 10,
        "min_score": 70,
    }


def test_parameter_sensitivity_exposes_only_valid_parameter_columns():
    grid = DEFAULT_STABILITY_PARAMETER_GRIDS["volume_breakout_strategy"]
    metrics = pd.DataFrame(
        [
            {"volume_ratio": 1.5, "net_CAGR": 0.10},
            {"volume_ratio": 1.8, "net_CAGR": 0.08},
        ]
    )

    result = ParameterSensitivityAnalyzer(
        strategy_name="volume_breakout_strategy",
        parameter_grid=grid,
        metrics_by_param=metrics,
    ).analyze()

    assert result["param_columns"] == ["volume_ratio"]
    assert "return_min" not in result["param_columns"]
    assert "return_max" not in result["param_columns"]
