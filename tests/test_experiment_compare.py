from __future__ import annotations

import pandas as pd

from research.experiment_compare import compare_experiments
from research.experiment_store import ExperimentStore


def _seed_experiment(store: ExperimentStore, name: str, metrics: dict[str, float], equity: list[float]) -> str:
    experiment_id = store.create_experiment_run(
        name=name,
        experiment_type="portfolio_backtest",
        metrics=metrics,
    )
    store.save_equity_curve(
        experiment_id,
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=len(equity), freq="D"),
                "total_equity": equity,
                "gross_total_equity": [value * 1.001 for value in equity],
                "drawdown": [0.0] + [min(0.0, equity[idx] / max(equity[: idx + 1]) - 1) for idx in range(1, len(equity))],
            }
        ),
    )
    store.save_trades(
        experiment_id,
        pd.DataFrame(
            {
                "code": ["000001"],
                "net_return": [metrics.get("net_total_return", 0)],
                "commission": [10.0],
                "stamp_tax": [5.0],
                "slippage_cost": [3.0],
                "total_cost": [18.0],
            }
        ),
    )
    return experiment_id


def test_compare_two_experiments_outputs_metric_and_curve_tables(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    first = _seed_experiment(
        store,
        "A",
        {"final_equity": 110, "net_CAGR": 0.10, "max_drawdown": -0.10, "sharpe_ratio": 1.0, "total_cost": 18},
        [100, 110],
    )
    second = _seed_experiment(
        store,
        "B",
        {"final_equity": 105, "net_CAGR": 0.08, "max_drawdown": -0.30, "sharpe_ratio": 0.8, "total_cost": 9},
        [100, 95, 105],
    )

    result = compare_experiments([first, second], store=store)

    assert len(result["metrics_comparison"]) == 2
    assert {"date", "A", "B"}.issubset(result["normalized_equity_curves"].columns)
    assert {"date", "A", "B"}.issubset(result["drawdown_curves"].columns)
    assert result["winner_by_metric"]["net_CAGR"]["experiment_id"] == first
    assert result["winner_by_metric"]["max_drawdown"]["experiment_id"] == first
    assert result["winner_by_metric"]["total_cost"]["experiment_id"] == second


def test_equity_curves_are_normalized_from_one(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    first = _seed_experiment(store, "A", {"net_CAGR": 0.1}, [200, 220])
    second = _seed_experiment(store, "B", {"net_CAGR": 0.2}, [50, 75])

    result = compare_experiments([first, second], store=store)
    curves = result["normalized_equity_curves"].sort_values("date")

    assert curves.iloc[0]["A"] == 1.0
    assert curves.iloc[0]["B"] == 1.0
    assert round(curves.iloc[-1]["A"], 2) == 1.10
    assert round(curves.iloc[-1]["B"], 2) == 1.50


def test_compare_handles_empty_curves_without_error(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    first = store.create_experiment_run(name="Empty A", experiment_type="portfolio_backtest", metrics={"net_CAGR": 0.0})
    second = store.create_experiment_run(name="Empty B", experiment_type="portfolio_backtest", metrics={"net_CAGR": 0.1})

    result = compare_experiments([first, second], store=store)

    assert len(result["metrics_comparison"]) == 2
    assert result["normalized_equity_curves"].empty


def test_compare_requires_two_to_five_experiments(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    first = store.create_experiment_run(name="Only", experiment_type="portfolio_backtest")

    try:
        compare_experiments([first], store=store)
    except ValueError as exc:
        assert "2 to 5" in str(exc)
    else:
        raise AssertionError("compare_experiments should reject fewer than two ids")


def test_compare_rejects_more_than_five_experiments(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    ids = [store.create_experiment_run(name=f"Run {idx}", experiment_type="portfolio_backtest") for idx in range(6)]

    try:
        compare_experiments(ids, store=store)
    except ValueError as exc:
        assert "2 to 5" in str(exc)
    else:
        raise AssertionError("compare_experiments should reject more than five ids")


def test_cost_comparison_sums_trade_cost_components(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    first = _seed_experiment(store, "A", {"net_CAGR": 0.1}, [100, 110])
    second = _seed_experiment(store, "B", {"net_CAGR": 0.2}, [100, 120])

    result = compare_experiments([first, second], store=store)

    assert set(result["cost_comparison"]["commission"]) == {10.0}
    assert set(result["cost_comparison"]["stamp_tax"]) == {5.0}
    assert set(result["cost_comparison"]["slippage_cost"]) == {3.0}
