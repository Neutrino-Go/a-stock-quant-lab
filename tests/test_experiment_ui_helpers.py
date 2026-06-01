from __future__ import annotations

import pandas as pd

from research.ui_helpers import (
    EXPERIMENT_TYPE_OPTIONS,
    RATING_OPTIONS,
    delete_confirmation_matches,
    experiment_option_label,
    filter_experiment_table,
    parse_tags,
    rating_display_name,
    sort_experiment_table,
    sort_metric_values,
    stability_result_to_frames,
)


def test_parse_tags_accepts_commas_spaces_and_deduplicates():
    assert parse_tags("trend, portfolio  trend\nwatch") == ["trend", "portfolio", "watch"]


def test_parse_tags_handles_empty_values():
    assert parse_tags("") == []
    assert parse_tags(None) == []


def test_parse_tags_accepts_chinese_commas():
    assert parse_tags("alpha，beta,alpha") == ["alpha", "beta"]


def test_rating_display_name_includes_research_meaning():
    assert rating_display_name("A").startswith("A:")
    assert rating_display_name("unknown").startswith("unrated:")
    assert "unrated" in RATING_OPTIONS
    assert "strategy_stability_analysis" in EXPERIMENT_TYPE_OPTIONS


def test_experiment_option_label_contains_name_type_and_date():
    row = {
        "id": "abc123456789",
        "name": "Experiment",
        "experiment_type": "portfolio_backtest",
        "created_at": "2024-01-01T10:00:00",
    }

    label = experiment_option_label(row)

    assert "Experiment" in label
    assert "portfolio_backtest" in label
    assert "abc12345" in label


def test_filter_experiment_table_by_type_strategy_tag_and_rating():
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "experiment_type": "portfolio_backtest",
                "strategy_name": "ensemble_strategy",
                "tags": ["portfolio", "watch"],
                "rating": "A",
                "created_at": "2024-01-01",
            },
            {
                "id": "2",
                "experiment_type": "single_strategy_backtest",
                "strategy_name": "trend_strength_strategy",
                "tags": ["trend"],
                "rating": "D",
                "created_at": "2024-01-02",
            },
        ]
    )

    result = filter_experiment_table(
        df,
        experiment_type="portfolio_backtest",
        strategy_name="ensemble_strategy",
        tag="watch",
        rating="A",
    )

    assert list(result["id"]) == ["1"]


def test_filter_experiment_table_by_created_date_range():
    df = pd.DataFrame(
        [
            {"id": "1", "created_at": "2024-01-01T00:00:00", "tags": [], "rating": "A"},
            {"id": "2", "created_at": "2024-02-01T00:00:00", "tags": [], "rating": "B"},
        ]
    )

    result = filter_experiment_table(df, start_date="2024-01-15", end_date="2024-02-15")

    assert list(result["id"]) == ["2"]


def test_filter_experiment_table_by_keyword_search():
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "name": "Momentum run",
                "description": "bull market",
                "notes": "",
                "conclusion": "",
                "tags": [],
                "rating": "A",
            },
            {
                "id": "2",
                "name": "Pullback run",
                "description": "",
                "notes": "watch liquidity",
                "conclusion": "discard",
                "tags": [],
                "rating": "D",
            },
        ]
    )

    result = filter_experiment_table(df, keyword="liquidity")

    assert list(result["id"]) == ["2"]


def test_sort_metric_values_uses_metric_semantics_and_nulls_last():
    assert sort_metric_values("max_drawdown", [-0.30, None, -0.10, -0.20]) == [-0.10, -0.20, -0.30, None]
    assert sort_metric_values("net_CAGR", [0.05, None, 0.20]) == [0.20, 0.05, None]
    assert sort_metric_values("cost_impact", [0.05, None, 0.01]) == [0.01, 0.05, None]


def test_sort_experiment_table_orders_drawdown_cost_and_dates():
    df = pd.DataFrame(
        [
            {"id": "1", "max_drawdown": -0.30, "cost_impact": 0.05, "created_at": "2024-01-01"},
            {"id": "2", "max_drawdown": -0.10, "cost_impact": 0.01, "created_at": "2024-02-01"},
            {"id": "3", "max_drawdown": None, "cost_impact": None, "created_at": None},
        ]
    )

    assert list(sort_experiment_table(df, "max_drawdown", ascending=False)["id"]) == ["2", "1", "3"]
    assert list(sort_experiment_table(df, "cost_impact", ascending=False)["id"]) == ["2", "1", "3"]
    assert list(sort_experiment_table(df, "created_at", ascending=False)["id"]) == ["2", "1", "3"]


def test_sort_experiment_table_can_reverse_best_first_order():
    df = pd.DataFrame(
        [
            {"id": "1", "net_CAGR": 0.20, "cost_impact": 0.01},
            {"id": "2", "net_CAGR": 0.05, "cost_impact": 0.05},
        ]
    )

    assert list(sort_experiment_table(df, "net_CAGR", ascending=True)["id"]) == ["2", "1"]
    assert list(sort_experiment_table(df, "cost_impact", ascending=True)["id"]) == ["2", "1"]


def test_delete_confirmation_matches_requires_exact_experiment_name():
    assert delete_confirmation_matches("Research Run", "Research Run")
    assert delete_confirmation_matches("Research Run", " Research Run ")
    assert not delete_confirmation_matches("Research Run", "research run")
    assert not delete_confirmation_matches("Research Run", "")


def test_stability_result_to_frames_handles_empty_and_populated_results():
    empty = stability_result_to_frames(None)
    assert empty["walk_forward_folds"].empty
    result = stability_result_to_frames(
        {
            "walk_forward_result": {"folds": [{"fold": 1}]},
            "sensitivity_result": {"parameter_table": [{"rank": 1, "net_CAGR": 0.1}]},
            "market_regime_breakdown": {"BULL": {"trade_count": 3}},
            "overfit_result": {"risk_flags": ["low_trade_count"]},
        }
    )

    assert list(result["walk_forward_folds"]["fold"]) == [1]
    assert list(result["risk_flags"]["risk_flag"]) == ["low_trade_count"]
