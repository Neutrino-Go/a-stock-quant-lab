from __future__ import annotations

import pandas as pd

from research.ui_helpers import (
    RATING_OPTIONS,
    experiment_option_label,
    filter_experiment_table,
    parse_tags,
    rating_display_name,
)


def test_parse_tags_accepts_commas_spaces_and_deduplicates():
    assert parse_tags("trend, portfolio  trend\nwatch") == ["trend", "portfolio", "watch"]


def test_rating_display_name_includes_research_meaning():
    assert rating_display_name("A") == "A：值得继续研究"
    assert rating_display_name("unknown") == "unrated：未评级"
    assert "unrated" in RATING_OPTIONS


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


def test_parse_tags_handles_empty_values():
    assert parse_tags("") == []
    assert parse_tags(None) == []
