from __future__ import annotations

import pandas as pd
import pytest

from stability.splitter import create_train_test_split, create_walk_forward_splits


def test_train_test_split_preserves_time_order_and_no_overlap():
    split = create_train_test_split("2020-01-01", "2020-12-31", train_ratio=0.7)

    assert pd.Timestamp(split["train_start"]) == pd.Timestamp("2020-01-01")
    assert pd.Timestamp(split["test_start"]) > pd.Timestamp(split["train_end"])
    assert pd.Timestamp(split["test_end"]) == pd.Timestamp("2020-12-31")


def test_train_test_split_accepts_timestamp_inputs():
    split = create_train_test_split(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-04-01"), train_ratio=0.5)

    assert pd.Timestamp(split["train_end"]) < pd.Timestamp(split["test_start"])


def test_train_test_split_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="start_date must be earlier"):
        create_train_test_split("2020-12-31", "2020-01-01")

    with pytest.raises(ValueError, match="train_ratio"):
        create_train_test_split("2020-01-01", "2020-12-31", train_ratio=1.0)


def test_walk_forward_splits_roll_forward_without_overlap():
    splits = create_walk_forward_splits(
        "2020-01-01",
        "2022-12-31",
        train_months=12,
        test_months=6,
        step_months=6,
    )

    assert len(splits) >= 3
    for fold in splits:
        assert pd.Timestamp(fold["test_start"]) > pd.Timestamp(fold["train_end"])
        assert pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["test_end"])

    assert pd.Timestamp(splits[1]["train_start"]) > pd.Timestamp(splits[0]["train_start"])


def test_walk_forward_drops_incomplete_final_test_window():
    splits = create_walk_forward_splits(
        "2020-01-01",
        "2020-05-01",
        train_months=3,
        test_months=3,
        step_months=1,
    )

    assert splits == []


def test_walk_forward_rejects_non_positive_month_settings():
    with pytest.raises(ValueError, match="months"):
        create_walk_forward_splits("2020-01-01", "2021-01-01", train_months=0)
