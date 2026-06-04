from __future__ import annotations

import pandas as pd

from data_quality.validators import DataValidator


def _valid_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": "000001",
                "name": "Ping An",
                "date": "2024-01-02",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
                "amount": 1000,
                "turnover": 1,
                "adjust_type": "qfq",
            }
        ]
    )


def test_invalid_ohlc_and_negative_values_are_detected():
    df = _valid_rows()
    df.loc[0, "high"] = 9
    df.loc[0, "volume"] = -1

    issues = DataValidator().validate_daily_prices(df)

    assert {issue.issue_type for issue in issues} >= {"invalid_ohlc", "negative_volume"}


def test_duplicate_price_row_detected():
    df = pd.concat([_valid_rows(), _valid_rows()], ignore_index=True)

    issues = DataValidator().validate_daily_prices(df)

    assert any(issue.issue_type == "duplicate_price_row" for issue in issues)


def test_abnormal_return_detected():
    df = pd.concat([_valid_rows(), _valid_rows()], ignore_index=True)
    df.loc[1, "date"] = "2024-01-03"
    df.loc[1, "close"] = 20

    issues = DataValidator(abnormal_return_threshold=0.35).validate_daily_prices(df)

    assert any(issue.issue_type == "abnormal_return" for issue in issues)


def test_adjust_type_mixed_and_empty_dataframe_do_not_crash():
    df = pd.concat([_valid_rows(), _valid_rows()], ignore_index=True)
    df.loc[1, "date"] = "2024-01-03"
    df.loc[1, "adjust_type"] = "raw"

    issues = DataValidator().validate_daily_prices(df)

    assert any(issue.issue_type == "adjust_type_mismatch" for issue in issues)
    assert DataValidator().validate_daily_prices(pd.DataFrame()) == []


def test_missing_required_fields_detected():
    df = _valid_rows().drop(columns=["close"])

    issues = DataValidator().validate_daily_prices(df)

    assert any(issue.issue_type == "missing_required_field" and issue.field == "close" for issue in issues)


def test_unsorted_data_detected():
    df = pd.concat([_valid_rows(), _valid_rows()], ignore_index=True)
    df.loc[0, "date"] = "2024-01-03"
    df.loc[1, "date"] = "2024-01-02"

    issues = DataValidator().validate_daily_prices(df)

    assert any(issue.issue_type == "data_not_sorted" for issue in issues)


def test_negative_amount_and_turnover_detected():
    df = _valid_rows()
    df.loc[0, "amount"] = -100
    df.loc[0, "turnover"] = -1

    issues = DataValidator().validate_daily_prices(df)

    assert {issue.issue_type for issue in issues} >= {"negative_amount", "negative_turnover"}


def test_zero_volume_detected():
    df = _valid_rows()
    df["volume"] = 0

    issues = DataValidator().validate_daily_prices(df)

    assert any(issue.issue_type == "zero_volume" for issue in issues)


def test_stale_data_detected():
    df = pd.concat([_valid_rows()] * 4, ignore_index=True)
    df["date"] = pd.date_range("2024-01-02", periods=4)
    df["volume"] = [100, 100, 100, 100]

    issues = DataValidator(stale_days=3).validate_daily_prices(df)

    assert any(issue.issue_type == "stale_data" for issue in issues)


def test_missing_adjust_type_warns_instead_of_assuming_requested_type():
    df = _valid_rows().drop(columns=["adjust_type"])

    issues = DataValidator().validate_daily_prices(df, adjust_type="qfq")

    legacy = [issue for issue in issues if issue.issue_type == "adjust_type_missing_legacy"]
    assert legacy
    assert legacy[0].details["requested_adjust_type"] == "qfq"
    assert legacy[0].details["actual_adjust_type"] == "unknown"
