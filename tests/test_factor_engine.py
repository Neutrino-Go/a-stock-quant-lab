import numpy as np
import pandas as pd

from factor.factor_engine import compute_factors


def _sample_daily_frame(days: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=days)
    close = pd.Series(np.arange(1, days + 1, dtype=float) + 10)
    return pd.DataFrame(
        {
            "code": "000001",
            "name": "平安银行",
            "date": dates,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.arange(1000, 1000 + days, dtype=float),
            "amount": 300_000_000.0,
            "turnover": 1.5,
        }
    )


def test_compute_factors_uses_current_and_past_rows_only():
    source = _sample_daily_frame()

    original = compute_factors(source)
    changed_future = source.copy()
    changed_future.loc[40:, "close"] = changed_future.loc[40:, "close"] * 100
    recomputed = compute_factors(changed_future)

    check_date = source.loc[30, "date"]
    original_row = original.loc[original["date"] == check_date].iloc[0]
    recomputed_row = recomputed.loc[recomputed["date"] == check_date].iloc[0]

    columns = [
        "ma5",
        "ma10",
        "ma20",
        "ma60",
        "return_5d",
        "return_10d",
        "return_20d",
        "volume_ma20",
        "volume_ratio",
        "volatility_20d",
        "breakout_20d",
        "breakout_60d",
        "distance_to_ma20",
        "max_drawdown_20d",
    ]
    pd.testing.assert_series_equal(
        original_row[columns],
        recomputed_row[columns],
        check_names=False,
    )


def test_compute_factors_calculates_required_columns():
    source = _sample_daily_frame()
    result = compute_factors(source)

    row_20 = result.iloc[19]
    assert row_20["ma20"] == np.mean(source["close"].iloc[:20])
    assert row_20["volume_ma20"] == np.mean(source["volume"].iloc[:20])
    assert bool(row_20["breakout_20d"]) is True

    row_60 = result.iloc[59]
    assert row_60["ma60"] == np.mean(source["close"].iloc[:60])
    assert bool(row_60["ma_bullish"]) is True

    row_21 = result.iloc[20]
    expected_return_20d = source.loc[20, "close"] / source.loc[0, "close"] - 1
    assert row_21["return_20d"] == expected_return_20d
    assert row_21["distance_to_ma20"] == row_21["close"] / row_21["ma20"] - 1
    assert row_21["max_drawdown_20d"] == 0


def test_compute_factors_handles_multiple_codes_independently():
    first = _sample_daily_frame(30)
    second = _sample_daily_frame(30)
    second["code"] = "000002"
    second["name"] = "万科A"
    second["close"] = second["close"] * 2
    source = pd.concat([first, second], ignore_index=True).sample(frac=1, random_state=7)

    result = compute_factors(source)

    first_day_second_code = result[result["code"] == "000002"].sort_values("date").iloc[0]
    assert pd.isna(first_day_second_code["return_5d"])
    assert first_day_second_code["ma5"] == first_day_second_code["close"]
