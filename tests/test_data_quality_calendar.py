from __future__ import annotations

import pandas as pd

from data.database import get_engine, upsert_stock_daily
from data_quality.calendar import TradingCalendarService


def test_trading_calendar_fallback_from_prices(tmp_path):
    service = TradingCalendarService(tmp_path / "calendar.db")
    prices = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "code": ["000001", "000001"]})

    calendar = service.get_trading_calendar("2024-01-01", "2024-01-05", price_df=prices)

    assert calendar.loc[calendar["date"].eq(pd.Timestamp("2024-01-02")), "is_trading_day"].item()
    assert calendar["source"].eq("inferred_from_prices").any()


def test_previous_next_trading_day(tmp_path):
    service = TradingCalendarService(tmp_path / "calendar.db")
    service.save_calendar(pd.DataFrame({"date": ["2024-01-02", "2024-01-03", "2024-01-05"], "is_trading_day": [1, 1, 1]}))

    assert service.is_trading_day("2024-01-04") is False
    assert service.get_previous_trading_day("2024-01-04") == "2024-01-03"
    assert service.get_next_trading_day("2024-01-04") == "2024-01-05"
    assert service.count_trading_days("2024-01-01", "2024-01-05") == 3


def test_find_missing_trading_days(tmp_path):
    service = TradingCalendarService(tmp_path / "calendar.db")
    service.save_calendar(pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "is_trading_day": [1, 1]}))
    prices = pd.DataFrame({"date": ["2024-01-02"], "code": ["000001"]})

    missing = service.find_missing_trading_days(prices, "000001", "2024-01-02", "2024-01-03")

    assert missing == ["2024-01-03"]


def test_empty_calendar_does_not_crash(tmp_path):
    service = TradingCalendarService(tmp_path / "calendar.db")

    assert service.get_previous_trading_day("2024-01-04") is None
    assert service.get_next_trading_day("2024-01-04") is None


def test_previous_next_trading_day_can_fallback_to_price_dates(tmp_path):
    db_path = tmp_path / "calendar.db"
    upsert_stock_daily(
        pd.DataFrame(
            {
                "code": ["000001", "000001"],
                "name": ["A", "A"],
                "date": ["2024-01-02", "2024-01-05"],
                "open": [10, 10],
                "high": [11, 11],
                "low": [9, 9],
                "close": [10, 10],
                "volume": [1, 1],
                "amount": [1, 1],
                "turnover": [1, 1],
            }
        ),
        engine=get_engine(db_path),
    )
    service = TradingCalendarService(db_path)

    assert service.is_trading_day("2024-01-02") is True
    assert service.get_previous_trading_day("2024-01-04") == "2024-01-02"
    assert service.get_next_trading_day("2024-01-04") == "2024-01-05"


def test_saved_calendar_respects_non_trading_day(tmp_path):
    service = TradingCalendarService(tmp_path / "calendar.db")
    service.save_calendar(
        pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "is_trading_day": [1, 0],
                "source": ["manual", "manual"],
            }
        )
    )

    calendar = service.get_trading_calendar("2024-01-02", "2024-01-03")

    assert service.is_trading_day("2024-01-02") is True
    assert service.is_trading_day("2024-01-03") is False
    assert calendar["source"].eq("manual").all()


def test_missing_days_empty_when_price_has_all_trading_days(tmp_path):
    service = TradingCalendarService(tmp_path / "calendar.db")
    service.save_calendar(pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "is_trading_day": [1, 1]}))
    prices = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "code": ["000001", "000001"]})

    assert service.find_missing_trading_days(prices, "000001", "2024-01-02", "2024-01-03") == []
