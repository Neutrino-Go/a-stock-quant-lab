from datetime import date

import data.data_updater as updater
from data.data_updater import DataUpdateService


def test_update_stock_daily_skips_when_local_data_is_current(monkeypatch):
    service = object.__new__(DataUpdateService)
    service.engine = object()
    today = date.today().strftime("%Y-%m-%d")

    monkeypatch.setattr(updater, "get_latest_trade_date", lambda code, engine: today)

    def fail_fetch(*args, **kwargs):
        raise AssertionError("fetch_stock_daily should not be called")

    monkeypatch.setattr(updater, "fetch_stock_daily", fail_fetch)

    assert service.update_stock_daily("000001") == 0
