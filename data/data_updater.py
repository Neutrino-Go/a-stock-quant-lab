from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import pandas as pd
from sqlalchemy.engine import Engine

from config import DEFAULT_START_DATE
from data.data_fetcher import fetch_stock_daily, fetch_stock_list
from data.database import (
    get_engine,
    get_latest_trade_date,
    get_stock_list,
    get_stock_name,
    init_db,
    upsert_stock_daily,
    upsert_stock_list,
)


def _ak_date(value: str | pd.Timestamp) -> str:
    return pd.to_datetime(value).strftime("%Y%m%d")


@dataclass
class DataUpdateService:
    engine: Engine | None = None

    def __post_init__(self) -> None:
        self.engine = self.engine or get_engine()
        init_db(self.engine)

    def update_stock_list(self) -> int:
        stock_list = fetch_stock_list()
        return upsert_stock_list(stock_list, self.engine)

    def update_stock_daily(
        self,
        code: str,
        name: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        force: bool = False,
    ) -> int:
        latest = get_latest_trade_date(code, self.engine)
        if not force and latest:
            start = pd.to_datetime(latest) + timedelta(days=1)
            start_date = _ak_date(start)
        else:
            start_date = start_date or DEFAULT_START_DATE

        effective_end = end_date or date.today().strftime("%Y%m%d")
        if pd.to_datetime(start_date) > pd.to_datetime(effective_end):
            return 0

        resolved_name = name or get_stock_name(code, self.engine) or ""
        daily = fetch_stock_daily(
            code=code,
            name=resolved_name,
            start_date=start_date,
            end_date=effective_end,
        )
        return upsert_stock_daily(daily, self.engine)

    def update_batch(
        self,
        codes: Iterable[str] | None = None,
        limit: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        if codes is None:
            stocks = get_stock_list(self.engine)
            if stocks.empty:
                self.update_stock_list()
                stocks = get_stock_list(self.engine)
            code_name_pairs = stocks.loc[:, ["code", "name"]].to_dict("records")
        else:
            code_name_pairs = [
                {"code": code, "name": get_stock_name(code, self.engine)}
                for code in codes
            ]

        if limit:
            code_name_pairs = code_name_pairs[:limit]

        results = []
        for item in code_name_pairs:
            code = item["code"]
            try:
                rows = self.update_stock_daily(
                    code=code,
                    name=item.get("name"),
                    start_date=start_date,
                    end_date=end_date,
                    force=force,
                )
                results.append(
                    {
                        "code": code,
                        "name": item.get("name") or "",
                        "rows": rows,
                        "status": "success",
                        "message": "",
                    }
                )
            except Exception as exc:  # pragma: no cover - network/data source variability
                results.append(
                    {
                        "code": code,
                        "name": item.get("name") or "",
                        "rows": 0,
                        "status": "failed",
                        "message": str(exc),
                    }
                )
        return pd.DataFrame(results)
