from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DATABASE_PATH
from data.database import get_engine, get_stock_daily
from data_quality.models import connect, date_iso, now_iso


class TradingCalendarService:
    def __init__(self, db_path: str | Path = DATABASE_PATH, market: str = "CN") -> None:
        self.db_path = Path(db_path)
        self.market = market
        connect(self.db_path).close()

    def save_calendar(self, calendar: pd.DataFrame, source: str = "manual") -> int:
        if calendar.empty:
            return 0
        records = []
        for row in calendar.to_dict("records"):
            records.append(
                {
                    "date": date_iso(row.get("date")),
                    "is_trading_day": 1 if bool(row.get("is_trading_day", True)) else 0,
                    "market": row.get("market") or self.market,
                    "source": row.get("source") or source,
                    "created_at": now_iso(),
                }
            )
        with connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO trading_calendar(date, is_trading_day, market, source, created_at)
                VALUES (:date, :is_trading_day, :market, :source, :created_at)
                ON CONFLICT(date, market) DO UPDATE SET
                    is_trading_day = excluded.is_trading_day,
                    source = excluded.source,
                    created_at = excluded.created_at
                """,
                records,
            )
        return len(records)

    def get_trading_calendar(
        self,
        start_date: str,
        end_date: str,
        price_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        start = date_iso(start_date)
        end = date_iso(end_date)
        with connect(self.db_path) as conn:
            frame = pd.read_sql_query(
                """
                SELECT date, is_trading_day, market, source, created_at
                FROM trading_calendar
                WHERE market = ? AND date >= ? AND date <= ?
                ORDER BY date
                """,
                conn,
                params=(self.market, start, end),
                parse_dates=["date"],
            )
        if not frame.empty:
            frame["is_trading_day"] = frame["is_trading_day"].astype(bool)
            return frame
        return self._infer_from_prices(start, end, price_df)

    def is_trading_day(self, value: str) -> bool:
        date = date_iso(value)
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT is_trading_day FROM trading_calendar WHERE market = ? AND date = ?", (self.market, date)).fetchone()
        return bool(row["is_trading_day"]) if row else False

    def get_previous_trading_day(self, value: str) -> str | None:
        date = date_iso(value)
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT date FROM trading_calendar WHERE market = ? AND is_trading_day = 1 AND date < ? ORDER BY date DESC LIMIT 1",
                (self.market, date),
            ).fetchone()
        return row["date"] if row else None

    def get_next_trading_day(self, value: str) -> str | None:
        date = date_iso(value)
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT date FROM trading_calendar WHERE market = ? AND is_trading_day = 1 AND date > ? ORDER BY date LIMIT 1",
                (self.market, date),
            ).fetchone()
        return row["date"] if row else None

    def count_trading_days(self, start_date: str, end_date: str) -> int:
        frame = self.get_trading_calendar(start_date, end_date)
        return int(frame["is_trading_day"].sum()) if not frame.empty else 0

    def find_missing_trading_days(self, price_df: pd.DataFrame, code: str, start_date: str, end_date: str) -> list[str]:
        calendar = self.get_trading_calendar(start_date, end_date)
        if calendar.empty or price_df.empty:
            return []
        prices = price_df.copy()
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        actual = set(prices.loc[prices["code"].astype(str).str.zfill(6).eq(str(code).zfill(6)), "date"].dt.strftime("%Y-%m-%d"))
        expected = calendar.loc[calendar["is_trading_day"], "date"].dt.strftime("%Y-%m-%d").tolist()
        return [date for date in expected if date not in actual]

    def _infer_from_prices(self, start_date: str | None, end_date: str | None, price_df: pd.DataFrame | None) -> pd.DataFrame:
        prices = price_df.copy() if isinstance(price_df, pd.DataFrame) else pd.DataFrame()
        if prices.empty:
            try:
                prices = get_stock_daily(start_date=start_date, end_date=end_date, engine=get_engine(self.db_path))
            except Exception:
                prices = pd.DataFrame()
        if prices.empty or "date" not in prices.columns:
            return pd.DataFrame(columns=["date", "is_trading_day", "market", "source", "created_at"])
        dates = pd.to_datetime(prices["date"], errors="coerce").dropna().drop_duplicates().sort_values()
        frame = pd.DataFrame({"date": dates, "is_trading_day": True, "market": self.market, "source": "inferred_from_prices", "created_at": now_iso()})
        return frame.reset_index(drop=True)
