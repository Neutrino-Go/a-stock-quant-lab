from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import DAILY_COLUMNS, DATABASE_PATH, ensure_directories


def get_engine(db_path: str | Path = DATABASE_PATH) -> Engine:
    ensure_directories()
    return create_engine(f"sqlite:///{Path(db_path).as_posix()}", future=True)


def init_db(engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS stock_list (
                    code TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS daily_price (
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    turnover REAL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (code, date)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_daily_price_date
                ON daily_price(date)
                """
            )
        )
        from workflow.research_store import RESEARCH_SCHEMA_SQL

        conn.connection.executescript(RESEARCH_SCHEMA_SQL)


def _normalize_code(value: Any) -> str:
    return str(value).strip().zfill(6)


def _normalize_date(value: Any) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def normalize_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    missing = set(DAILY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing daily columns: {sorted(missing)}")

    result = df.loc[:, DAILY_COLUMNS].copy()
    result["code"] = result["code"].map(_normalize_code)
    result["name"] = result["name"].fillna("")
    result["date"] = result["date"].map(_normalize_date)
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.drop_duplicates(["code", "date"]).sort_values(["code", "date"])


def upsert_stock_list(df: pd.DataFrame, engine: Engine | None = None) -> int:
    if df.empty:
        return 0

    engine = engine or get_engine()
    init_db(engine)
    records = (
        df.loc[:, ["code", "name"]]
        .dropna(subset=["code", "name"])
        .assign(code=lambda data: data["code"].map(_normalize_code))
        .drop_duplicates("code")
        .to_dict("records")
    )
    if not records:
        return 0

    sql = text(
        """
        INSERT INTO stock_list(code, name, updated_at)
        VALUES (:code, :name, CURRENT_TIMESTAMP)
        ON CONFLICT(code) DO UPDATE SET
            name = excluded.name,
            updated_at = CURRENT_TIMESTAMP
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, records)
    return len(records)


def upsert_stock_daily(df: pd.DataFrame, engine: Engine | None = None) -> int:
    normalized = normalize_daily_frame(df)
    if normalized.empty:
        return 0

    engine = engine or get_engine()
    init_db(engine)
    sql = text(
        """
        INSERT INTO daily_price(
            code, name, date, open, high, low, close, volume, amount, turnover, updated_at
        )
        VALUES (
            :code, :name, :date, :open, :high, :low, :close,
            :volume, :amount, :turnover, CURRENT_TIMESTAMP
        )
        ON CONFLICT(code, date) DO UPDATE SET
            name = excluded.name,
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            amount = excluded.amount,
            turnover = excluded.turnover,
            updated_at = CURRENT_TIMESTAMP
        """
    )
    records = normalized.to_dict("records")
    with engine.begin() as conn:
        conn.execute(sql, records)
    return len(records)


def get_stock_list(engine: Engine | None = None) -> pd.DataFrame:
    engine = engine or get_engine()
    init_db(engine)
    with engine.connect() as conn:
        return pd.read_sql_query(
            text("SELECT code, name, updated_at FROM stock_list ORDER BY code"),
            conn,
        )


def get_stock_name(code: str, engine: Engine | None = None) -> str | None:
    engine = engine or get_engine()
    init_db(engine)
    with engine.connect() as conn:
        value = conn.execute(
            text("SELECT name FROM stock_list WHERE code = :code"),
            {"code": _normalize_code(code)},
        ).scalar()
    return str(value) if value is not None else None


def get_latest_trade_date(code: str, engine: Engine | None = None) -> str | None:
    engine = engine or get_engine()
    init_db(engine)
    with engine.connect() as conn:
        value = conn.execute(
            text("SELECT MAX(date) FROM daily_price WHERE code = :code"),
            {"code": _normalize_code(code)},
        ).scalar()
    return str(value) if value else None


def get_stock_daily(
    code: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    engine: Engine | None = None,
) -> pd.DataFrame:
    engine = engine or get_engine()
    init_db(engine)

    conditions: list[str] = []
    params: dict[str, Any] = {}
    if code:
        conditions.append("code = :code")
        params["code"] = _normalize_code(code)
    if start_date:
        conditions.append("date >= :start_date")
        params["start_date"] = _normalize_date(start_date)
    if end_date:
        conditions.append("date <= :end_date")
        params["end_date"] = _normalize_date(end_date)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = text(
        f"""
        SELECT code, name, date, open, high, low, close, volume, amount, turnover
        FROM daily_price
        {where}
        ORDER BY code, date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
    return df


def get_database_status(engine: Engine | None = None) -> dict[str, Any]:
    engine = engine or get_engine()
    init_db(engine)
    with engine.connect() as conn:
        stock_count = conn.execute(text("SELECT COUNT(*) FROM stock_list")).scalar() or 0
        daily_rows = conn.execute(text("SELECT COUNT(*) FROM daily_price")).scalar() or 0
        date_range = conn.execute(
            text("SELECT MIN(date), MAX(date) FROM daily_price")
        ).one()
    return {
        "db_path": str(DATABASE_PATH),
        "stock_count": int(stock_count),
        "daily_rows": int(daily_rows),
        "min_date": date_range[0],
        "max_date": date_range[1],
    }
