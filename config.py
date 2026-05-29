from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
DATABASE_PATH = STORAGE_DIR / "stock_quant.db"

DEFAULT_START_DATE = "20100101"
DEFAULT_ADJUST = "qfq"

DAILY_COLUMNS = [
    "code",
    "name",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
]


def ensure_directories() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
