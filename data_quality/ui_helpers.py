from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def parse_code_list(value: str) -> list[str]:
    return [part.strip().zfill(6) for part in value.replace("\n", ",").split(",") if part.strip()]


def severity_badge(severity: str) -> str:
    mapping = {"CRITICAL": "严重", "ERROR": "错误", "WARNING": "警告", "INFO": "信息"}
    return mapping.get(str(severity).upper(), str(severity))


def file_info(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    return {"path": str(target), "size": target.stat().st_size if target.exists() else 0, "name": target.name}


def dataframe_or_empty(value: object, columns: list[str] | None = None) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame(columns=columns or [])
