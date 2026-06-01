from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from research.experiment_store import ExperimentStore


DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "experiments"
CsvKind = Literal["trades", "orders", "equity_curve"]


def export_markdown_report(
    store: ExperimentStore,
    experiment_id: str,
    output_dir: str | Path = DEFAULT_REPORT_DIR,
) -> Path:
    detail = store.get_experiment_detail(experiment_id)
    if detail["run"] is None:
        raise KeyError(f"Experiment not found: {experiment_id}")
    output_path = _output_path(output_dir, experiment_id, "md")
    output_path.write_text(_markdown_content(detail), encoding="utf-8")
    store.add_artifact(experiment_id, "markdown_report", output_path)
    return output_path


def export_experiment_csv(
    store: ExperimentStore,
    experiment_id: str,
    kind: CsvKind,
    output_dir: str | Path = DEFAULT_REPORT_DIR,
) -> Path:
    detail = store.get_experiment_detail(experiment_id)
    if detail["run"] is None:
        raise KeyError(f"Experiment not found: {experiment_id}")
    if kind == "trades":
        frame = detail["trades"]
    elif kind == "orders":
        frame = detail["orders"]
    elif kind == "equity_curve":
        frame = detail["equity_curve"].copy()
        if not frame.empty and "total_equity" not in frame.columns and "net_equity" in frame.columns:
            frame["total_equity"] = frame["net_equity"]
        if not frame.empty and "gross_total_equity" not in frame.columns and "gross_equity" in frame.columns:
            frame["gross_total_equity"] = frame["gross_equity"]
    else:
        raise ValueError(f"Unsupported CSV export kind: {kind}")

    output_path = _output_path(output_dir, experiment_id, f"{kind}.csv")
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    store.add_artifact(experiment_id, "csv_export", output_path)
    return output_path


def export_config_json(
    store: ExperimentStore,
    experiment_id: str,
    output_dir: str | Path = DEFAULT_REPORT_DIR,
) -> Path:
    config_json = store.duplicate_experiment_config(experiment_id)
    output_path = _output_path(output_dir, experiment_id, "config.json")
    import json

    output_path.write_text(json.dumps(config_json, ensure_ascii=False, indent=2), encoding="utf-8")
    store.add_artifact(experiment_id, "json_config", output_path)
    return output_path


def _output_path(output_dir: str | Path, experiment_id: str, suffix: str) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return directory / f"experiment_{experiment_id}_{timestamp}.{suffix}"


def _markdown_content(detail: dict[str, object]) -> str:
    run = detail["run"]
    metrics = run["metrics_json"]
    config = run["config_json"]
    trades = detail["trades"]

    best_trade = _best_trade(trades)
    worst_trade = _worst_trade(trades)
    avg_holding_days = _avg(trades, "holding_days")

    return f"""# 实验报告

## 基本信息
- 实验名称：{run["name"]}
- 实验类型：{run["experiment_type"]}
- 创建时间：{run["created_at"]}
- 回测区间：{run.get("start_date") or "-"} 至 {run.get("end_date") or "-"}
- 策略：{run.get("strategy_name") or "-"}
- Git Commit：{run.get("git_commit") or "-"}

## 参数配置
- strategy_config：`{config.get("strategy_config", {})}`
- cost_config：`{config.get("cost_config", {})}`
- position_config：`{config.get("position_config", {})}`
- exit_config：`{config.get("exit_config", {})}`

## 核心指标
- final_equity：{metrics.get("final_equity", "-")}
- total_return：{metrics.get("total_return", "-")}
- net_total_return：{metrics.get("net_total_return", "-")}
- CAGR：{metrics.get("CAGR", "-")}
- net_CAGR：{metrics.get("net_CAGR", "-")}
- max_drawdown：{metrics.get("max_drawdown", "-")}
- sharpe_ratio：{metrics.get("sharpe_ratio", "-")}
- win_rate：{metrics.get("win_rate", "-")}
- cost_impact：{metrics.get("cost_impact", "-")}

## 成本分析
- commission：{metrics.get("commission", metrics.get("total_commission", _sum(trades, "commission")))}
- stamp_tax：{metrics.get("stamp_tax", metrics.get("total_stamp_tax", _sum(trades, "stamp_tax")))}
- regulatory_fee：{metrics.get("regulatory_fee", metrics.get("total_regulatory_fee", _sum(trades, "regulatory_fee")))}
- transfer_fee：{metrics.get("transfer_fee", metrics.get("total_transfer_fee", _sum(trades, "transfer_fee")))}
- slippage_cost：{metrics.get("slippage_cost", metrics.get("total_slippage_cost", _sum(trades, "slippage_cost")))}

## 交易摘要
- total_trades：{metrics.get("total_trades", len(trades))}
- avg_holding_days：{metrics.get("avg_holding_days", avg_holding_days)}
- best_trade：{best_trade}
- worst_trade：{worst_trade}

## 研究结论
- notes：{run.get("notes") or "-"}
- conclusion：{run.get("conclusion") or "-"}
- tags：{", ".join(run.get("tags") or []) or "-"}
- rating：{run.get("rating") or "unrated"}

## 风险提示
- 回测结果不代表未来收益
- 本工具不构成投资建议
- 本工具不支持自动交易
"""


def _sum(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def _avg(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[column], errors="coerce").fillna(0).mean())


def _best_trade(df: pd.DataFrame) -> str:
    if df.empty:
        return "-"
    column = "net_return" if "net_return" in df.columns else "return"
    if column not in df.columns:
        return "-"
    idx = pd.to_numeric(df[column], errors="coerce").idxmax()
    return str(df.loc[idx].to_dict())


def _worst_trade(df: pd.DataFrame) -> str:
    if df.empty:
        return "-"
    column = "net_return" if "net_return" in df.columns else "return"
    if column not in df.columns:
        return "-"
    idx = pd.to_numeric(df[column], errors="coerce").idxmin()
    return str(df.loc[idx].to_dict())
