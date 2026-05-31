from __future__ import annotations

import pandas as pd


DEFAULT_STRATEGY_WEIGHTS = {
    "trend_strength_strategy": 0.25,
    "pullback_ma20_strategy": 0.30,
    "volume_breakout_strategy": 0.20,
    "multi_factor_score_strategy": 0.25,
}


def combine_strategy_signals(
    signals_by_strategy: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
    deduplicate: bool = True,
    top_n_per_day: int | None = None,
) -> pd.DataFrame:
    weights = weights or DEFAULT_STRATEGY_WEIGHTS
    frames = []
    for strategy_name, frame in signals_by_strategy.items():
        if frame is None or frame.empty:
            continue
        current = frame.copy()
        current["strategy_name"] = current.get("strategy_name", strategy_name)
        current["source_strategy"] = strategy_name
        current["weight"] = float(weights.get(strategy_name, 0.0))
        frames.append(current)

    if not frames:
        return _empty_ensemble_frame()

    signals = pd.concat(frames, ignore_index=True)
    signals["code"] = signals["code"].astype(str).str.zfill(6)
    signals["signal_date"] = pd.to_datetime(signals["signal_date"])
    signals["score"] = pd.to_numeric(signals["score"], errors="coerce").fillna(0)
    if not deduplicate:
        result = signals.copy()
        result["triggered_strategies"] = result["source_strategy"]
        result["strategy_name"] = "ensemble_strategy"
        return _sort_and_limit(result, top_n_per_day)

    rows = []
    for (_, _, signal_date), group in signals.groupby(["code", "name", "signal_date"], sort=False):
        weight_sum = group["weight"].sum()
        score = (
            (group["score"] * group["weight"]).sum() / weight_sum
            if weight_sum > 0
            else group["score"].mean()
        )
        triggered = sorted(group["source_strategy"].astype(str).unique())
        rows.append(
            {
                "code": group["code"].iloc[0],
                "name": group["name"].iloc[0],
                "signal_date": signal_date,
                "strategy_name": "ensemble_strategy",
                "score": round(float(score), 2),
                "signal_reason": " | ".join(group["signal_reason"].astype(str).dropna().unique()),
                "suggested_buy_price": float(pd.to_numeric(group["suggested_buy_price"]).mean()),
                "suggested_stop_loss": float(pd.to_numeric(group["suggested_stop_loss"]).min()),
                "suggested_take_profit": float(pd.to_numeric(group["suggested_take_profit"]).max()),
                "risk_level": _combine_risk(group["risk_level"]),
                "triggered_strategies": ",".join(triggered),
            }
        )
    return _sort_and_limit(pd.DataFrame(rows), top_n_per_day)


def _sort_and_limit(df: pd.DataFrame, top_n_per_day: int | None) -> pd.DataFrame:
    if df.empty:
        return _empty_ensemble_frame()
    result = df.sort_values(["signal_date", "score"], ascending=[True, False]).reset_index(drop=True)
    if top_n_per_day is not None and top_n_per_day > 0:
        result = result.groupby("signal_date", group_keys=False).head(int(top_n_per_day)).reset_index(drop=True)
    return result


def _combine_risk(values: pd.Series) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    reverse = {value: key for key, value in order.items()}
    highest = max(order.get(str(value), 1) for value in values)
    return reverse[highest]


def _empty_ensemble_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "code",
            "name",
            "signal_date",
            "strategy_name",
            "score",
            "signal_reason",
            "suggested_buy_price",
            "suggested_stop_loss",
            "suggested_take_profit",
            "risk_level",
            "triggered_strategies",
        ]
    )
