from __future__ import annotations


STRATEGY_DISPLAY_NAMES = {
    "trend_strength_strategy": "趋势强势股策略",
    "pullback_ma20_strategy": "MA20回踩策略",
    "volume_breakout_strategy": "放量突破策略",
    "multi_factor_score_strategy": "多因子评分策略",
    "ensemble_strategy": "策略组合",
}


STRATEGY_DESCRIPTIONS = {
    "trend_strength_strategy": {
        "name": "趋势强势股策略",
        "conditions": [
            "close > MA20 > MA60",
            "20日涨幅 > 10%",
            "成交额 > 2亿",
            "排除 ST",
        ],
        "suitable_for": ["牛市", "主升浪"],
        "risk": "容易追高，震荡市中假突破较多。",
    },
    "pullback_ma20_strategy": {
        "name": "MA20回踩策略",
        "conditions": [
            "MA20 > MA60",
            "价格距离 MA20 不超过 3%",
            "最近5日涨跌幅在 -8% 到 3%",
            "量比 < 1.2",
        ],
        "suitable_for": ["趋势延续", "温和回踩"],
        "risk": "趋势转弱时容易接到下跌中继。",
    },
    "volume_breakout_strategy": {
        "name": "放量突破策略",
        "conditions": [
            "突破 60 日新高",
            "量比 > 1.8",
            "当日涨幅在 3% 到 8%",
        ],
        "suitable_for": ["强势突破", "题材扩散"],
        "risk": "高位放量后可能快速回落。",
    },
    "multi_factor_score_strategy": {
        "name": "多因子评分策略",
        "conditions": [
            "趋势强度",
            "20日动量",
            "成交额活跃度",
            "60日突破",
            "MA20距离",
            "20日回撤风险",
        ],
        "suitable_for": ["多策略筛选", "横向比较"],
        "risk": "评分模型需要持续校准。",
    },
}


def display_strategy_name(strategy_name: str) -> str:
    return STRATEGY_DISPLAY_NAMES.get(strategy_name, strategy_name)
