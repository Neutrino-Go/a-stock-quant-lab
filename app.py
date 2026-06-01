from __future__ import annotations

import json
from datetime import date

import pandas as pd
import streamlit as st

from backtest.backtest_engine import apply_realistic_costs_to_trades, backtest_signals, run_strategy_backtest
from data.data_updater import DataUpdateService
from data.database import get_database_status, get_stock_daily, get_stock_list
from execution.cost_model import TradingCostConfig
from factor.factor_engine import compute_factors
from market.index_data import get_index_prices, update_hs300_index
from market.regime_engine import build_market_proxy_from_prices, classify_market_regime, summarize_trades_by_regime
from optimizer.parameter_optimizer import DEFAULT_PARAM_GRIDS, optimize_strategy
from portfolio.portfolio_engine import ExitConfig, PortfolioBacktestEngine
from portfolio.position_sizer import PositionSizingConfig
from portfolio.strategy_ensemble import DEFAULT_STRATEGY_WEIGHTS, combine_strategy_signals
from ranking.leaderboard import build_strategy_leaderboard
from ranking.ranking_engine import STRATEGIES, rank_recommendations, run_all_strategies, run_strategy
from research.experiment_compare import compare_experiments
from research.experiment_service import ExperimentService
from research.report_exporter import export_config_json, export_experiment_csv, export_markdown_report
from research.ui_helpers import (
    EXPERIMENT_TYPE_OPTIONS,
    RATING_OPTIONS,
    SORTABLE_EXPERIMENT_COLUMNS,
    delete_confirmation_matches,
    experiment_option_label,
    filter_experiment_table,
    parse_tags,
    rating_display_name,
    sort_experiment_table,
)
from report.charts import (
    plot_cost_breakdown,
    plot_drawdown_curve,
    plot_equity_curve,
    plot_gross_vs_net_return,
    plot_in_sample_vs_out_sample,
    plot_monthly_returns,
    plot_overfit_risk_flags,
    plot_parameter_sensitivity_heatmap,
    plot_portfolio_drawdown_curve,
    plot_portfolio_equity_curve,
    plot_position_count,
    plot_price_with_ma,
    plot_regime_stability,
    plot_trade_returns,
    plot_walk_forward_returns,
)
from stability.stability_service import DEFAULT_STABILITY_PARAMETER_GRIDS, StrategyStabilityService
from strategy.metadata import STRATEGY_DESCRIPTIONS, display_strategy_name


st.set_page_config(page_title="A股量化研究平台", layout="wide")

STRATEGY_OPTIONS = {display_strategy_name(key): key for key in STRATEGIES}


@st.cache_data(ttl=300)
def load_daily_data(
    code: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    return get_stock_daily(code=code, start_date=start_date, end_date=end_date)


@st.cache_data(ttl=300)
def load_factor_data(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    daily = get_stock_daily(start_date=start_date, end_date=end_date)
    if daily.empty:
        return daily
    return compute_factors(daily)


@st.cache_resource
def experiment_service() -> ExperimentService:
    return ExperimentService()


@st.cache_resource
def stability_service() -> StrategyStabilityService:
    return StrategyStabilityService(experiment_service=experiment_service(), factor_data_loader=load_factor_data)


def clear_data_cache() -> None:
    load_daily_data.clear()
    load_factor_data.clear()


def pct(value: object) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "-"


def num(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def config_to_dict(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return {key: config_to_dict(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {key: config_to_dict(item) for key, item in value.items()}
    return value


def experiment_equity_for_chart(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return equity.copy()
    result = equity.copy()
    if "total_equity" not in result.columns and "net_equity" in result.columns:
        result["total_equity"] = result["net_equity"]
    if "gross_total_equity" not in result.columns and "gross_equity" in result.columns:
        result["gross_total_equity"] = result["gross_equity"]
    if "positions_count" not in result.columns and "position_count" in result.columns:
        result["positions_count"] = result["position_count"]
    return result


def select_strategy(label: str, key: str) -> str:
    selected = st.selectbox(label, list(STRATEGY_OPTIONS.keys()), key=key)
    return STRATEGY_OPTIONS[selected]


def strategy_params_panel(strategy_name: str, prefix: str = "") -> dict[str, object]:
    if strategy_name == "trend_strength_strategy":
        col1, col2 = st.columns(2)
        with col1:
            min_amount = st.number_input("最低成交额", min_value=0, value=200_000_000, step=50_000_000, key=f"{prefix}min_amount")
        with col2:
            min_return_20d = st.slider("20日涨幅阈值", min_value=0.05, max_value=0.30, value=0.10, step=0.01, key=f"{prefix}min_return_20d")
        return {"min_amount": float(min_amount), "min_return_20d": float(min_return_20d)}
    if strategy_name == "pullback_ma20_strategy":
        return {"max_distance": st.slider("MA20距离上限", 0.01, 0.10, 0.03, 0.01, key=f"{prefix}max_distance")}
    if strategy_name == "volume_breakout_strategy":
        return {"min_volume_ratio": st.slider("最低量比", 1.0, 5.0, 1.8, 0.1, key=f"{prefix}min_volume_ratio")}
    if strategy_name == "multi_factor_score_strategy":
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.number_input("Top N", min_value=1, max_value=200, value=30, key=f"{prefix}top_n")
        with col2:
            min_score = st.number_input("最低分", min_value=0, max_value=100, value=60, key=f"{prefix}min_score")
        return {"top_n": int(top_n), "min_score": float(min_score)}
    return {}


def strategy_card(strategy_name: str) -> None:
    meta = STRATEGY_DESCRIPTIONS[strategy_name]
    with st.container(border=True):
        st.subheader(meta["name"])
        st.markdown("**条件**")
        for condition in meta["conditions"]:
            st.write(f"- {condition}")
        st.markdown("**适用**")
        st.write(" / ".join(meta["suitable_for"]))
        st.markdown("**风险**")
        st.write(meta["risk"])


def cost_config_panel(prefix: str = "") -> TradingCostConfig:
    with st.expander("交易成本设置", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            commission_rate = st.number_input("佣金比例", min_value=0.0, value=0.0003, format="%.5f", key=f"{prefix}commission_rate")
            stamp_tax_rate_sell = st.number_input("卖出印花税", min_value=0.0, value=0.0005, format="%.5f", key=f"{prefix}stamp_tax")
        with col2:
            min_commission = st.number_input("最低佣金", min_value=0.0, value=5.0, step=1.0, key=f"{prefix}min_commission")
            regulatory_fee_rate = st.number_input("规费", min_value=0.0, value=0.00002, format="%.5f", key=f"{prefix}regulatory")
        with col3:
            transfer_fee_rate = st.number_input("过户费", min_value=0.0, value=0.00001, format="%.5f", key=f"{prefix}transfer")
            slippage_bps = st.number_input("滑点 bps", min_value=0.0, value=5.0, step=1.0, key=f"{prefix}slippage")
    return TradingCostConfig(
        commission_rate=float(commission_rate),
        min_commission=float(min_commission),
        stamp_tax_rate_sell=float(stamp_tax_rate_sell),
        regulatory_fee_rate=float(regulatory_fee_rate),
        transfer_fee_rate=float(transfer_fee_rate),
        slippage_bps=float(slippage_bps),
    )


def with_display_strategy_names(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "strategy_name" not in df.columns:
        return df.copy()
    display = df.copy()
    display["strategy_name"] = display["strategy_name"].map(
        lambda value: ", ".join(display_strategy_name(item.strip()) for item in str(value).split(","))
    )
    if "triggered_strategies" in display.columns:
        display["triggered_strategies"] = display["triggered_strategies"].map(
            lambda value: ", ".join(display_strategy_name(item.strip()) for item in str(value).split(",") if item)
        )
    return display


def render_metrics(metrics: dict[str, object]) -> None:
    items = [
        ("交易数", str(int(metrics.get("total_trades", 0)))),
        ("胜率", pct(metrics.get("win_rate", 0))),
        ("平均收益", pct(metrics.get("avg_return", metrics.get("total_return", 0)))),
        ("CAGR", pct(metrics.get("CAGR", 0))),
        ("最大回撤", pct(metrics.get("max_drawdown", 0))),
        ("夏普", num(metrics.get("sharpe_ratio", 0))),
        ("盈亏比", num(metrics.get("profit_loss_ratio", 0))),
        ("期望收益", pct(metrics.get("expectancy", 0))),
    ]
    columns = st.columns(4)
    for idx, (label, value) in enumerate(items):
        columns[idx % 4].metric(label, value)


def render_portfolio_metrics(metrics: dict[str, object], initial_cash: float) -> None:
    items = [
        ("初始资金", f"{initial_cash:,.0f}"),
        ("最终权益", f"{float(metrics.get('final_equity', initial_cash)):,.0f}"),
        ("总收益", pct(metrics.get("net_total_return", 0))),
        ("年化收益", pct(metrics.get("CAGR", 0))),
        ("最大回撤", pct(metrics.get("max_drawdown", 0))),
        ("夏普", num(metrics.get("sharpe_ratio", 0))),
        ("胜率", pct(metrics.get("win_rate", 0))),
        ("交易成本影响", pct(metrics.get("cost_impact", 0))),
    ]
    columns = st.columns(4)
    for idx, (label, value) in enumerate(items):
        columns[idx % 4].metric(label, value)


def data_center() -> None:
    st.header("数据中心")
    status = get_database_status()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("股票数", status["stock_count"])
    col2.metric("日线记录", status["daily_rows"])
    col3.metric("开始日期", status["min_date"] or "-")
    col4.metric("结束日期", status["max_date"] or "-")

    service = DataUpdateService()
    with st.expander("更新股票列表", expanded=True):
        if st.button("更新股票列表"):
            with st.spinner("正在从 AKShare 获取股票列表"):
                count = service.update_stock_list()
            clear_data_cache()
            st.success(f"已更新 {count} 只股票")

    with st.expander("更新单只股票数据", expanded=True):
        code = st.text_input("股票代码", value="000001", max_chars=6)
        col1, col2, col3 = st.columns(3)
        with col1:
            start = st.date_input("开始日期", value=date(2020, 1, 1), key="single_start")
        with col2:
            end = st.date_input("结束日期", value=date.today(), key="single_end")
        with col3:
            force = st.checkbox("强制覆盖日期区间", value=False)
        if st.button("更新单只股票"):
            with st.spinner("正在更新日线"):
                rows = service.update_stock_daily(code=code, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"), force=force)
            clear_data_cache()
            st.success(f"写入/更新 {rows} 行")

    with st.expander("批量更新股票数据"):
        codes_text = st.text_area("股票代码列表", placeholder="000001, 000002, 600000")
        limit = st.number_input("最多更新数量", min_value=1, max_value=5000, value=20)
        col1, col2 = st.columns(2)
        with col1:
            batch_start = st.date_input("批量开始日期", value=date(2020, 1, 1))
        with col2:
            batch_end = st.date_input("批量结束日期", value=date.today())
        if st.button("批量更新"):
            codes = [item.strip() for item in codes_text.replace("\n", ",").split(",") if item.strip()]
            with st.spinner("正在批量更新"):
                result = service.update_batch(codes=codes or None, limit=int(limit), start_date=batch_start.strftime("%Y%m%d"), end_date=batch_end.strftime("%Y%m%d"))
            clear_data_cache()
            st.dataframe(result, use_container_width=True)

    with st.expander("更新沪深300指数"):
        col1, col2 = st.columns(2)
        with col1:
            idx_start = st.date_input("指数开始日期", value=date(2020, 1, 1))
        with col2:
            idx_end = st.date_input("指数结束日期", value=date.today())
        if st.button("更新沪深300"):
            with st.spinner("正在更新沪深300指数"):
                rows = update_hs300_index(start_date=idx_start.strftime("%Y%m%d"), end_date=idx_end.strftime("%Y%m%d"))
            st.success(f"写入/更新 {rows} 行指数数据")

    stocks = get_stock_list()
    if not stocks.empty:
        st.subheader("股票列表")
        st.dataframe(stocks.head(200), use_container_width=True)


def strategy_lab() -> None:
    st.header("策略实验室")
    factors = load_factor_data()
    if factors.empty:
        st.warning("数据库暂无日线数据")
        return
    latest_date = pd.to_datetime(factors["date"]).max().date()
    col1, col2 = st.columns(2)
    with col1:
        strategy_name = select_strategy("策略", "lab_strategy")
    with col2:
        signal_date = st.date_input("信号日期", value=latest_date)
    strategy_card(strategy_name)
    params = strategy_params_panel(strategy_name, prefix="lab_")
    if st.button("运行策略"):
        result = run_strategy(strategy_name, factors, signal_date=pd.Timestamp(signal_date), **params)
        st.dataframe(with_display_strategy_names(result), use_container_width=True)


def backtest_center() -> None:
    st.header("回测中心")
    factors = load_factor_data()
    if factors.empty:
        st.warning("数据库暂无日线数据")
        return

    min_date = pd.to_datetime(factors["date"]).min().date()
    max_date = pd.to_datetime(factors["date"]).max().date()
    strategy_name = select_strategy("回测策略", "backtest_strategy")
    strategy_card(strategy_name)
    params = strategy_params_panel(strategy_name, prefix="bt_")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_date = st.date_input("开始", value=min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("结束", value=max_date, min_value=min_date, max_value=max_date)
    with col3:
        holding_days = st.selectbox("持仓周期", [5, 10, 20, 60], index=2)
    with col4:
        sell_on_ma20_break = st.checkbox("跌破MA20卖出", value=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        take_profit_pct = st.selectbox("止盈", [0.10, 0.15, 0.20], format_func=lambda value: f"{value:.0%}")
    with col2:
        stop_loss_pct = st.selectbox("止损", [0.05, 0.08, 0.10], index=1, format_func=lambda value: f"{value:.0%}")
    with col3:
        enable_costs = st.checkbox("启用真实交易成本", value=False)
    cost_config = cost_config_panel(prefix="single_bt_") if enable_costs else TradingCostConfig(slippage_bps=0, commission_rate=0, min_commission=0, stamp_tax_rate_sell=0, regulatory_fee_rate=0, transfer_fee_rate=0)

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        experiment_name = st.text_input("实验名称", value=f"{display_strategy_name(strategy_name)} 回测实验", key="single_exp_name")
    with exp_col2:
        experiment_tags = st.text_input("实验标签", value="single_strategy", key="single_exp_tags")

    if st.button("运行回测"):
        with st.spinner("正在生成历史信号并回测"):
            result = run_strategy_backtest(
                factor_df=factors,
                strategy_func=STRATEGIES[strategy_name],
                start_date=pd.Timestamp(start_date),
                end_date=pd.Timestamp(end_date),
                strategy_params=params,
                holding_days=int(holding_days),
                take_profit_pct=float(take_profit_pct),
                stop_loss_pct=float(stop_loss_pct),
                sell_on_ma20_break=sell_on_ma20_break,
            )
        if enable_costs and not result["trade_details"].empty:
            adjusted_trades, cost_metrics = apply_realistic_costs_to_trades(result["trade_details"], cost_config=cost_config)
            result.update(cost_metrics)
            result["trade_details"] = adjusted_trades

        st.session_state["single_backtest_result"] = {
            "result": result,
            "strategy_name": strategy_name,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "config": {
                "strategy_config": params,
                "cost_config": config_to_dict(cost_config),
                "exit_config": {
                    "holding_days": int(holding_days),
                    "take_profit_pct": float(take_profit_pct),
                    "stop_loss_pct": float(stop_loss_pct),
                    "sell_on_ma20_break": sell_on_ma20_break,
                    "enable_costs": enable_costs,
                },
            },
        }

    saved = st.session_state.get("single_backtest_result")
    if not saved:
        return

    result = saved["result"]
    render_metrics(result)
    trades = result["trade_details"]
    if st.button("保存本次单策略回测"):
        experiment_id = experiment_service().save_single_strategy_backtest_result(
            name=experiment_name,
            strategy_name=saved["strategy_name"],
            result=result,
            start_date=saved["start_date"],
            end_date=saved["end_date"],
            config=saved["config"],
            tags=parse_tags(experiment_tags),
        )
        st.success(f"实验已保存：{experiment_id}")

    if not trades.empty:
        left, right = st.columns(2)
        left.plotly_chart(plot_equity_curve(trades), use_container_width=True)
        right.plotly_chart(plot_drawdown_curve(result["drawdown_curve"]), use_container_width=True)
        st.plotly_chart(plot_trade_returns(trades), use_container_width=True)
        if enable_costs:
            st.plotly_chart(
                plot_gross_vs_net_return(
                    {
                        "gross_total_return": result.get("gross_return", 0),
                        "net_total_return": result.get("net_return", 0),
                        "cost_impact": result.get("cost_impact", 0),
                    }
                ),
                use_container_width=True,
            )
            st.plotly_chart(plot_cost_breakdown(trades), use_container_width=True)
            st.info(
                f"成本影响 {pct(result.get('cost_impact', 0))}，佣金 {num(result.get('total_commission', 0))}，"
                f"印花税 {num(result.get('total_stamp_tax', 0))}，滑点成本 {num(result.get('total_slippage_cost', 0))}。"
            )
        st.subheader("交易明细")
        st.dataframe(with_display_strategy_names(trades), use_container_width=True)
        st.subheader("市场环境分析")
        regimes = load_market_regime(factors)
        st.dataframe(summarize_trades_by_regime(trades, regimes), use_container_width=True)
    else:
        st.info("当前参数下没有有效交易")


def parameter_optimizer_page() -> None:
    st.header("参数优化器")
    factors = load_factor_data()
    if factors.empty:
        st.warning("数据库暂无日线数据")
        return
    supported = {display_strategy_name(name): name for name in DEFAULT_PARAM_GRIDS}
    selected = st.selectbox("优化策略", list(supported.keys()))
    strategy_name = supported[selected]
    st.json(DEFAULT_PARAM_GRIDS[strategy_name], expanded=False)
    min_date = pd.to_datetime(factors["date"]).min().date()
    max_date = pd.to_datetime(factors["date"]).max().date()
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("优化开始", value=min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("优化结束", value=max_date, min_value=min_date, max_value=max_date)
    with col3:
        holding_days = st.selectbox("优化持仓周期", [5, 10, 20, 60], index=2)
    if st.button("运行参数优化"):
        result = optimize_strategy(factors, strategy_name=strategy_name, start_date=pd.Timestamp(start_date), end_date=pd.Timestamp(end_date), holding_days=int(holding_days), top_n=20)
        result["strategy"] = result["strategy_name"].map(display_strategy_name)
        st.session_state["optimizer_result"] = {
            "result": result,
            "strategy_name": strategy_name,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "config": {
                "optimizer_config": {
                    "param_grid": DEFAULT_PARAM_GRIDS[strategy_name],
                    "holding_days": int(holding_days),
                    "top_n": 20,
                }
            },
        }
    saved = st.session_state.get("optimizer_result")
    if saved:
        st.dataframe(saved["result"], use_container_width=True)
        if st.button("保存优化结果"):
            experiment_id = experiment_service().save_optimizer_result(
                name=f"{display_strategy_name(saved['strategy_name'])} 参数优化",
                result=saved["result"],
                strategy_name=saved["strategy_name"],
                start_date=saved["start_date"],
                end_date=saved["end_date"],
                config=saved["config"],
                tags=["optimizer"],
            )
            st.success(f"优化实验已保存：{experiment_id}")


def strategy_leaderboard_page() -> None:
    st.header("策略排行榜")
    factors = load_factor_data()
    if factors.empty:
        st.warning("数据库暂无日线数据")
        return
    latest_date = pd.to_datetime(factors["date"]).max()
    with st.spinner("正在统计策略排行榜"):
        leaderboard = build_strategy_leaderboard(factors, as_of=latest_date)
    st.session_state["leaderboard_snapshot"] = {
        "leaderboard": leaderboard,
        "as_of": str(latest_date.date()),
        "config": {"as_of": str(latest_date.date())},
    }
    st.caption(f"统计截至：{latest_date.date()}，排行榜使用 net result 口径。")
    st.dataframe(leaderboard, use_container_width=True)
    if st.button("保存排行榜快照"):
        experiment_id = experiment_service().save_leaderboard_snapshot(
            name=f"策略排行榜快照 {latest_date.date()}",
            leaderboard=leaderboard,
            as_of=latest_date,
            config={"leaderboard_config": {"as_of": str(latest_date.date())}},
            tags=["leaderboard"],
        )
        st.success(f"排行榜实验已保存：{experiment_id}")


def portfolio_backtest_page() -> None:
    st.header("组合回测")
    factors = load_factor_data()
    if factors.empty:
        st.warning("数据库暂无日线数据")
        return
    min_date = pd.to_datetime(factors["date"]).min().date()
    max_date = pd.to_datetime(factors["date"]).max().date()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_date = st.date_input("组合开始", value=min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("组合结束", value=max_date, min_value=min_date, max_value=max_date)
    with col3:
        initial_cash = st.number_input("初始资金", min_value=10_000, value=1_000_000, step=50_000)
    with col4:
        max_positions = st.number_input("最大持仓数", min_value=1, max_value=50, value=5)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        max_position_pct = st.slider("单票最大仓位", 0.05, 1.0, 0.20, 0.05)
    with col2:
        position_method = st.selectbox("仓位方法", ["equal_cash", "fixed_cash", "risk_percent"])
    with col3:
        fixed_cash = st.number_input("固定每笔金额", min_value=1_000, value=100_000, step=10_000)
    with col4:
        risk_per_trade = st.slider("单笔风险", 0.001, 0.05, 0.01, 0.001)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cash_reserve_pct = st.slider("现金保留", 0.0, 0.5, 0.05, 0.01)
    with col2:
        daily_top_n = st.number_input("每日最多买入", min_value=1, max_value=50, value=5)
    with col3:
        enable_limit_check = st.checkbox("启用涨跌停检查", value=True)
    with col4:
        st.checkbox(
            "允许同股加仓",
            value=False,
            disabled=True,
            key="portfolio_allow_add_position_disabled",
            help="V3 暂不支持同股加仓；V4 将支持 lot-level 持仓核算。",
        )

    st.subheader("策略组合权重")
    weight_cols = st.columns(4)
    weights = {}
    for idx, (strategy_name, default_weight) in enumerate(DEFAULT_STRATEGY_WEIGHTS.items()):
        with weight_cols[idx % 4]:
            weights[strategy_name] = st.slider(display_strategy_name(strategy_name), 0.0, 1.0, float(default_weight), 0.05)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        holding_days = st.selectbox("组合持仓周期", [5, 10, 20, 60], index=2)
    with col2:
        take_profit_pct = st.selectbox("组合止盈", [0.10, 0.15, 0.20], index=0, format_func=lambda value: f"{value:.0%}")
    with col3:
        stop_loss_pct = st.selectbox("组合止损", [0.05, 0.08, 0.10], index=1, format_func=lambda value: f"{value:.0%}")
    with col4:
        sell_on_ma20_break = st.checkbox("组合跌破MA20卖出", value=True)

    cost_config = cost_config_panel(prefix="portfolio_")
    position_config = PositionSizingConfig(
        initial_cash=float(initial_cash),
        max_positions=int(max_positions),
        position_size_method=position_method,
        fixed_cash_per_trade=float(fixed_cash),
        risk_per_trade=float(risk_per_trade),
        max_position_pct=float(max_position_pct),
        cash_reserve_pct=float(cash_reserve_pct),
    )
    exit_config = ExitConfig(
        holding_days=int(holding_days),
        take_profit_pct=float(take_profit_pct),
        stop_loss_pct=float(stop_loss_pct),
        sell_on_ma20_break=sell_on_ma20_break,
        allow_add_position=False,
        enable_limit_check=enable_limit_check,
        daily_top_n=int(daily_top_n),
    )

    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        experiment_name = st.text_input("实验名称", value="组合回测实验", key="portfolio_exp_name")
    with exp_col2:
        experiment_tags = st.text_input("实验标签", value="portfolio ensemble", key="portfolio_exp_tags")
    with exp_col3:
        auto_save_experiment = st.checkbox("运行后自动保存实验", value=False)

    if st.button("运行组合回测"):
        with st.spinner("正在生成策略组合信号并执行组合回测"):
            signals_by_strategy = build_strategy_signal_history(factors, pd.Timestamp(start_date), pd.Timestamp(end_date))
            ensemble = combine_strategy_signals(signals_by_strategy, weights=weights, top_n_per_day=int(daily_top_n))
            engine = PortfolioBacktestEngine(cost_config=cost_config, position_config=position_config, exit_config=exit_config)
            result = engine.run(ensemble, factors, start_date=pd.Timestamp(start_date), end_date=pd.Timestamp(end_date))
        st.session_state["portfolio_result"] = result
        st.session_state["portfolio_initial_cash"] = float(initial_cash)
        st.session_state["portfolio_experiment_context"] = {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "config": {
                "strategy_config": {"weights": weights, "daily_top_n": int(daily_top_n)},
                "cost_config": config_to_dict(cost_config),
                "position_config": config_to_dict(position_config),
                "exit_config": config_to_dict(exit_config),
            },
        }
        if auto_save_experiment:
            experiment_id = experiment_service().save_portfolio_backtest_result(
                name=experiment_name,
                result=result,
                start_date=str(start_date),
                end_date=str(end_date),
                config=st.session_state["portfolio_experiment_context"]["config"],
                tags=parse_tags(experiment_tags),
            )
            st.success(f"实验已自动保存：{experiment_id}")

    result = st.session_state.get("portfolio_result")
    if not result:
        st.info("设置参数后运行组合回测。")
        return
    metrics = result["metrics"]
    render_portfolio_metrics(metrics, st.session_state.get("portfolio_initial_cash", float(initial_cash)))
    equity = result["portfolio_equity_curve"]
    trades = result["trades"]
    orders = result["orders"]
    positions = result["positions_snapshot"]

    if st.button("保存本次组合回测实验"):
        context = st.session_state.get("portfolio_experiment_context", {})
        experiment_id = experiment_service().save_portfolio_backtest_result(
            name=experiment_name,
            result=result,
            start_date=context.get("start_date", str(start_date)),
            end_date=context.get("end_date", str(end_date)),
            config=context.get("config", {}),
            tags=parse_tags(experiment_tags),
        )
        st.success(f"实验已保存：{experiment_id}，可前往实验中心查看。")

    left, right = st.columns(2)
    left.plotly_chart(plot_portfolio_equity_curve(equity), use_container_width=True)
    right.plotly_chart(plot_portfolio_drawdown_curve(result["portfolio_drawdown_curve"]), use_container_width=True)
    left.plotly_chart(plot_monthly_returns(equity), use_container_width=True)
    right.plotly_chart(plot_position_count(equity), use_container_width=True)
    left.plotly_chart(plot_cost_breakdown(trades if not trades.empty else orders), use_container_width=True)
    right.plotly_chart(plot_gross_vs_net_return(metrics), use_container_width=True)

    tabs = st.tabs(["交易明细", "订单明细", "每日权益", "持仓快照"])
    tabs[0].dataframe(with_display_strategy_names(trades), use_container_width=True)
    tabs[1].dataframe(with_display_strategy_names(orders), use_container_width=True)
    tabs[2].dataframe(equity, use_container_width=True)
    tabs[3].dataframe(with_display_strategy_names(positions), use_container_width=True)


def strategy_stability_page() -> None:
    st.header("策略稳定性")
    factors = load_factor_data()
    if factors.empty:
        st.warning("暂无本地行情数据。请先在数据中心更新数据后再运行稳定性分析。")
        return

    factors = factors.copy()
    factors["date"] = pd.to_datetime(factors["date"])
    min_date = factors["date"].min().date()
    max_date = factors["date"].max().date()

    col1, col2, col3 = st.columns(3)
    with col1:
        strategy_name = select_strategy("策略名称", "stability_strategy")
    with col2:
        start_date = st.date_input("分析起始日期", value=min_date, min_value=min_date, max_value=max_date, key="stability_start")
    with col3:
        end_date = st.date_input("分析结束日期", value=max_date, min_value=min_date, max_value=max_date, key="stability_end")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        train_ratio = st.slider("样本内比例", 0.5, 0.9, 0.7, 0.05, key="stability_train_ratio")
    with col2:
        train_months = st.number_input("Walk-forward 训练月数", min_value=1, max_value=120, value=24, key="stability_train_months")
    with col3:
        test_months = st.number_input("Walk-forward 测试月数", min_value=1, max_value=60, value=6, key="stability_test_months")
    with col4:
        step_months = st.number_input("Walk-forward 步长月数", min_value=1, max_value=60, value=6, key="stability_step_months")

    default_grid = DEFAULT_STABILITY_PARAMETER_GRIDS.get(strategy_name, {})
    parameter_grid_text = st.text_area(
        "参数网格 JSON",
        value=json.dumps(default_grid, ensure_ascii=False, indent=2),
        height=180,
        key=f"stability_grid_{strategy_name}",
    )
    try:
        parameter_grid = json.loads(parameter_grid_text) if parameter_grid_text.strip() else {}
    except json.JSONDecodeError as exc:
        st.error(f"参数网格 JSON 无法解析：{exc}")
        return

    cost_config = cost_config_panel(prefix="stability_")
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        experiment_name = st.text_input("实验名称", value=f"{display_strategy_name(strategy_name)} 稳定性分析", key="stability_exp_name")
    with exp_col2:
        experiment_tags = st.text_input("实验标签", value="stability, overfit", key="stability_exp_tags")
    with exp_col3:
        notes = st.text_input("备注", value="V3.2 stability analysis", key="stability_notes")

    context = {
        "strategy_name": strategy_name,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "parameter_grid": parameter_grid,
        "train_ratio": float(train_ratio),
        "walk_forward_config": {
            "train_months": int(train_months),
            "test_months": int(test_months),
            "step_months": int(step_months),
        },
        "cost_config": config_to_dict(cost_config),
        "position_config": {},
        "exit_config": {},
        "market_config": {},
    }

    if st.button("运行稳定性分析"):
        with st.spinner("正在运行样本内 / 样本外、Walk-forward、参数敏感性和过拟合检测..."):
            result = stability_service().run_stability_analysis(
                strategy_name=strategy_name,
                parameter_grid=parameter_grid,
                start_date=str(start_date),
                end_date=str(end_date),
                train_ratio=float(train_ratio),
                walk_forward_config=context["walk_forward_config"],
                cost_config=cost_config,
            )
        st.session_state["stability_result"] = result
        st.session_state["stability_context"] = context

    result = st.session_state.get("stability_result")
    if not result:
        st.info("设置参数后运行稳定性分析。样本外表现比样本内表现更重要。")
        return
    render_stability_result(result)

    if st.button("保存为稳定性实验"):
        context = st.session_state.get("stability_context", context)
        experiment_id = experiment_service().save_stability_analysis_result(
            name=experiment_name,
            result=result,
            start_date=context.get("start_date", str(start_date)),
            end_date=context.get("end_date", str(end_date)),
            strategy_name=context.get("strategy_name", strategy_name),
            config={
                "strategy_config": {
                    "parameter_grid": context.get("parameter_grid", {}),
                    "best_params": result.get("best_params", {}),
                },
                "cost_config": context.get("cost_config", {}),
                "position_config": context.get("position_config", {}),
                "exit_config": context.get("exit_config", {}),
                "market_config": context.get("market_config", {}),
                "walk_forward_config": context.get("walk_forward_config", {}),
                "train_ratio": context.get("train_ratio", train_ratio),
            },
            tags=parse_tags(experiment_tags),
            notes=notes,
        )
        st.success(f"稳定性实验已保存：{experiment_id}")


def render_stability_result(result: dict[str, object]) -> None:
    raw_stability = result.get("stability_score", {}) or {}
    stability = dict(raw_stability) if isinstance(raw_stability, dict) else {"stability_score": raw_stability}
    raw_overfit = result.get("overfit_result", {}) or {}
    overfit = dict(raw_overfit) if isinstance(raw_overfit, dict) else {}
    if "overfit_risk" in result and "overfit_risk" not in overfit:
        overfit["overfit_risk"] = result.get("overfit_risk")
    if "overfit_score" in result and "overfit_score" not in overfit:
        overfit["overfit_score"] = result.get("overfit_score")
    in_sample = dict(result.get("in_sample_result", {}) or {})
    out_sample = dict(result.get("out_of_sample_result", {}) or {})
    walk_forward = dict(result.get("walk_forward_result", {}) or {})
    sensitivity = dict(result.get("sensitivity_result", {}) or {})
    regimes = dict(result.get("market_regime_breakdown", {}) or {})

    cols = st.columns(7)
    cols[0].metric("稳定性评分", num(stability.get("stability_score", 0)))
    cols[1].metric("过拟合风险", overfit.get("overfit_risk", "-"))
    cols[2].metric("样本内 net_CAGR", pct(in_sample.get("net_CAGR", 0)))
    cols[3].metric("样本外 net_CAGR", pct(out_sample.get("net_CAGR", 0)))
    cols[4].metric("WF 正收益比例", pct(dict(walk_forward.get("summary", {}) or {}).get("positive_test_fold_ratio", 0)))
    cols[5].metric("参数稳定性", num(sensitivity.get("sensitivity_score", 0)))
    cols[6].metric("评级", stability.get("final_rating", "-"))

    left, right = st.columns(2)
    left.plotly_chart(plot_in_sample_vs_out_sample(in_sample, out_sample), use_container_width=True)
    right.plotly_chart(plot_walk_forward_returns(walk_forward), use_container_width=True)
    left.plotly_chart(plot_parameter_sensitivity_heatmap(sensitivity), use_container_width=True)
    right.plotly_chart(plot_regime_stability(regimes), use_container_width=True)
    st.plotly_chart(plot_overfit_risk_flags(overfit), use_container_width=True)

    tabs = st.tabs(["Walk-forward folds", "参数搜索 Top", "市场环境", "过拟合 flags"])
    folds = pd.DataFrame(walk_forward.get("folds", []) or [])
    sensitivity_table = pd.DataFrame(sensitivity.get("top_10_params", []) or sensitivity.get("parameter_table", []) or [])
    regime_table = pd.DataFrame.from_dict(regimes, orient="index").reset_index(names="regime") if regimes else pd.DataFrame()
    flags = pd.DataFrame({"risk_flag": overfit.get("risk_flags", []) or []})
    tabs[0].dataframe(folds, use_container_width=True)
    tabs[1].dataframe(sensitivity_table, use_container_width=True)
    tabs[2].dataframe(regime_table, use_container_width=True)
    tabs[3].dataframe(flags, use_container_width=True)


def _unused_experiment_center_page_v31() -> None:
    st.header("实验中心")
    service = experiment_service()
    experiments = service.build_experiment_summary()
    tabs = st.tabs(["实验列表", "实验详情", "实验对比", "报告导出"])

    with tabs[0]:
        _unused_experiment_list_tab_v31(service, experiments)
    with tabs[1]:
        _unused_experiment_detail_tab_v31(service, experiments)
    with tabs[2]:
        experiment_compare_tab(service, experiments)
    with tabs[3]:
        _unused_experiment_export_tab_v31(service, experiments)


def _unused_experiment_list_tab_v31(service: ExperimentService, experiments: pd.DataFrame) -> None:
    if experiments.empty:
        st.info("当前暂无实验记录。请先在“组合回测”或“回测中心”中运行并保存一次实验。")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        type_options = [""] + sorted(experiments["experiment_type"].dropna().unique().tolist())
        experiment_type = st.selectbox("实验类型", type_options, format_func=lambda value: value or "全部")
    with col2:
        strategy_options = [""] + sorted(experiments["strategy_name"].dropna().unique().tolist())
        strategy_name = st.selectbox("策略名称", strategy_options, format_func=lambda value: value or "全部")
    with col3:
        rating = st.selectbox("评级", [""] + RATING_OPTIONS, format_func=lambda value: rating_display_name(value) if value else "全部")
    with col4:
        tag = st.text_input("标签筛选")

    sort_by = st.selectbox(
        "排序字段",
        ["created_at", "net_CAGR", "max_drawdown", "sharpe_ratio", "final_equity"],
        index=0,
    )
    filtered = filter_experiment_table(
        experiments,
        experiment_type=experiment_type or None,
        strategy_name=strategy_name or None,
        tag=tag or None,
        rating=rating or None,
    )
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=sort_by == "max_drawdown", na_position="last")

    display_columns = [
        "name",
        "experiment_type",
        "strategy_name",
        "start_date",
        "end_date",
        "net_CAGR",
        "max_drawdown",
        "sharpe_ratio",
        "win_rate",
        "cost_impact",
        "rating",
        "created_at",
    ]
    st.dataframe(filtered.loc[:, [column for column in display_columns if column in filtered.columns]], use_container_width=True)

    selected_id = select_experiment_id(filtered, "list_select_experiment")
    if selected_id:
        st.session_state["selected_experiment_id"] = selected_id
        confirm = st.checkbox("确认删除所选实验", key="confirm_delete_experiment")
        if st.button("删除实验", disabled=not confirm):
            service.store.delete_experiment(selected_id)
            st.success("实验已删除，请刷新实验列表。")


def _unused_experiment_detail_tab_v31(service: ExperimentService, experiments: pd.DataFrame) -> None:
    selected_id = st.session_state.get("selected_experiment_id") or select_experiment_id(experiments, "detail_select_experiment")
    if not selected_id:
        st.info("请选择一个实验查看详情。")
        return
    detail = service.build_experiment_detail(selected_id)
    if detail["run"] is None:
        st.warning("实验不存在或已删除。")
        return
    run = detail["run"]
    metrics = run["metrics_json"]

    cols = st.columns(4)
    cols[0].metric("实验类型", run["experiment_type"])
    cols[1].metric("策略", run.get("strategy_name") or "-")
    cols[2].metric("评级", rating_display_name(run.get("rating")))
    cols[3].metric("Git Commit", (run.get("git_commit") or "-")[:8])

    render_portfolio_metrics(metrics, float(metrics.get("initial_cash", 1_000_000)))
    st.json(run["config_json"], expanded=False)

    equity = experiment_equity_for_chart(detail["equity_curve"])
    left, right = st.columns(2)
    left.plotly_chart(plot_portfolio_equity_curve(equity), use_container_width=True)
    right.plotly_chart(plot_portfolio_drawdown_curve(equity.loc[:, ["date", "drawdown"]] if "drawdown" in equity.columns else pd.DataFrame()), use_container_width=True)
    left.plotly_chart(plot_cost_breakdown(detail["trades"]), use_container_width=True)

    table_tabs = st.tabs(["Trades", "Orders", "Daily Equity"])
    table_tabs[0].dataframe(detail["trades"], use_container_width=True)
    table_tabs[1].dataframe(detail["orders"], use_container_width=True)
    table_tabs[2].dataframe(detail["equity_curve"], use_container_width=True)

    notes = st.text_area("Notes", value=run.get("notes") or "", key=f"notes_{selected_id}")
    conclusion = st.text_area("Conclusion", value=run.get("conclusion") or "", key=f"conclusion_{selected_id}")
    tags = st.text_input("Tags", value=", ".join(run.get("tags") or []), key=f"tags_{selected_id}")
    rating = st.selectbox(
        "Rating",
        RATING_OPTIONS,
        index=RATING_OPTIONS.index(run.get("rating") or "unrated"),
        format_func=rating_display_name,
        key=f"rating_{selected_id}",
    )
    if st.button("保存备注 / 标签 / 评级"):
        service.store.update_notes(selected_id, notes=notes, conclusion=conclusion)
        service.store.update_tags(selected_id, parse_tags(tags))
        service.store.update_rating(selected_id, rating)
        st.success("实验研究记录已更新。")


def experiment_compare_tab(service: ExperimentService, experiments: pd.DataFrame) -> None:
    if experiments.empty or len(experiments) < 2:
        st.info("至少需要 2 个实验才能进行对比。")
        return
    labels = {experiment_option_label(row): row["id"] for _, row in experiments.iterrows()}
    default_labels = list(labels.keys())[:2]
    selected_labels = st.multiselect("选择 2 到 5 个实验", list(labels.keys()), default=default_labels, max_selections=5)
    if len(selected_labels) < 2:
        st.info("请选择至少 2 个实验。")
        return

    result = compare_experiments([labels[label] for label in selected_labels], store=service.store)
    st.subheader("指标对比")
    st.dataframe(result["metrics_comparison"], use_container_width=True)
    st.subheader("最佳实验")
    st.json(result["winner_by_metric"], expanded=False)
    if not result["normalized_equity_curves"].empty:
        st.line_chart(result["normalized_equity_curves"].set_index("date"))
    if not result["drawdown_curves"].empty:
        st.line_chart(result["drawdown_curves"].set_index("date"))
    if not result["cost_comparison"].empty:
        st.bar_chart(result["cost_comparison"].set_index("name")[["total_cost", "cost_impact"]])


def _unused_experiment_export_tab_v31(service: ExperimentService, experiments: pd.DataFrame) -> None:
    selected_id = select_experiment_id(experiments, "export_select_experiment")
    if not selected_id:
        st.info("请选择一个实验导出报告。")
        return
    exports = [
        ("导出 Markdown", lambda: export_markdown_report(service.store, selected_id)),
        ("导出 Trades CSV", lambda: export_experiment_csv(service.store, selected_id, "trades")),
        ("导出 Orders CSV", lambda: export_experiment_csv(service.store, selected_id, "orders")),
        ("导出 Equity CSV", lambda: export_experiment_csv(service.store, selected_id, "equity_curve")),
        ("导出 Config JSON", lambda: export_config_json(service.store, selected_id)),
    ]
    for label, action in exports:
        if st.button(label):
            path = action()
            st.success(f"已导出：{path}")
            data = path.read_bytes()
            st.download_button(f"下载 {path.name}", data=data, file_name=path.name)


def _legacy_select_experiment_id(experiments: pd.DataFrame, key: str) -> str | None:
    if experiments.empty:
        return None
    labels = {experiment_option_label(row): row["id"] for _, row in experiments.iterrows()}
    selected = st.selectbox("选择实验", list(labels), key=key)
    return labels.get(selected)


def today_picks() -> None:
    st.header("今日选股")
    factors = load_factor_data()
    if factors.empty:
        st.warning("数据库暂无日线数据")
        return
    latest_date = pd.to_datetime(factors["date"]).max()
    signals = run_all_strategies(factors, signal_date=latest_date)
    ranked = rank_recommendations(signals)
    st.caption(f"信号日期：{latest_date.date()}")
    if ranked.empty:
        st.info("今日无符合条件股票")
        st.write("可能原因：")
        st.write("- 条件过严")
        st.write("- 市场环境不匹配")
        st.write("建议：")
        st.write("- 调低成交额阈值")
        st.write("- 更换策略")
        return
    st.dataframe(with_display_strategy_names(ranked), use_container_width=True)


def stock_analysis() -> None:
    st.header("个股分析")
    code = st.text_input("股票代码", value="000001", max_chars=6)
    daily = load_daily_data(code=code)
    if daily.empty:
        st.warning("该股票暂无本地数据")
        return
    factors = compute_factors(daily)
    stock_name = factors["name"].dropna().iloc[-1] if not factors["name"].dropna().empty else code
    st.plotly_chart(plot_price_with_ma(factors, f"{code} {stock_name}"), use_container_width=True)
    factor_columns = [
        "date",
        "close",
        "ma20",
        "ma60",
        "return_5d",
        "return_20d",
        "volume_ratio",
        "volatility_20d",
        "breakout_20d",
        "breakout_60d",
        "distance_to_ma20",
        "max_drawdown_20d",
    ]
    st.subheader("最近因子")
    st.dataframe(factors.loc[:, factor_columns].tail(20), use_container_width=True)
    signals = stock_signals(factors, code)
    st.subheader("策略统计")
    render_stock_signal_stats(factors, signals)
    st.subheader("历史策略信号")
    st.dataframe(with_display_strategy_names(signals), use_container_width=True)


def build_strategy_signal_history(
    factors: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    signal_dates = sorted(pd.to_datetime(factors.loc[factors["date"].between(start_date, end_date), "date"]).unique())
    result = {}
    for strategy_name, strategy_func in STRATEGIES.items():
        frames = []
        for signal_date in signal_dates:
            signals = strategy_func(factors, signal_date=signal_date)
            if not signals.empty:
                frames.append(signals)
        result[strategy_name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return result


def load_market_regime(factors: pd.DataFrame) -> pd.DataFrame:
    index_prices = get_index_prices()
    if index_prices.empty:
        st.info("本地没有沪深300指数数据，当前使用股票池等权代理指数进行市场环境分析。")
        market_proxy = build_market_proxy_from_prices(factors)
        return classify_market_regime(market_proxy)
    return classify_market_regime(index_prices)


def stock_signals(factors: pd.DataFrame, code: str) -> pd.DataFrame:
    recent_dates = sorted(pd.to_datetime(factors["date"]).drop_duplicates().tail(120))
    signal_frames = []
    normalized_code = str(code).zfill(6)
    for signal_date in recent_dates:
        frame = run_all_strategies(factors, signal_date=signal_date)
        if not frame.empty:
            signal_frames.append(frame[frame["code"].eq(normalized_code)])
    return pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()


def render_stock_signal_stats(factors: pd.DataFrame, signals: pd.DataFrame) -> None:
    if signals.empty:
        cols = st.columns(4)
        cols[0].metric("触发次数", "0")
        cols[1].metric("平均收益", "-")
        cols[2].metric("最佳策略", "-")
        cols[3].metric("最差策略", "-")
        return
    result = backtest_signals(factors, signals, holding_days=20, take_profit_pct=0.10, stop_loss_pct=0.08, sell_on_ma20_break=True)
    trades = result["trade_details"]
    if trades.empty:
        avg_return = 0
        best_strategy = "-"
        worst_strategy = "-"
    else:
        grouped = trades.groupby("strategy_name")["return"].mean().sort_values(ascending=False)
        avg_return = trades["return"].mean()
        best_strategy = display_strategy_name(grouped.index[0])
        worst_strategy = display_strategy_name(grouped.index[-1])
    cols = st.columns(4)
    cols[0].metric("触发次数", str(len(signals)))
    cols[1].metric("平均收益", pct(avg_return))
    cols[2].metric("最佳策略", best_strategy)
    cols[3].metric("最差策略", worst_strategy)


def experiment_list_tab(service: ExperimentService, experiments: pd.DataFrame) -> None:
    if experiments.empty:
        st.info("当前暂无实验记录。请先在“组合回测”或“回测中心”中运行并保存一次实验。")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        available_types = set(experiments["experiment_type"].dropna().astype(str).tolist())
        type_options = [value for value in EXPERIMENT_TYPE_OPTIONS if value == "all" or value in available_types]
        experiment_type = st.selectbox("实验类型", type_options)
    with col2:
        strategy_options = ["all"] + sorted(experiments["strategy_name"].dropna().astype(str).unique().tolist())
        strategy_name = st.selectbox("策略名称", strategy_options)
    with col3:
        rating = st.selectbox(
            "评级",
            ["all"] + RATING_OPTIONS,
            format_func=lambda value: "all" if value == "all" else rating_display_name(value),
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        created_start = st.date_input("创建起始日期", value=None)
    with col5:
        created_end = st.date_input("创建结束日期", value=None)
    with col6:
        tag = st.text_input("标签关键字")
    keyword = st.text_input("关键词搜索（名称 / 描述 / 备注 / 结论）")

    sort_col, direction_col = st.columns(2)
    with sort_col:
        sort_by = st.selectbox("排序字段", SORTABLE_EXPERIMENT_COLUMNS, index=0)
    with direction_col:
        direction = st.selectbox("排序方向", ["降序 / 最佳优先", "升序 / 反向"], index=0)

    filtered = filter_experiment_table(
        experiments,
        experiment_type=experiment_type,
        strategy_name=strategy_name,
        start_date=created_start,
        end_date=created_end,
        tag=tag or None,
        rating=rating,
        keyword=keyword or None,
    )
    filtered = sort_experiment_table(filtered, sort_by, ascending=direction.startswith("升序"))

    if filtered.empty:
        st.info("没有找到符合条件的实验，请调整筛选条件。")
        return

    display_columns = [
        "name",
        "experiment_type",
        "strategy_name",
        "start_date",
        "end_date",
        "net_CAGR",
        "max_drawdown",
        "sharpe_ratio",
        "win_rate",
        "cost_impact",
        "rating",
        "tags",
        "created_at",
    ]
    st.dataframe(filtered.loc[:, [column for column in display_columns if column in filtered.columns]], use_container_width=True)

    selected_id = select_experiment_id(filtered, "list_select_experiment")
    if not selected_id:
        return

    st.session_state["selected_experiment_id"] = selected_id
    selected_row = filtered.loc[filtered["id"].eq(selected_id)].iloc[0].to_dict()
    if st.button("准备删除", key="prepare_delete_experiment"):
        st.session_state["delete_experiment_id"] = selected_id
    if st.session_state.get("delete_experiment_id") == selected_id:
        experiment_name = str(selected_row.get("name") or "")
        st.warning(f"删除实验将同时删除资金曲线、交易、订单和导出记录。请输入实验名称确认：{experiment_name}")
        confirmation = st.text_input("输入实验名称确认删除", key=f"delete_confirm_{selected_id}")
        can_delete = delete_confirmation_matches(experiment_name, confirmation)
        if st.button("确认删除实验", disabled=not can_delete, key=f"delete_button_{selected_id}"):
            deleted = service.store.delete_experiment(selected_id)
            st.session_state.pop("delete_experiment_id", None)
            st.session_state.pop("selected_experiment_id", None)
            if deleted:
                st.success("实验已删除")
            else:
                st.info("实验不存在，可能已被删除。")
            st.rerun()


def experiment_detail_tab(service: ExperimentService, experiments: pd.DataFrame) -> None:
    selected_id = st.session_state.get("selected_experiment_id")
    if not selected_id:
        selected_id = select_experiment_id(experiments, "detail_select_experiment", default_first=False)
    if not selected_id:
        st.info("请先在实验列表中选择一个实验。")
        return

    detail = service.build_experiment_detail(selected_id)
    if detail["run"] is None:
        st.warning("未找到该实验，可能已被删除。")
        return
    run = detail["run"]
    metrics = run["metrics_json"]

    cols = st.columns(4)
    cols[0].metric("实验类型", run["experiment_type"])
    cols[1].metric("策略", run.get("strategy_name") or "-")
    cols[2].metric("评级", rating_display_name(run.get("rating")))
    cols[3].metric("Git Commit", (run.get("git_commit") or "-")[:8])

    render_portfolio_metrics(metrics, float(metrics.get("initial_cash", 1_000_000)))

    config = run["config_json"]
    for key in ["strategy_config", "cost_config", "position_config", "exit_config", "market_config"]:
        with st.expander(key, expanded=False):
            st.json(config.get(key, {}), expanded=False)
    extra_config = {key: value for key, value in config.items() if key not in {"strategy_config", "cost_config", "position_config", "exit_config", "market_config"}}
    if extra_config:
        with st.expander("other_config", expanded=False):
            st.json(extra_config, expanded=False)

    if run.get("experiment_type") == "strategy_stability_analysis":
        st.subheader("稳定性分析详情")
        render_stability_result(metrics)

    equity = experiment_equity_for_chart(detail["equity_curve"])
    left, right = st.columns(2)
    left.plotly_chart(plot_portfolio_equity_curve(equity), use_container_width=True)
    right.plotly_chart(plot_portfolio_drawdown_curve(equity.loc[:, ["date", "drawdown"]] if "drawdown" in equity.columns else pd.DataFrame()), use_container_width=True)
    left.plotly_chart(plot_cost_breakdown(detail["trades"]), use_container_width=True)

    table_tabs = st.tabs(["Trades", "Orders", "Daily Equity"])
    table_tabs[0].dataframe(detail["trades"], use_container_width=True)
    table_tabs[1].dataframe(detail["orders"], use_container_width=True)
    table_tabs[2].dataframe(detail["equity_curve"], use_container_width=True)

    notes = st.text_area("Notes", value=run.get("notes") or "", key=f"notes_{selected_id}")
    conclusion = st.text_area("Conclusion", value=run.get("conclusion") or "", key=f"conclusion_{selected_id}")
    tags = st.text_input("Tags", value=", ".join(run.get("tags") or []), key=f"tags_{selected_id}")
    rating_value = run.get("rating") if run.get("rating") in RATING_OPTIONS else "unrated"
    rating = st.selectbox(
        "Rating",
        RATING_OPTIONS,
        index=RATING_OPTIONS.index(rating_value),
        format_func=rating_display_name,
        key=f"rating_{selected_id}",
    )
    if st.button("保存备注 / 结论"):
        service.store.update_notes(selected_id, notes=notes, conclusion=conclusion)
        st.success("备注和结论已保存。")
    if st.button("保存标签"):
        service.store.update_tags(selected_id, parse_tags(tags))
        st.success("标签已保存。")
        st.rerun()
    if st.button("保存评级"):
        service.store.update_rating(selected_id, rating)
        st.success("评级已保存。")
        st.rerun()


def experiment_export_tab(service: ExperimentService, experiments: pd.DataFrame) -> None:
    selected_id = select_experiment_id(experiments, "export_select_experiment")
    if not selected_id:
        st.info("请选择一个实验导出报告。")
        return
    exports = [
        ("导出 Markdown", lambda: export_markdown_report(service.store, selected_id)),
        ("导出 Trades CSV", lambda: export_experiment_csv(service.store, selected_id, "trades")),
        ("导出 Orders CSV", lambda: export_experiment_csv(service.store, selected_id, "orders")),
        ("导出 Equity CSV", lambda: export_experiment_csv(service.store, selected_id, "equity_curve")),
        ("导出 Config JSON", lambda: export_config_json(service.store, selected_id)),
    ]
    for label, action in exports:
        if st.button(label):
            path = action()
            size = path.stat().st_size
            generated_at = pd.Timestamp.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"已导出：{path}")
            st.caption(f"文件大小：{size} bytes | 生成时间：{generated_at}")
            data = path.read_bytes()
            st.download_button(f"下载 {path.name}", data=data, file_name=path.name)


def select_experiment_id(experiments: pd.DataFrame, key: str, default_first: bool = True) -> str | None:
    if experiments.empty:
        return None
    labels = {experiment_option_label(row): row["id"] for _, row in experiments.iterrows()}
    options = list(labels)
    if default_first:
        selected = st.selectbox("选择实验", options, key=key)
        return labels.get(selected)
    selected = st.selectbox("选择实验", [""] + options, key=key, format_func=lambda value: value or "请选择")
    return labels.get(selected) if selected else None


def experiment_center_page() -> None:
    st.header("实验中心")
    service = experiment_service()
    experiments = service.build_experiment_summary()
    tabs = st.tabs(["实验列表", "实验详情", "实验对比", "报告导出"])

    with tabs[0]:
        experiment_list_tab(service, experiments)
    with tabs[1]:
        experiment_detail_tab(service, experiments)
    with tabs[2]:
        experiment_compare_tab(service, experiments)
    with tabs[3]:
        experiment_export_tab(service, experiments)


PAGES = {
    "数据中心": data_center,
    "策略实验室": strategy_lab,
    "回测中心": backtest_center,
    "参数优化器": parameter_optimizer_page,
    "策略排行榜": strategy_leaderboard_page,
    "组合回测": portfolio_backtest_page,
    "策略稳定性": strategy_stability_page,
    "实验中心": experiment_center_page,
    "今日选股": today_picks,
    "个股分析": stock_analysis,
}


def main() -> None:
    st.title("A股量化研究平台")
    page = st.sidebar.radio("功能", list(PAGES.keys()))
    PAGES[page]()


if __name__ == "__main__":
    main()
