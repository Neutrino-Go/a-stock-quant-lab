from __future__ import annotations

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
from report.charts import (
    plot_cost_breakdown,
    plot_drawdown_curve,
    plot_equity_curve,
    plot_gross_vs_net_return,
    plot_monthly_returns,
    plot_portfolio_drawdown_curve,
    plot_portfolio_equity_curve,
    plot_position_count,
    plot_price_with_ma,
    plot_trade_returns,
)
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

        render_metrics(result)
        trades = result["trade_details"]
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
        st.dataframe(result, use_container_width=True)


def strategy_leaderboard_page() -> None:
    st.header("策略排行榜")
    factors = load_factor_data()
    if factors.empty:
        st.warning("数据库暂无日线数据")
        return
    latest_date = pd.to_datetime(factors["date"]).max()
    with st.spinner("正在统计策略排行榜"):
        leaderboard = build_strategy_leaderboard(factors, as_of=latest_date)
    st.caption(f"统计截至：{latest_date.date()}，排行榜使用 net result 口径。")
    st.dataframe(leaderboard, use_container_width=True)


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
        allow_add_position = st.checkbox("允许同股加仓", value=False)

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
        allow_add_position=allow_add_position,
        enable_limit_check=enable_limit_check,
        daily_top_n=int(daily_top_n),
    )

    if st.button("运行组合回测"):
        with st.spinner("正在生成策略组合信号并执行组合回测"):
            signals_by_strategy = build_strategy_signal_history(factors, pd.Timestamp(start_date), pd.Timestamp(end_date))
            ensemble = combine_strategy_signals(signals_by_strategy, weights=weights, top_n_per_day=int(daily_top_n))
            engine = PortfolioBacktestEngine(cost_config=cost_config, position_config=position_config, exit_config=exit_config)
            result = engine.run(ensemble, factors, start_date=pd.Timestamp(start_date), end_date=pd.Timestamp(end_date))
        st.session_state["portfolio_result"] = result
        st.session_state["portfolio_initial_cash"] = float(initial_cash)

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


PAGES = {
    "数据中心": data_center,
    "策略实验室": strategy_lab,
    "回测中心": backtest_center,
    "参数优化器": parameter_optimizer_page,
    "策略排行榜": strategy_leaderboard_page,
    "组合回测": portfolio_backtest_page,
    "今日选股": today_picks,
    "个股分析": stock_analysis,
}


def main() -> None:
    st.title("A股量化研究平台")
    page = st.sidebar.radio("功能", list(PAGES.keys()))
    PAGES[page]()


if __name__ == "__main__":
    main()
