from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from backtest.backtest_engine import apply_realistic_costs_to_trades, backtest_signals, run_strategy_backtest
from candidate.candidate_models import CandidateRevalidationConfig
from candidate.candidate_report import (
    export_lifecycle_events_csv,
    export_revalidation_history_csv,
    export_revalidation_markdown_report,
)
from candidate.candidate_revalidator import CandidateRevalidator
from candidate.candidate_store import CandidateStore
from candidate.candidate_ui_helpers import (
    action_to_candidate_status,
    candidate_option_label,
    format_revalidation_mode,
    revalidation_summary_metrics,
    run_batch_revalidation,
)
from data.data_fetcher import DataFetchError, is_daily_provider_available
from data.data_updater import DataUpdateService
from data.database import get_database_status, get_stock_daily, get_stock_list
from data_quality.audit import DataQualityAuditor
from data_quality.data_health_service import DataHealthService
from data_quality.report_exporter import export_data_quality_issues_csv, export_data_quality_markdown_report
from data_quality.ui_helpers import file_info as data_quality_file_info
from data_quality.ui_helpers import parse_code_list
from data_quality.universe import StockUniverseBuilder
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
from workflow.research_models import ResearchConfig
from workflow.research_report import export_research_candidates_csv, export_research_json_summary, export_research_markdown_report
from workflow.research_store import ResearchStore
from workflow.research_templates import get_research_template
from workflow.research_workflow import StrategyResearchWorkflow


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


@st.cache_resource
def research_store() -> ResearchStore:
    return ResearchStore()


@st.cache_resource
def candidate_store() -> CandidateStore:
    return CandidateStore()


@st.cache_resource
def research_workflow_service() -> StrategyResearchWorkflow:
    return StrategyResearchWorkflow(
        research_store=research_store(),
        stability_service=stability_service(),
        factor_data_loader=load_factor_data,
    )


@st.cache_resource
def candidate_revalidator_service() -> CandidateRevalidator:
    return CandidateRevalidator(
        research_store=research_store(),
        candidate_store=candidate_store(),
        stability_service=stability_service(),
        experiment_service=experiment_service(),
    )


@st.cache_resource
def data_health_service() -> DataHealthService:
    return DataHealthService()


@st.cache_resource
def data_quality_auditor() -> DataQualityAuditor:
    return DataQualityAuditor()


@st.cache_resource
def stock_universe_builder() -> StockUniverseBuilder:
    return StockUniverseBuilder()


def clear_data_cache() -> None:
    load_daily_data.clear()
    load_factor_data.clear()


def data_fetch_error_technical_detail(exc: BaseException) -> str:
    if isinstance(exc, DataFetchError) and exc.provider_errors:
        return "\n".join(exc.provider_errors)
    detail = exc.__cause__ if isinstance(exc, DataFetchError) and exc.__cause__ else exc
    return str(detail)


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


def data_quality_page() -> None:
    st.header("数据质量")
    st.caption("数据质量检查用于识别缺失、重复、复权混用和股票池偏差。检查通过不代表策略未来有效。")
    tabs = st.tabs(["数据健康概览", "运行数据审计", "问题明细", "股票池构建", "数据质量报告"])
    auditor = data_quality_auditor()
    health = data_health_service()

    with tabs[0]:
        summary = health.get_database_health_summary()
        columns = st.columns(5)
        columns[0].metric("股票数", summary.get("stock_count", 0))
        columns[1].metric("行情记录", summary.get("daily_price_count", 0))
        columns[2].metric("最新交易日", summary.get("latest_trading_day") or "-")
        columns[3].metric("CRITICAL", summary.get("critical_issue_count", 0))
        columns[4].metric("WARNING", summary.get("warning_issue_count", 0))
        if int(summary.get("critical_issue_count") or 0) > 0:
            st.warning("当前数据存在严重问题，回测结果可能不可靠。")
        st.write("数据区间", summary.get("start_date") or "-", "至", summary.get("end_date") or "-")
        st.write("当前主要 adjust_type", summary.get("adjust_type") or "unknown")
        if summary.get("adjust_type") == "unknown":
            st.warning("当前行情缺少真实 adjust_type，回测复权口径为假设值，结果不可与其他复权口径直接比较。")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("issue_type 分布")
            issue_summary = health.get_issue_summary_by_type()
            if issue_summary.empty:
                st.info("暂无数据质量问题记录。")
            else:
                st.dataframe(issue_summary, use_container_width=True)
        with col2:
            st.subheader("adjust_type 分布")
            st.dataframe(health.get_adjust_type_summary(), use_container_width=True)

        st.subheader("stale stocks")
        stale = health.get_stale_stocks(max_lag_days=7)
        if stale.empty:
            st.info("暂无 stale 股票。")
        else:
            st.dataframe(stale.head(200), use_container_width=True)

        st.subheader("coverage by code")
        coverage = health.get_data_coverage_by_code()
        if coverage.empty:
            st.info("暂无行情覆盖数据。")
        else:
            st.dataframe(coverage.head(200), use_container_width=True)

    with tabs[1]:
        st.info("全市场多年审计可能耗时较久，建议先小范围测试。")
        col1, col2, col3 = st.columns(3)
        with col1:
            scope = st.selectbox("scope", ["daily_prices", "single_stock", "stock_list", "universe", "index_prices", "all"], index=0)
            start = st.date_input("start_date", value=date(2020, 1, 1), key="dq_audit_start")
        with col2:
            adjust_type = st.selectbox("adjust_type", ["qfq", "raw", "hfq", "unknown"], index=0)
            end = st.date_input("end_date", value=date.today(), key="dq_audit_end")
        with col3:
            max_codes = st.number_input("最大检查股票数", min_value=1, max_value=5000, value=50, step=10)
            codes_text = st.text_area("codes", placeholder="000001, 000002", key="dq_audit_codes")
        check_calendar = st.checkbox("检查交易日缺口", value=True)
        check_bias = st.checkbox("检查幸存者偏差", value=True)
        check_adjustment = st.checkbox("检查复权一致性", value=True)
        if st.button("运行审计"):
            with st.spinner("正在运行数据审计"):
                result = auditor.run_audit(
                    scope=scope,
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    codes=parse_code_list(codes_text),
                    adjust_type=adjust_type,
                    check_calendar_gaps=check_calendar,
                    check_survivorship_bias=check_bias,
                    check_adjustment=check_adjustment,
                    max_codes=int(max_codes),
                )
            st.success(f"审计完成：{result.run_id} / {result.status}")
            st.json(result.summary)
            if result.issues:
                st.dataframe(pd.DataFrame([issue.to_record(result.run_id) for issue in result.issues]), use_container_width=True)
            else:
                st.info("本次审计未发现问题。")

    with tabs[2]:
        runs = auditor.list_runs()
        run_options = [""] + (runs["id"].tolist() if not runs.empty and "id" in runs else [])
        col1, col2, col3 = st.columns(3)
        with col1:
            run_id = st.selectbox("run_id", run_options, format_func=lambda value: value or "全部", key="dq_issue_run")
            severity = st.selectbox("severity", ["all", "CRITICAL", "ERROR", "WARNING", "INFO"], key="dq_issue_severity")
        with col2:
            issue_types = ["all"] + sorted(auditor.list_issues()["issue_type"].dropna().unique().tolist()) if not auditor.list_issues().empty else ["all"]
            issue_type = st.selectbox("issue_type", issue_types, key="dq_issue_type")
            code = st.text_input("code", key="dq_issue_code")
        with col3:
            keyword = st.text_input("keyword", key="dq_issue_keyword")
            issue_start = st.date_input("date start", value=None, key="dq_issue_start")
            issue_end = st.date_input("date end", value=None, key="dq_issue_end")
        issues = auditor.list_issues(
            run_id=run_id or None,
            severity=severity,
            issue_type=issue_type,
            code=code or None,
            start_date=issue_start.isoformat() if issue_start else None,
            end_date=issue_end.isoformat() if issue_end else None,
            keyword=keyword or None,
        )
        if issues.empty:
            st.info("没有找到符合条件的问题记录。")
        else:
            st.dataframe(issues, use_container_width=True)
        if st.button("导出当前 issues CSV"):
            path = export_data_quality_issues_csv(issues, Path("reports/data_quality") / "data_quality_issues_export.csv")
            info = data_quality_file_info(path)
            st.success(f"已导出：{info['path']} ({info['size']} bytes)")
            st.download_button("下载 issues CSV", data=Path(path).read_bytes(), file_name=Path(path).name, mime="text/csv")

    with tabs[3]:
        builder = stock_universe_builder()
        col1, col2, col3 = st.columns(3)
        with col1:
            as_of_date = st.date_input("as_of_date", value=date.today(), key="universe_as_of")
            universe_name = st.text_input("universe_name", value="default")
            min_list_days = st.number_input("min_list_days", min_value=0, max_value=5000, value=250)
        with col2:
            exclude_st = st.checkbox("exclude_st", value=True)
            exclude_suspended = st.checkbox("exclude_suspended", value=True)
            exclude_delisted = st.checkbox("exclude_delisted", value=True)
        with col3:
            min_amount = st.number_input("min_amount", min_value=0.0, value=0.0, step=10_000_000.0)
            min_price = st.number_input("min_price", min_value=0.0, value=0.0)
            max_price = st.number_input("max_price", min_value=0.0, value=0.0)
        col4, col5, col6 = st.columns(3)
        with col4:
            market = st.text_input("market", value="", key="universe_market")
        with col5:
            industry = st.text_input("industry", value="", key="universe_industry")
        with col6:
            use_current_stock_list_fallback = st.checkbox("use_current_stock_list_fallback", value=True)
        if st.button("构建股票池"):
            result = builder.build_universe(
                as_of_date=as_of_date.isoformat(),
                universe_name=universe_name,
                min_list_days=int(min_list_days),
                exclude_st=exclude_st,
                exclude_suspended=exclude_suspended,
                exclude_delisted=exclude_delisted,
                min_amount=float(min_amount) if min_amount else None,
                min_price=float(min_price) if min_price else None,
                max_price=float(max_price) if max_price else None,
                market=market or None,
                industry=industry or None,
                use_current_stock_list_fallback=use_current_stock_list_fallback,
            )
            st.session_state["latest_universe_result"] = result.to_dict()
            st.metric("stock_count", result.stock_count)
            for warning in result.warnings:
                st.warning(warning)
            st.dataframe(pd.DataFrame({"code": result.codes}), use_container_width=True)
        if "latest_universe_result" in st.session_state:
            latest = st.session_state["latest_universe_result"]
            if st.button("保存 snapshot"):
                saved_id = builder.save_snapshot_from_dict(latest)
                st.success(f"已保存 snapshot：{saved_id}")
                st.write("stock_count", latest.get("stock_count", 0))
                st.write("warnings count", len(latest.get("warnings", [])))

    with tabs[4]:
        runs = auditor.list_runs()
        if runs.empty:
            st.info("暂无 data_quality_run，请先运行一次审计。")
        else:
            run_id = st.selectbox("选择 data_quality_run", runs["id"].tolist(), key="dq_report_run")
            if st.button("导出 Markdown 报告"):
                path = export_data_quality_markdown_report(auditor, run_id)
                info = data_quality_file_info(path)
                st.success(f"已导出：{info['path']} ({info['size']} bytes)")
                st.download_button("下载 Markdown", data=Path(path).read_bytes(), file_name=Path(path).name, mime="text/markdown")
            if st.button("导出 issues CSV"):
                path = export_data_quality_issues_csv(auditor.list_issues(run_id=run_id), Path("reports/data_quality") / f"data_quality_{run_id}_issues.csv")
                info = data_quality_file_info(path)
                st.success(f"已导出：{info['path']} ({info['size']} bytes)")
                st.download_button("下载 CSV", data=Path(path).read_bytes(), file_name=Path(path).name, mime="text/csv")


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
        provider = st.selectbox(
            "数据源",
            options=["auto", "eastmoney", "sina"],
            index=0,
            help="auto 会先尝试东方财富，失败后自动尝试新浪 fallback。",
        )
        if provider == "sina" and not is_daily_provider_available("sina"):
            st.warning("当前 AKShare 环境没有可用的新浪 fallback 接口。")
        if st.button("更新单只股票"):
            try:
                with st.spinner("正在更新日线"):
                    update_result = service.update_stock_daily(
                        code=code,
                        start_date=start.strftime("%Y%m%d"),
                        end_date=end.strftime("%Y%m%d"),
                        force=force,
                        preferred_provider=provider,
                    )
            except (DataFetchError, requests.exceptions.ProxyError) as exc:
                st.error("A股日线数据获取失败。东方财富失败时系统会尝试 fallback，但本次没有成功。")
                st.info(
                    "建议：数据源选择 auto；关闭 VPN / Clash / v2rayN / 代理软件后重试；"
                    "或使用 scripts/windows/start_streamlit_no_proxy.bat 启动。"
                    "如果本地已有数据，可以先继续测试个股分析、回测、实验中心。"
                )
                if isinstance(exc, DataFetchError) and exc.provider_errors:
                    st.warning("数据源尝试结果：" + " | ".join(exc.provider_errors))
                with st.expander("技术详情"):
                    st.code(data_fetch_error_technical_detail(exc))
            else:
                clear_data_cache()
                st.success(f"写入/更新 {update_result.rows} 行")
                st.caption(f"数据源：{update_result.provider or provider}")
                for warning in update_result.warnings:
                    st.warning(warning)

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
                result = service.update_batch(
                    codes=codes or None,
                    limit=int(limit),
                    start_date=batch_start.strftime("%Y%m%d"),
                    end_date=batch_end.strftime("%Y%m%d"),
                    preferred_provider="auto",
                )
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
    regime_source = dict(regimes.get("_meta", {}) or {}).get("regime_source", "unavailable")
    st.caption(f"市场环境来源：{market_regime_source_label(regime_source)}")
    st.plotly_chart(plot_overfit_risk_flags(overfit), use_container_width=True)

    tabs = st.tabs(["Walk-forward folds", "参数搜索 Top", "市场环境", "过拟合 flags"])
    folds = pd.DataFrame(walk_forward.get("folds", []) or [])
    sensitivity_table = pd.DataFrame(sensitivity.get("top_20_params", []) or sensitivity.get("parameter_table", []) or [])
    regime_rows = {key: value for key, value in regimes.items() if not str(key).startswith("_")}
    regime_table = pd.DataFrame.from_dict(regime_rows, orient="index").reset_index(names="regime") if regime_rows else pd.DataFrame()
    flags = pd.DataFrame({"risk_flag": overfit.get("risk_flags", []) or []})
    tabs[0].dataframe(folds, use_container_width=True)
    tabs[1].dataframe(sensitivity_table, use_container_width=True)
    tabs[2].dataframe(regime_table, use_container_width=True)
    tabs[3].dataframe(flags, use_container_width=True)


def market_regime_source_label(regime_source: object) -> str:
    labels = {
        "HS300": "沪深300",
        "fallback_equal_weight_proxy": "股票池等权代理指数 fallback，仅供参考",
        "unavailable": "不可用",
    }
    return labels.get(str(regime_source), str(regime_source))


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


def research_workflow_page() -> None:
    st.header("研究工作流")
    tabs = st.tabs(["新建研究", "研究任务", "候选策略库", "研究报告"])
    store = research_store()
    with tabs[0]:
        research_new_tab()
    with tabs[1]:
        research_runs_tab(store)
    with tabs[2]:
        research_candidates_tab(store)
    with tabs[3]:
        research_reports_tab(store)


def research_new_tab() -> None:
    factors = load_factor_data()
    if factors.empty:
        st.warning("暂无本地行情数据。请先在数据中心更新数据后再运行研究工作流。")
        return
    factors = factors.copy()
    factors["date"] = pd.to_datetime(factors["date"])
    min_date = factors["date"].min().date()
    max_date = factors["date"].max().date()

    col1, col2, col3 = st.columns(3)
    with col1:
        research_name = st.text_input("研究任务名称", value="V3.3 strategy research", key="research_name")
        strategy_name = select_strategy("策略名称", "research_strategy")
    with col2:
        template_name = st.selectbox("研究模板", ["quick", "standard", "deep", "custom"], index=1)
        start_date = st.date_input("研究起始日期", value=min_date, min_value=min_date, max_value=max_date, key="research_start")
    with col3:
        end_date = st.date_input("研究结束日期", value=max_date, min_value=min_date, max_value=max_date, key="research_end")
        top_n = st.number_input(
            "Top N 参数组合",
            min_value=1,
            max_value=20,
            value=int(get_research_template(template_name)["top_n_params"]),
        )

    template = get_research_template(template_name)
    wf_defaults = template["walk_forward_config"]
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        train_ratio = st.slider("样本内比例", 0.5, 0.9, float(template["train_ratio"]), 0.05, key="research_train_ratio")
    with col5:
        train_months = st.number_input("WF 训练月数", min_value=1, max_value=120, value=int(wf_defaults["train_months"]), key="research_train_months")
    with col6:
        test_months = st.number_input("WF 测试月数", min_value=1, max_value=60, value=int(wf_defaults["test_months"]), key="research_test_months")
    with col7:
        step_months = st.number_input("WF 步长月数", min_value=1, max_value=60, value=int(wf_defaults["step_months"]), key="research_step_months")

    default_grid = DEFAULT_STABILITY_PARAMETER_GRIDS.get(strategy_name, {})
    grid_text = st.text_area(
        "参数网格 JSON",
        value=json.dumps(default_grid, ensure_ascii=False, indent=2),
        height=160,
        key=f"research_grid_{strategy_name}",
    )
    try:
        parameter_grid = json.loads(grid_text) if grid_text.strip() else {}
    except json.JSONDecodeError as exc:
        st.error(f"参数网格 JSON 无法解析：{exc}")
        return

    cost_config = cost_config_panel(prefix="research_")
    col8, col9 = st.columns(2)
    with col8:
        tags = st.text_input("tags", value="workflow, research", key="research_tags")
    with col9:
        notes = st.text_input("notes", value="V3.3 synchronous research workflow", key="research_notes")

    st.info("本次研究将执行参数优化和稳定性分析，可能耗时较久。系统将在当前页面同步运行，不会在后台继续执行。")
    if st.button("运行研究工作流"):
        config = ResearchConfig(
            name=research_name,
            strategy_name=strategy_name,
            start_date=str(start_date),
            end_date=str(end_date),
            research_template=template_name,
            parameter_grid=parameter_grid,
            top_n_params=int(top_n),
            train_ratio=float(train_ratio),
            walk_forward_config={
                "train_months": int(train_months),
                "test_months": int(test_months),
                "step_months": int(step_months),
            },
            cost_config=config_to_dict(cost_config),
            tags=parse_tags(tags),
            notes=notes,
            save_intermediate_experiments=True,
        )
        with st.spinner("正在同步运行研究工作流..."):
            result = research_workflow_service().run(config)
        st.session_state["research_workflow_result"] = result.to_dict()

    result = st.session_state.get("research_workflow_result")
    if result:
        render_research_result(result)


def render_research_result(result: dict[str, object]) -> None:
    scorecard = dict(result.get("scorecard", {}) or {})
    cols = st.columns(5)
    cols[0].metric("状态", result.get("status", "-"))
    cols[1].metric("综合评分", num(scorecard.get("overall_score", 0)))
    cols[2].metric("评级", scorecard.get("final_rating", "-"))
    cols[3].metric("决策", result.get("decision", "-"))
    cols[4].metric("candidate_id", result.get("candidate_id") or "-")
    st.write(scorecard.get("decision_reason", ""))
    if scorecard:
        st.json(scorecard, expanded=False)
    steps = pd.DataFrame(result.get("steps", []) or [])
    if not steps.empty:
        columns = [column for column in ["step_name", "status", "message", "experiment_id"] if column in steps.columns]
        st.dataframe(steps.loc[:, columns], use_container_width=True)


def research_runs_tab(store: ResearchStore) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        strategy_filter = st.selectbox(
            "策略名称",
            ["all"] + list(STRATEGY_OPTIONS.values()),
            format_func=lambda value: "all" if value == "all" else display_strategy_name(value),
            key="research_runs_strategy",
        )
    with col2:
        status_filter = st.selectbox("状态", ["all", "draft", "running", "completed", "failed", "cancelled"], key="research_runs_status")
    with col3:
        decision_filter = st.selectbox("决策", ["all", "continue_research", "add_to_candidates", "watch", "reject", "failed"], key="research_runs_decision")
    col4, col5, col6 = st.columns(3)
    with col4:
        created_start = st.date_input("创建起始日期", value=None, key="research_runs_created_start")
    with col5:
        created_end = st.date_input("创建结束日期", value=None, key="research_runs_created_end")
    with col6:
        tag_filter = st.text_input("标签关键字", key="research_runs_tag")
    keyword = st.text_input("关键词（名称 / 备注 / 决策原因）", key="research_runs_keyword")
    runs = store.list_research_runs(
        strategy_name=strategy_filter,
        status=status_filter,
        decision=decision_filter,
        created_start=created_start,
        created_end=created_end,
        tag=tag_filter or None,
        keyword=keyword or None,
    )
    if runs.empty:
        st.info("当前暂无研究任务，或没有符合筛选条件的记录。")
        return
    display_columns = [
        "name",
        "strategy_name",
        "status",
        "research_template",
        "decision",
        "overall_score",
        "final_rating",
        "best_candidate_id",
        "created_at",
    ]
    st.dataframe(runs.loc[:, [column for column in display_columns if column in runs.columns]], use_container_width=True)
    selected = select_research_run_id(runs, "research_runs_select")
    if not selected:
        return
    detail = store.get_research_run_detail(selected)
    run = detail["run"]
    st.subheader("研究详情")
    cols = st.columns(4)
    cols[0].metric("状态", run.get("status", "-"))
    cols[1].metric("决策", run.get("decision", "-"))
    cols[2].metric("best_candidate_id", run.get("best_candidate_id") or "-")
    cols[3].metric("Git Commit", (run.get("git_commit") or "-")[:8])
    st.json(run.get("summary_json", {}), expanded=False)

    st.subheader("研究步骤")
    st.dataframe(detail["steps"], use_container_width=True)
    if not detail["steps"].empty and "experiment_id" in detail["steps"].columns:
        linked = detail["steps"]["experiment_id"].dropna().astype(str)
        st.caption("Linked experiment IDs: " + (", ".join(linked.tolist()) if not linked.empty else "-"))

    st.subheader("Artifacts / 报告路径")
    st.dataframe(detail["artifacts"], use_container_width=True)

    st.subheader("删除研究任务")
    if st.button("准备删除研究任务", key=f"prepare_delete_research_{selected}"):
        st.session_state["delete_research_run_id"] = selected
    if st.session_state.get("delete_research_run_id") == selected:
        run_name = str(run.get("name") or "")
        st.warning("删除研究任务会删除 steps/artifacts，但不会删除候选策略库中的记录。请输入完整研究任务名称确认删除。")
        confirmation = st.text_input("输入研究任务名称确认删除", key=f"delete_research_confirm_{selected}")
        can_delete = delete_confirmation_matches(run_name, confirmation)
        if st.button("确认删除研究任务", disabled=not can_delete, key=f"delete_research_button_{selected}"):
            deleted = store.delete_research_run(selected)
            st.session_state.pop("delete_research_run_id", None)
            if deleted:
                st.success("研究任务已删除。")
            else:
                st.info("研究任务不存在，可能已被删除。")
            st.rerun()


def research_candidates_tab(store: ResearchStore) -> None:
    st.info("候选策略仅表示值得继续研究，不代表可以买入或保证盈利。")
    col1, col2, col3 = st.columns(3)
    with col1:
        strategy_filter = st.selectbox(
            "策略名称",
            ["all"] + list(STRATEGY_OPTIONS.values()),
            format_func=lambda value: "all" if value == "all" else display_strategy_name(value),
            key="candidate_strategy",
        )
    with col2:
        status_filter = st.selectbox("候选状态", ["all", "active", "watching", "rejected", "archived"], key="candidate_status")
    with col3:
        overfit_filter = st.selectbox("过拟合风险", ["all", "LOW", "MEDIUM", "HIGH"], key="candidate_overfit")
    col4, col5 = st.columns(2)
    with col4:
        rating_filter = st.selectbox("评级", ["all", "A", "B", "C", "D"], key="candidate_rating")
    with col5:
        tag_filter = st.text_input("标签关键字", key="candidate_tag")
    candidates = store.list_strategy_candidates(
        strategy_name=strategy_filter,
        status=status_filter,
        rating=rating_filter,
        overfit_risk=overfit_filter,
        tag=tag_filter or None,
    )
    if candidates.empty:
        st.info("当前暂无候选策略，或没有符合筛选条件的候选记录。")
        return
    display_columns = [
        "candidate_name",
        "strategy_name",
        "status",
        "overall_score",
        "overfit_risk",
        "net_CAGR",
        "max_drawdown",
        "sharpe_ratio",
        "walk_forward_positive_ratio",
        "parameter_stability_score",
        "created_at",
    ]
    table = candidates.loc[:, [column for column in display_columns if column in candidates.columns]].copy()
    if "overall_score" in table.columns:
        table = table.rename(columns={"overall_score": "研究综合分"})
    st.dataframe(table, use_container_width=True)

    if st.button("导出当前筛选候选 CSV"):
        path = export_research_candidates_csv(store, None, candidates)
        st.success(f"已导出：{path}")
        st.caption(f"文件大小：{path.stat().st_size} bytes")
        st.download_button(f"下载 {path.name}", data=path.read_bytes(), file_name=path.name, key="download_candidate_filter_csv")

    selected = select_candidate_id(candidates, "candidate_select")
    if selected:
        candidate = store.get_strategy_candidate(selected)
        if not candidate:
            st.warning("未找到该候选策略，可能已被删除。")
            return
        st.subheader("候选详情")
        scorecard = dict(candidate.get("scorecard_json", {}) or {})
        metrics = dict(candidate.get("metrics_json", {}) or {})
        detail_cols = st.columns(3)
        detail_cols[0].metric("scorecard overall_score", num(scorecard.get("overall_score", 0)))
        detail_cols[1].metric("parameter_stability_score", num(metrics.get("parameter_stability_score", 0)))
        detail_cols[2].metric("regime_stability_score", num(metrics.get("regime_stability_score", 0)))
        with st.expander("scorecard_json", expanded=False):
            st.json(scorecard, expanded=False)
        with st.expander("metrics_json", expanded=False):
            st.json(metrics, expanded=False)

        latest = candidate_store().get_candidate_with_latest_revalidation(selected).get("latest_revalidation")
        st.subheader("最近复检结果")
        if latest:
            review_cols = st.columns(3)
            review_cols[0].metric("recommended_action", latest.get("recommended_action") or "-")
            review_cols[1].metric("最近复检时间", str(latest.get("created_at") or "-")[:19])
            review_cols[2].metric("latest net_CAGR", pct(dict(latest.get("latest_metrics_json", {}) or {}).get("net_CAGR")))
        else:
            st.info("该候选策略尚未复检。请在侧边栏进入候选复检页面进行复检。")
        events = candidate_store().list_lifecycle_events(selected)
        if not events.empty:
            st.caption("生命周期日志摘要")
            st.dataframe(events.head(5), use_container_width=True)

        status_options = ["active", "watching", "rejected", "archived"]
        current_status = candidate.get("status") if candidate.get("status") in status_options else "watching"
        new_status = st.selectbox("修改状态", status_options, index=status_options.index(current_status), key=f"candidate_status_edit_{selected}")
        new_notes = st.text_area("notes", value=candidate.get("notes") or "", key=f"candidate_notes_{selected}")
        new_tags = st.text_input("tags", value=", ".join(candidate.get("tags") or []), key=f"candidate_tags_{selected}")
        if st.button("保存候选策略修改", key=f"save_candidate_{selected}"):
            store.update_strategy_candidate(selected, status=new_status, notes=new_notes, tags=parse_tags(new_tags))
            st.success("候选策略已保存。")
            st.rerun()


def research_reports_tab(store: ResearchStore) -> None:
    runs = store.list_research_runs()
    selected = select_research_run_id(runs, "research_report_select")
    if not selected:
        st.info("请选择一个研究任务导出报告。")
        return
    detail = store.get_research_run_detail(selected)
    run = detail["run"]
    candidate_id = (run.get("summary_json", {}) or {}).get("candidate_id") or run.get("best_candidate_id") if run else None
    candidate_frame = _candidate_frame_for_research(store, candidate_id)
    exports = [
        ("导出 Markdown 研究报告", lambda: export_research_markdown_report(store, selected)),
        ("导出 JSON Summary", lambda: export_research_json_summary(store, selected)),
        ("导出 candidates CSV", lambda: export_research_candidates_csv(store, selected, candidate_frame)),
    ]
    for label, action in exports:
        if st.button(label):
            path = action()
            st.success(f"已导出：{path}")
            st.caption(f"文件大小：{path.stat().st_size} bytes")
            st.download_button(f"下载 {path.name}", data=path.read_bytes(), file_name=path.name)

    st.subheader("已有 artifacts")
    refreshed = store.get_research_run_detail(selected)
    artifacts = refreshed["artifacts"]
    if artifacts.empty:
        st.info("当前研究任务暂无导出记录。")
    else:
        columns = [column for column in ["artifact_type", "file_path", "created_at"] if column in artifacts.columns]
        st.dataframe(artifacts.loc[:, columns], use_container_width=True)


def _candidate_frame_for_research(store: ResearchStore, candidate_id: str | None) -> pd.DataFrame:
    if not candidate_id:
        return pd.DataFrame()
    candidate = store.get_strategy_candidate(candidate_id)
    return pd.DataFrame([candidate]) if candidate else pd.DataFrame()


def select_research_run_id(runs: pd.DataFrame, key: str) -> str | None:
    if runs.empty:
        return None
    labels = {
        f"{row['name']} | {row['strategy_name']} | {str(row['created_at'])[:10]} | {str(row['id'])[:8]}": row["id"]
        for _, row in runs.iterrows()
    }
    selected = st.selectbox("选择研究任务", list(labels), key=key)
    return labels.get(selected)


def select_candidate_id(candidates: pd.DataFrame, key: str) -> str | None:
    if candidates.empty:
        return None
    labels = {
        f"{row['candidate_name']} | {row['strategy_name']} | {row['status']} | {str(row['id'])[:8]}": row["id"]
        for _, row in candidates.iterrows()
    }
    selected = st.selectbox("选择候选策略", list(labels), key=key)
    return labels.get(selected)


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


def candidate_revalidation_page() -> None:
    st.header("候选复检")
    st.info("候选策略仅表示值得继续研究，不代表可以买入或保证盈利。")
    store = candidate_store()
    research = research_store()
    tabs = st.tabs(["复检仪表盘", "单个候选复检", "批量复检", "复检历史", "生命周期日志"])
    with tabs[0]:
        candidate_dashboard_tab(research, store)
    with tabs[1]:
        candidate_single_revalidation_tab(research, store)
    with tabs[2]:
        candidate_batch_revalidation_tab(research, store)
    with tabs[3]:
        candidate_history_tab(research, store)
    with tabs[4]:
        candidate_lifecycle_tab(research, store)


def candidate_dashboard_tab(research: ResearchStore, store: CandidateStore) -> None:
    candidates = research.list_strategy_candidates()
    due = store.list_candidates_due_for_review(max_age_days=30)
    history = store.list_revalidation_runs()
    status_counts = candidates["status"].value_counts().to_dict() if not candidates.empty else {}
    cols = st.columns(5)
    cols[0].metric("候选总数", int(len(candidates)))
    cols[1].metric("active", int(status_counts.get("active", 0)))
    cols[2].metric("watching", int(status_counts.get("watching", 0)))
    cols[3].metric("rejected", int(status_counts.get("rejected", 0)))
    cols[4].metric("archived", int(status_counts.get("archived", 0)))
    more = st.columns(4)
    more[0].metric("需要复检", int(len(due)))
    summary = revalidation_summary_metrics(history)
    more[1].metric("最近复检", str(summary["last_revalidation_at"])[:19])
    avg_stability = candidates["stability_score"].dropna().mean() if not candidates.empty and "stability_score" in candidates else 0
    more[2].metric("平均 stability_score", num(avg_stability))
    risk_counts = candidates["overfit_risk"].value_counts().to_dict() if not candidates.empty and "overfit_risk" in candidates else {}
    more[3].metric("overfit_risk 分布", ", ".join(f"{key}: {value}" for key, value in risk_counts.items()) or "-")
    if due.empty:
        st.info("当前没有需要复检的候选策略。")
        return
    columns = [
        "candidate_name",
        "strategy_name",
        "status",
        "last_revalidation_at",
        "days_since_revalidation",
        "latest_recommended_action",
        "latest_net_CAGR",
        "latest_max_drawdown",
        "latest_overfit_risk",
    ]
    st.dataframe(due.loc[:, [column for column in columns if column in due.columns]], use_container_width=True)


def candidate_single_revalidation_tab(research: ResearchStore, store: CandidateStore) -> None:
    candidates = research.list_strategy_candidates(status="all")
    if candidates.empty:
        st.info("当前暂无候选策略。请先通过研究工作流生成候选策略。")
        return
    selected = select_candidate_id(candidates, "single_candidate_revalidation_select")
    if not selected:
        return
    detail = store.get_candidate_with_latest_revalidation(selected)
    candidate = detail["candidate"]
    if not candidate:
        st.warning("未找到候选策略，可能已被删除。")
        return

    st.subheader("候选当前信息")
    info_cols = st.columns(4)
    info_cols[0].metric("strategy_name", candidate.get("strategy_name", "-"))
    info_cols[1].metric("status", candidate.get("status", "-"))
    info_cols[2].metric("source_research_run_id", str(candidate.get("source_research_run_id") or "-")[:8])
    info_cols[3].metric("source_experiment_id", str(candidate.get("source_experiment_id") or "-")[:8])
    with st.expander("params_json", expanded=False):
        st.json(candidate.get("params_json", {}), expanded=False)
    with st.expander("baseline metrics", expanded=False):
        st.json(candidate.get("metrics_json", {}), expanded=False)

    col1, col2, col3 = st.columns(3)
    with col1:
        mode = st.selectbox("复检模式", ["forward_only", "recent_window", "full_retest"], format_func=format_revalidation_mode, key="candidate_single_mode")
        if mode == "full_retest":
            st.caption("full_retest 默认使用来源研究区间；找不到来源区间时请手动填写复检日期。")
    with col2:
        recent_months = st.number_input("recent_months", min_value=1, max_value=60, value=6, key="candidate_single_recent_months")
    with col3:
        save_as_experiment = st.checkbox("保存为 experiment", value=True, key="candidate_single_save_exp")
    date_cols = st.columns(2)
    with date_cols[0]:
        review_start = st.date_input("复检起始日期（可选）", value=None, key="candidate_single_start")
    with date_cols[1]:
        review_end = st.date_input("复检结束日期（可选）", value=None, key="candidate_single_end")
    notes = st.text_input("notes", value="candidate revalidation", key="candidate_single_notes")
    tags = st.text_input("tags", value="candidate, revalidation", key="candidate_single_tags")
    cost_config = cost_config_panel(prefix="candidate_single_cost_")

    if st.button("运行复检", key="run_single_candidate_revalidation"):
        config = CandidateRevalidationConfig(
            candidate_id=selected,
            mode=mode,
            review_start_date=str(review_start) if review_start else None,
            review_end_date=str(review_end) if review_end else None,
            recent_months=int(recent_months),
            cost_config=config_to_dict(cost_config),
            walk_forward_config={"train_months": 24, "test_months": 6, "step_months": 6},
            save_as_experiment=save_as_experiment,
            notes=notes,
            tags=parse_tags(tags),
        )
        with st.spinner("正在同步运行候选复检..."):
            result = candidate_revalidator_service().run_revalidation(config)
        st.session_state["candidate_revalidation_result"] = result.to_dict()
        st.rerun()

    result = st.session_state.get("candidate_revalidation_result")
    if result and result.get("candidate_id") == selected:
        render_candidate_revalidation_result(result, candidate, store)


def render_candidate_revalidation_result(result: dict[str, object], candidate: dict[str, object], store: CandidateStore) -> None:
    st.subheader("baseline vs latest")
    baseline = dict(result.get("baseline_metrics", {}) or {})
    latest = dict(result.get("latest_metrics", {}) or {})
    cols = st.columns(4)
    cols[0].metric("baseline net_CAGR", pct(baseline.get("net_CAGR")))
    cols[1].metric("latest net_CAGR", pct(latest.get("net_CAGR")))
    cols[2].metric("latest max_drawdown", pct(latest.get("max_drawdown")))
    cols[3].metric("latest overfit_risk", latest.get("overfit_risk") or "-")
    st.subheader("差异比较")
    st.dataframe(pd.DataFrame([result.get("comparison", {})]), use_container_width=True)
    decision = dict(result.get("decision", {}) or {})
    action = str(result.get("recommended_action") or decision.get("recommended_action") or "continue_monitoring")
    st.subheader("生命周期建议")
    st.metric("recommended_action", action)
    st.write(result.get("action_reason") or decision.get("action_reason") or "-")
    flags = decision.get("flags", []) or []
    if flags:
        st.dataframe(pd.DataFrame({"risk_flags": flags}), use_container_width=True)
    target_status = action_to_candidate_status(action, str(candidate.get("status") or "watching"))
    if st.button("应用建议到候选状态", key=f"apply_candidate_action_{result.get('revalidation_id')}"):
        if target_status == candidate.get("status"):
            st.info("该建议不会改变当前候选状态。")
        else:
            ok = store.apply_candidate_status_change(str(candidate["id"]), target_status, str(result.get("action_reason") or "candidate revalidation action"))
            store.update_revalidation_run(
                str(result.get("revalidation_id")),
                applied_action=ok,
                old_candidate_status=candidate.get("status"),
                new_candidate_status=target_status,
            )
            st.success(f"状态已更新为 {target_status}" if ok else "候选策略不存在，无法更新状态。")
            st.rerun()
    if st.button("导出复检报告", key=f"export_candidate_report_{result.get('revalidation_id')}"):
        path = export_revalidation_markdown_report(store, str(result.get("revalidation_id")))
        st.success(f"已导出：{path}")
        st.caption(f"文件大小：{path.stat().st_size} bytes")
        st.download_button(f"下载 {path.name}", data=path.read_bytes(), file_name=path.name)


def candidate_batch_revalidation_tab(research: ResearchStore, store: CandidateStore) -> None:
    st.info("批量复检会同步运行，候选较多时可能耗时较久。默认不自动应用建议。")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_filter = st.selectbox("候选状态", ["active", "watching", "all"], key="batch_candidate_status")
    with col2:
        strategy_filter = st.selectbox("策略名称", ["all"] + list(STRATEGY_OPTIONS.values()), format_func=lambda value: "all" if value == "all" else display_strategy_name(value), key="batch_candidate_strategy")
    with col3:
        max_age_days = st.number_input("max_age_days", min_value=0, max_value=365, value=30, key="batch_max_age_days")
    with col4:
        max_count = st.number_input("最多复检数量", min_value=1, max_value=100, value=10, key="batch_max_count")
    tag = st.text_input("标签关键字", key="batch_candidate_tag")
    mode = st.selectbox("复检模式", ["forward_only", "recent_window"], format_func=format_revalidation_mode, key="batch_revalidation_mode")
    due = store.list_candidates_due_for_review(max_age_days=int(max_age_days))
    if status_filter != "all" and not due.empty:
        due = due[due["status"].eq(status_filter)]
    if strategy_filter != "all" and not due.empty:
        due = due[due["strategy_name"].eq(strategy_filter)]
    if tag and not due.empty:
        due = due[due["tags"].map(lambda tags: any(tag.lower() in str(item).lower() for item in tags))]
    due = due.head(int(max_count)).reset_index(drop=True)
    if due.empty:
        st.info("当前没有符合条件的待复检候选策略。")
        return
    st.dataframe(due.loc[:, [column for column in ["candidate_name", "strategy_name", "status", "days_since_revalidation"] if column in due.columns]], use_container_width=True)
    if st.button("运行批量复检", key="run_batch_candidate_revalidation"):
        configs = [CandidateRevalidationConfig(candidate_id=row["id"], mode=mode, recent_months=6, tags=parse_tags(tag)) for _, row in due.iterrows()]
        with st.spinner("正在同步批量复检..."):
            rows = run_batch_revalidation(candidate_revalidator_service(), configs)
        display = []
        for row in rows:
            latest = dict(row.get("latest_metrics", {}) or {})
            display.append(
                {
                    "candidate_id": row.get("candidate_id"),
                    "status": row.get("status"),
                    "recommended_action": row.get("recommended_action"),
                    "latest_net_CAGR": latest.get("net_CAGR"),
                    "latest_max_drawdown": latest.get("max_drawdown"),
                    "error_message": row.get("error_message"),
                }
            )
        st.session_state["candidate_batch_results"] = display
    if st.session_state.get("candidate_batch_results"):
        st.dataframe(pd.DataFrame(st.session_state["candidate_batch_results"]), use_container_width=True)


def candidate_history_tab(research: ResearchStore, store: CandidateStore) -> None:
    candidates = research.list_strategy_candidates(status="all")
    candidate_id = None
    if not candidates.empty:
        labels = {"all": None}
        labels.update({candidate_option_label(row): row["id"] for _, row in candidates.iterrows()})
        selected_candidate = st.selectbox("candidate", list(labels), key="history_candidate")
        candidate_id = labels[selected_candidate]
    col1, col2, col3 = st.columns(3)
    with col1:
        strategy_name = st.selectbox("策略名称", ["all"] + list(STRATEGY_OPTIONS.values()), format_func=lambda value: "all" if value == "all" else display_strategy_name(value), key="history_strategy")
    with col2:
        mode = st.selectbox("模式", ["all", "forward_only", "recent_window", "full_retest"], key="history_mode")
    with col3:
        action = st.selectbox("recommended_action", ["all", "keep_active", "downgrade_to_watching", "reject", "archive", "needs_more_data", "continue_monitoring"], key="history_action")
    status = st.selectbox("状态", ["all", "completed", "failed", "draft"], key="history_status")
    date_cols = st.columns(2)
    with date_cols[0]:
        created_start = st.date_input("created_at 起始日期", value=None, key="history_created_start")
    with date_cols[1]:
        created_end = st.date_input("created_at 结束日期", value=None, key="history_created_end")
    history = store.list_revalidation_runs(
        candidate_id=candidate_id,
        strategy_name=strategy_name,
        mode=mode,
        recommended_action=action,
        status=status,
        created_start=str(created_start) if created_start else None,
        created_end=str(created_end) if created_end else None,
    )
    if history.empty:
        st.info("当前暂无复检历史。")
        return
    columns = ["candidate_name", "strategy_name", "mode", "status", "review_start_date", "review_end_date", "recommended_action", "latest_net_CAGR", "latest_max_drawdown", "latest_overfit_risk", "created_at"]
    st.dataframe(history.loc[:, [column for column in columns if column in history.columns]], use_container_width=True)
    if st.button("导出复检历史 CSV"):
        path = export_revalidation_history_csv(history, Path("reports/candidates/revalidation_history.csv"))
        st.success(f"已导出：{path}")
        st.download_button(f"下载 {path.name}", data=path.read_bytes(), file_name=path.name)
    labels = {f"{row['candidate_name']} | {row['recommended_action']} | {str(row['id'])[:8]}": row["id"] for _, row in history.iterrows()}
    selected = st.selectbox("选择复检详情", list(labels), key="history_detail_select")
    detail = store.get_revalidation_run(labels[selected])
    if detail:
        with st.expander("baseline_metrics_json", expanded=False):
            st.json(detail.get("baseline_metrics_json", {}), expanded=False)
        with st.expander("latest_metrics_json", expanded=False):
            st.json(detail.get("latest_metrics_json", {}), expanded=False)
        with st.expander("comparison_json", expanded=False):
            st.json(detail.get("comparison_json", {}), expanded=False)
        with st.expander("decision_json", expanded=False):
            st.json(detail.get("decision_json", {}), expanded=False)


def candidate_lifecycle_tab(research: ResearchStore, store: CandidateStore) -> None:
    candidates = research.list_strategy_candidates()
    candidate_id = select_candidate_id(candidates, "lifecycle_candidate_select") if not candidates.empty else None
    event_type = st.selectbox("event_type", ["all", "created", "revalidated", "status_changed", "note_updated", "archived", "rejected"], key="lifecycle_event_type")
    date_cols = st.columns(2)
    with date_cols[0]:
        created_start = st.date_input("created_at 起始日期", value=None, key="lifecycle_created_start")
    with date_cols[1]:
        created_end = st.date_input("created_at 结束日期", value=None, key="lifecycle_created_end")
    events = store.list_lifecycle_events(
        candidate_id=candidate_id,
        event_type=event_type,
        created_start=str(created_start) if created_start else None,
        created_end=str(created_end) if created_end else None,
    )
    if events.empty:
        st.info("当前暂无生命周期日志。")
    else:
        columns = ["candidate_name", "event_type", "old_status", "new_status", "message", "created_at"]
        st.dataframe(events.loc[:, [column for column in columns if column in events.columns]], use_container_width=True)
        if st.button("导出生命周期日志 CSV"):
            path = export_lifecycle_events_csv(events, Path("reports/candidates/lifecycle_events.csv"))
            st.success(f"已导出：{path}")
            st.download_button(f"下载 {path.name}", data=path.read_bytes(), file_name=path.name)
    if candidate_id:
        note = st.text_input("添加手动 note event", key="candidate_lifecycle_note")
        if st.button("保存 note event", disabled=not bool(note.strip())):
            candidate = research.get_strategy_candidate(candidate_id)
            status = candidate.get("status") if candidate else None
            store.create_lifecycle_event(candidate_id, "note_updated", status, status, note)
            st.success("生命周期 note 已保存。")
            st.rerun()


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
    "数据质量": data_quality_page,
    "数据中心": data_center,
    "策略实验室": strategy_lab,
    "回测中心": backtest_center,
    "参数优化器": parameter_optimizer_page,
    "策略排行榜": strategy_leaderboard_page,
    "组合回测": portfolio_backtest_page,
    "策略稳定性": strategy_stability_page,
    "研究工作流": research_workflow_page,
    "候选复检": candidate_revalidation_page,
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
