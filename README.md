# A股量化研究平台 MVP

本项目是一个本地运行的 Python A股量化研究平台，用于个人投资研究，不接入自动交易。第一阶段覆盖“数据获取 -> 因子计算 -> 策略选股 -> 回测 -> Streamlit 界面展示”的完整闭环。

## 安装方法

建议使用 Python 3.11 或更高版本。

```bash
cd "D:\Project\Stock Trading"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 运行方法

```bash
streamlit run app.py
```

首次使用建议在“数据中心”先更新股票列表，再更新单只股票或批量更新日线数据。

## 数据来源

数据通过 AKShare 获取，当前使用：

- `stock_info_a_code_name`：A股股票代码与名称
- `stock_zh_a_hist`：单只股票历史日线行情

日线数据写入本地 SQLite：`storage/stock_quant.db`。日线表以 `code + date` 作为主键，重复更新会执行 upsert，避免重复下载后产生重复记录。

## 因子说明

`factor/factor_engine.py` 实现以下因子：

- MA5、MA10、MA20、MA60
- return_5d、return_10d、return_20d
- volume_ma20、volume_ratio
- volatility_20d
- breakout_20d、breakout_60d
- ma_bullish
- distance_to_ma20
- max_drawdown_20d

所有 rolling 与 pct_change 计算都按股票分组、按日期升序执行，只使用当前日及以前的数据。

## 策略说明

当前内置 4 个策略：

- `trend_strength_strategy`：趋势多头、20日动量强、成交额活跃、排除 ST。
- `pullback_ma20_strategy`：MA20 在 MA60 上方，价格回踩 MA20 附近，短期涨跌幅温和，量能不过热。
- `volume_breakout_strategy`：60日新高突破，量比放大，当日涨幅控制在 3% 到 8%。
- `multi_factor_score_strategy`：综合趋势强度、20日动量、成交额、60日突破、MA20距离、20日回撤风险打分。

每个策略统一输出：

- code
- name
- signal_date
- strategy_name
- score
- signal_reason
- suggested_buy_price
- suggested_stop_loss
- suggested_take_profit
- risk_level

## 回测逻辑

回测引擎位于 `backtest/backtest_engine.py`。

买入规则：

- 信号日后一交易日开盘价买入。

卖出规则：

- 固定持仓 5 / 10 / 20 / 60 个交易日。
- 支持固定止盈，例如 10%、15%、20%。
- 支持固定止损，例如 5%、8%、10%。
- 可选择跌破 MA20 卖出。

如果同一交易日同时触发止损和止盈，回测按更保守的止损优先处理。

V2 回测指标包括：

- total_trades
- win_rate
- avg_return
- median_return
- CAGR
- sharpe_ratio
- max_drawdown
- profit_loss_ratio
- max_consecutive_losses
- expectancy
- equity_curve
- drawdown_curve

`expectancy = win_rate * avg_win - lose_rate * avg_loss`。权益曲线从初始资金 1.0 开始，只按清洗后的交易收益序列累计一次，最大回撤由该单一权益曲线计算。

## V2 研究模块

### 参数优化器

`optimizer/parameter_optimizer.py` 提供网格搜索：

- 趋势强势股策略：`min_amount`、`min_return_20d`
- MA20回踩策略：`max_distance`
- 放量突破策略：`min_volume_ratio`

输出排名前 20 的参数组合，包含胜率、年化收益、最大回撤和夏普。

### 策略排行榜

`ranking/leaderboard.py` 支持最近 1个月、3个月、6个月、1年维度的策略统计，并按规则输出 A/B/C/D 评级。

### 市场环境分类

`market/regime_engine.py` 使用 MA60 将市场划分为：

- Bull：close > MA60
- Bear：close < MA60
- Sideways：其他

回测中心会展示按市场环境分组后的交易表现。

## 测试

```bash
pytest
python -c "from factor.factor_engine import compute_factors; from backtest.backtest_engine import BacktestEngine; from ranking.ranking_engine import STRATEGIES; print('imports ok')"
```

## 风险提示

- 本工具仅用于个人研究。
- 本工具不构成任何投资建议。
- 回测结果不代表未来收益。
- 策略信号需要结合交易制度、流动性、停复牌、涨跌停、财报事件与个人风险承受能力进一步判断。
