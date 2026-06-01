# A-Stock Quant Lab

本项目是本地运行的 Python A股量化研究平台，用于个人投资研究，不接入自动交易，不提供实盘下单，不承诺收益。

V3 将系统从“单策略信号回测”升级为“真实交易模拟 + 仓位管理 + 策略组合研究平台”，重点覆盖交易成本、滑点、A股下单约束、组合回测、gross/net 对比、策略组合和增强市场环境分析。

## 安装

建议使用 Python 3.11 或更高版本。

```bash
cd "C:\Users\18961\Documents\Stock Trading"
pip install -r requirements.txt
```

## 运行

```bash
streamlit run app.py
```

本地验收命令：

```bash
python scripts/validate_project.py
```

## 数据来源

股票列表、股票日线和沪深300指数数据通过 AKShare 获取，写入本地 SQLite：

- 股票日线表：`daily_price`
- 股票列表表：`stock_list`
- 指数行情表：`index_prices`

如果本地没有沪深300指数数据，市场环境模块允许使用股票池已有数据构造等权代理指数。该 fallback 仅用于研究辅助，不能替代真实指数行情。

## 因子说明

`factor/factor_engine.py` 计算：

- MA5、MA10、MA20、MA60
- return_5d、return_10d、return_20d
- volume_ma20、volume_ratio
- volatility_20d
- breakout_20d、breakout_60d
- ma_bullish
- distance_to_ma20
- max_drawdown_20d

所有 rolling 和 pct_change 均按股票分组、按日期升序执行，只使用当前日及以前数据。

## 策略说明

内置策略：

- 趋势强势股策略：`close > MA20 > MA60`、20日涨幅强、成交额活跃、排除 ST。
- MA20回踩策略：MA20 在 MA60 上方，价格回踩 MA20 附近，短期涨跌幅温和，量能不过热。
- 放量突破策略：突破 60 日新高，量比放大，当日涨幅控制在 3% 到 8%。
- 多因子评分策略：综合趋势、动量、成交额、突破、MA20距离和回撤风险评分。
- 策略组合：合并多个策略信号，按权重计算综合分并支持每日 Top N。

## 回测逻辑

V2 单策略回测保留：

- 信号日后一交易日开盘买入。
- 支持固定持仓 5 / 10 / 20 / 60 个交易日。
- 支持固定止盈、固定止损、跌破 MA20 卖出。
- 输出胜率、平均收益、CAGR、夏普、最大回撤、盈亏比、连续亏损和期望收益。

V3 组合回测新增：

- 每日按交易日推进。
- 收盘后生成信号，下一交易日开盘尝试买入。
- 买入检查现金、100股整数倍、已有持仓、涨停近似限制。
- 卖出检查止盈、止损、跌破 MA20、固定持仓周期、跌停近似限制。
- V3 暂不支持同股加仓；`allow_add_position=True` 会主动报错，V4 计划引入 lot-level 持仓核算后再支持。
- 每日更新 cash、positions、market_value、total_equity 和 drawdown。
- 输出组合资金曲线、回撤曲线、订单、交易、持仓快照和组合指标。

## 交易成本模型

默认参数仅用于研究，真实费率需要按个人券商账户调整：

- 佣金比例：`commission_rate = 0.0003`
- 单笔最低佣金：`min_commission = 5.0`
- 卖出印花税：`stamp_tax_rate_sell = 0.0005`
- 规费近似：`regulatory_fee_rate = 0.00002`
- 过户费近似：`transfer_fee_rate = 0.00001`

成本明细会拆分为 commission、stamp_tax、regulatory_fee、transfer_fee、total_cost。

## 滑点模型

默认滑点为 `slippage_bps = 5`：

- 买入成交价 = 原始价格 * `(1 + slippage_bps / 10000)`
- 卖出成交价 = 原始价格 * `(1 - slippage_bps / 10000)`

滑点成本会单独统计，并参与 net result。

## 仓位模型

支持三种仓位方法：

- `equal_cash`：可用资金按剩余可开仓数量等分。
- `fixed_cash`：每笔固定金额。
- `risk_percent`：根据买入价与止损价的风险距离计算仓位。

买入股数始终向下取整到 100 股整数倍，仓位计算会考虑交易费用。资金不足买入 100 股时，订单会跳过并记录原因。

## Gross 与 Net

- gross：不含佣金、印花税、规费、过户费和滑点。
- net：包含佣金、印花税、规费、过户费和滑点。
- `cost_impact = gross_total_return - net_total_return`。

UI 会显示交易成本对收益的影响。

## 市场环境

增强市场环境使用：

- MA20、MA60、MA120
- return_20d
- volatility_20d
- trend_slope_60d

分类：

- `BULL`：close > MA60，MA60 > MA120，return_20d > 0。
- `BEAR`：close < MA60，MA60 < MA120，return_20d < 0。
- `HIGH_VOLATILITY`：volatility_20d 高于最近一年 75 分位数。
- `SIDEWAYS`：其他情况。

涨跌停模型为 V3 近似版本，默认按 10% 阈值估算。后续 V4 可按主板、创业板、科创板、ST 等规则细分。

## V3.1 实验管理

实验记录用于保存一次研究动作的完整快照，包括参数配置、核心指标、交易明细、订单明细、资金曲线、备注、标签、评级、Git commit 和本地数据快照。

- 组合回测：在“组合回测”页面运行后，可手动保存实验，也可开启“运行后自动保存实验”。
- 单策略回测：在“回测中心”页面运行后，可保存为 `single_strategy_backtest` 实验。
- 参数优化：在“参数优化器”页面运行后，可将 Top N 参数组合保存为 `parameter_optimization` 实验。
- 策略排行榜：在“策略排行榜”页面可保存当日排行榜快照，用于后续比较策略表现变化。
- 实验中心：包含实验列表、实验详情、实验对比、报告导出四个标签页。
- 实验对比：支持 2 到 5 个实验横向比较，资金曲线会从 1.0 归一化，最大回撤比较中 -10% 优于 -30%。
- 报告导出：支持 Markdown 报告、Trades CSV、Orders CSV、Equity CSV、Config JSON，文件默认写入 `reports/experiments/`。
- 标签和评级：实验可标记为 A/B/C/D/unrated，分别代表“值得继续研究 / 可观察 / 普通 / 淘汰 / 未评级”。

保存实验参数的目的，是让每次研究结论都能回溯到当时的策略参数、成本参数、仓位参数、退出规则、数据范围和代码版本。回测结果只说明历史样本中的表现，不代表未来收益。

## 测试

```bash
pip install -r requirements.txt
python -m pytest -q
python -m ruff check .
python -m compileall .
python scripts/validate_project.py
```

## 风险提示

- 本工具仅用于个人研究。
- 本工具不构成任何投资建议。
- 回测结果不代表未来收益。
- 默认交易成本参数仅为研究假设。
- 未接入 Level-2 成交数据，未考虑完整盘口冲击成本。
- 未覆盖停牌、复权差异、流动性缺失等所有极端情况。
- V3 暂不支持同股加仓，V4 再支持 lot-level 持仓核算。
- 实验记录基于当前本地数据，如果后续数据源修正历史数据，旧实验结果不会自动重算。
- 实验保存的是当次结果快照，不是投资建议；报告导出仅用于个人研究。
- 本项目不是自动交易系统，不支持实盘下单。
