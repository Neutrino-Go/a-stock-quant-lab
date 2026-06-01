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

## V3.1.1 实验中心优化

V3.1.1 聚焦实验中心长期研究可用性，不新增策略、不接入自动交易。

### 实验筛选方法

实验中心的实验列表支持按实验类型、策略名称、创建日期区间、标签关键字、评级和关键词筛选。关键词会匹配实验名称、描述、备注和结论。

### 实验排序说明

实验列表支持按 `created_at`、`net_CAGR`、`max_drawdown`、`sharpe_ratio`、`final_equity`、`cost_impact`、`win_rate` 排序。默认“降序 / 最佳优先”会按研究意义排序。

### max_drawdown 负数排序

`max_drawdown` 通常为负数，数值越接近 0 风险越低。例如 `-10%` 优于 `-30%`，因此最佳优先排序时 `-10%` 会排在 `-30%` 前面，空值排最后。

### CSV 股票代码前导 0 保护

导出的 Trades / Orders CSV 会将 `code` 规范为 6 位字符串，例如 `1`、`"1"`、`"000001"` 都导出为 `000001`。由于 Excel 直接打开 CSV 可能吞掉前导 0，系统额外提供 `code_text` 列，例如 `="000001"`，便于 Excel 查看。

CSV 文件使用 `utf-8-sig` 编码，方便 Excel 打开中文内容。

### 报告导出说明

报告导出统一写入 `reports/experiments/`，支持 Markdown、Trades CSV、Orders CSV、Equity CSV 和 Config JSON。导出后页面显示文件路径、文件大小和生成时间，并记录到实验 artifacts 表。

### 删除实验说明

删除实验需要二次确认：先点击“准备删除”，再输入完整实验名称后才能删除。删除会级联删除该实验的资金曲线、交易明细、订单明细和导出记录。

### validate_project.py 使用说明

本地验收命令：

```bash
python scripts/validate_project.py
```

该脚本会检查关键目录、关键文件、pytest、compileall 和 app import。compileall 会排除 `.git`、`.venv`、`review_packages`、`reports/experiments`、`storage`、`.streamlit`、缓存目录等无关路径，避免生成物污染校验。

## V3.2 策略稳定性与过拟合检测

V3.2 新增“策略稳定性”页面和 `stability/` 模块，用于判断策略是否稳定、是否可能过拟合、是否值得继续研究。该模块只做历史研究，不接入自动交易，不构成投资建议。

### 样本内 / 样本外

样本内（In-Sample）用于参数搜索和选择最佳参数；样本外（Out-of-Sample）不参与调参，只用于验证策略泛化能力。系统按时间顺序切分，不随机打乱时间序列，且测试区间必须晚于训练区间。样本外表现比样本内表现更重要。

### Walk-forward 验证

Walk-forward 会按时间滚动执行：训练窗口内搜索参数，只用训练窗口选择最佳参数，再到随后的测试窗口验证样本外表现。系统记录每个 fold 的训练区间、测试区间、最佳参数、训练指标、测试指标和 degradation，测试区间结果不会用于选择参数。

### 参数敏感性

参数敏感性分析用于判断最佳参数是否是“孤岛”。如果最佳参数显著高于中位数，但相邻参数表现很差，则过拟合风险较高；如果多个相邻参数表现接近，则参数更稳定。`sensitivity_score` 范围为 0 到 100，分数越高表示参数越稳定。

### 过拟合风险

过拟合检测会综合样本内/样本外差异、Sharpe 退化、样本外收益、Walk-forward 正收益比例、参数敏感性、交易次数、参数数量、样本外回撤扩大和交易成本影响，输出 LOW / MEDIUM / HIGH 风险评级、0 到 100 的 `overfit_score`、风险 flags 和研究建议。

### 稳定性评分

`stability_score` 综合 generalization、walk-forward、parameter stability、market regime stability 和 trade sample 五类分数，最终评级为 A/B/C/D。A 代表值得继续研究，D 代表稳定性不足或过拟合风险较高。

### 市场环境分组

稳定性分析会保留市场环境分组结果：BULL、BEAR、SIDEWAYS、HIGH_VOLATILITY。如果指数数据不足或样本交易过少，页面会提示样本不足，结果仅供参考。

### 实验中心集成

稳定性分析可以保存为 `strategy_stability_analysis` 实验。实验中心可查看稳定性评分、过拟合风险、样本内/样本外对比、Walk-forward fold、参数敏感性、市场环境分组和系统建议，并可导出 Markdown 报告。

### 风险提示

- 稳定性分析基于历史数据，不代表未来收益。
- 样本外表现好也不保证未来有效。
- Walk-forward 结果依赖数据质量和参数网格设计。
- 参数敏感性分析不能完全消除过拟合。
- 市场环境分组在样本不足时仅供参考。
- 本工具不构成投资建议。
- 本工具不支持自动交易。
