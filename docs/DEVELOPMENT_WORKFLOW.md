# A-Stock Quant Lab Development Workflow

本项目后续所有功能开发都必须走 Git 分支 + PR 工作流。用户只提出业务目标和最终验收意见，不介入日常技术细节判断。

## 角色分工

- 用户：提出业务目标、约束、最终验收意见。
- GPT：拆解需求、制定验收标准、输出 Review 清单、做产品验收。
- Codex：实现功能、自测、修复问题、提交分支和 PR。

## 标准流程

1. 用户提出业务目标。
2. GPT 将目标拆解为功能范围、验收标准和 Review 清单。
3. Codex 从 `main` 创建功能分支。
4. Codex 实现功能，并主动运行本地验收。
5. Codex 提交 commit，创建 PR。
6. PR 交给 GPT 做产品验收和代码 Review。
7. GPT 验收通过后，用户再合并。

## 分支规则

- `main` 始终代表已验收、可运行版本。
- 每个新功能、修复或流程改动都必须从 `main` 创建独立分支。
- 分支命名建议：
  - `feature/<feature-name>`
  - `fix/<bug-name>`
  - `chore/<workflow-or-maintenance-name>`

## PR 必须包含

1. 功能说明
2. 修改文件
3. 自测结果
4. 截图/运行说明
5. 已知问题

## 合并前检查

每个 PR 合并前至少完成：

```bash
pip install -r requirements.txt
pytest
ruff check .
python -m compileall .
python scripts/validate_project.py
```

涉及 Streamlit 页面时，还必须确认：

```bash
streamlit run app.py
```

## 量化逻辑 Review 清单

- 无未来函数。
- rolling 因子只使用当前日及以前数据。
- 回测买入逻辑为信号日后一交易日开盘价。
- 收益、回撤、胜率、夏普、期望收益计算合理。
- 无重复信号错误累计。
- 不包含自动交易功能。

## 用户验收边界

用户只需要根据业务目标判断最终结果是否可接受。技术实现、测试失败修复、PR 描述、截图和运行说明由 Codex 完成，GPT 负责 Review 和验收把关。
