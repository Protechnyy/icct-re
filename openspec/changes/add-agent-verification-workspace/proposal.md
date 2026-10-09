# Proposal

## Why

用户需要在任务详情主区域看到校验智能体如何规划任务、调用工具、查阅原文和形成结论，以便展示项目中的智能体行为。现有页面显示当前任务及最近事件的字符串摘要，完整任务清单、调用等待状态和可阅读的原文依据需要通过统一的过程界面呈现。

## What Changes

- 在任务详情标题下方增加“智能体校验过程”，采用左侧任务清单、右侧执行时间线，包含规划任务、执行步骤、查看依据、形成结论四类信息。
- 在规划、模型调用、工具调用和核查阶段开始及结束时记录事件，显示真实的执行状态、参数、耗时和结论。
- 同时保存简短摘要与结构化详情，支持阅读原文、小节、页码以及关系和实体的具体改动。
- 执行期间把完整事件保存到 Redis，增加按 `seq` 分页增量读取及按事件读取详情的接口；沿用前端 2.5 秒轮询，支持刷新和切换任务后的恢复。
- 自动展开当前任务，允许用户选择历史任务；核查结束后保留完整过程，明确显示任务完成度和最终采用的改动。
- 对已保存的旧结果展示现有记录，明确标记详情缺失；没有可核查内容时展示跳过状态。

## Capabilities

### New Capabilities

- `agent-execution-events`：执行事件的生命周期、结构化详情、运行期间保存、增量查询和结束后的读取。
- `agent-verification-workspace`：任务详情主区域中的任务清单、执行时间线、证据阅读和结论展示，以及运行、完成、异常和历史结果的界面行为。

### Modified Capabilities

## Impact

- 后端：`backend/app/agent/trace.py`、`runtime.py`、`budget.py`、`workspace.py`、`pipeline.py`、`task_store.py` 和 `api.py`，增加执行事件记录、保存和查询。
- 前端：`frontend/src/App.jsx`、`lib/api.js`、`components/AgentVerification.jsx`、`components/ResultViewer.jsx` 和 `index.css`，增加主区域过程组件、增量数据读取和交互状态。
- API：新增 `GET /api/agent/<task_id>/events` 和 `GET /api/agent/<task_id>/events/<seq>`；现有状态接口及结果接口增加可选字段，原有调用方式继续可用。
- 使用现有 Redis、Flask、React 和 Arco Design。验证包含真实 Redis、真实模型调用和浏览器 DOM 交互检查。
- 本变更以 `add-document-agent-verification` 已有实现为基础。现有 `agent-trace-and-progress` 对最近事件、结果轨迹和逐关系详情的要求继续适用；新能力负责完整执行事件查询及主区域过程界面。当前 `openspec/specs/` 没有已发布规格。
