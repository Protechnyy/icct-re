# 实施验证记录

日期：2026-10-09。当前已完成 40/43 个跟踪任务；未归档。

## 已验证

- 在现有 `backend/.venv` 安装并锁定 `langgraph==1.2.14`、`langchain-openai==1.7.0`；完整 requirements 安装成功，`pip check` 无冲突。无需启用第 9 组 OCR 拆分备选任务，9.1–9.3 未执行、未勾选。
- 后端测试共 80 项通过，包括配置默认值及密钥隐藏、索引和来源定位、受约束的关系修改、模型规划降级、LangGraph 多任务执行、记忆、预算和失败回退、流水线接入、状态/结果接口、CSV 和离线评估。
- 抽取与核查的默认模型已改为 `qwen3.8-27b`，地址及本地 `.env` 的对应条目按用户要求同步，密钥未修改。修复 `qwen_api` SDK 忽略 `SKILL4RE_BASE_URL` 的问题，新增两项地址覆盖回归测试。使用本地 `qwen_api` 配置实测 SDK 已请求指定北京部署，`qwen3.8-27b` 返回预期的空关系列表，核查默认开关为关闭。
- 从临时复制的 `.env.example` 启动 worker，配置摘要显示 `agent_enabled=False`；在读取任务队列前停止，未消费生产队列。
- `npm run build` 成功。无头 Chrome 使用真实组件验证处理中进度和事件、摘要数字、删除列表、被更正关系的详情及原文位置、关闭状态，以及本地保存的旧结果；没有页面异常。
- `openspec validate add-document-agent-verification --strict` 与 `git diff --check` 通过。新增评估脚本的进度写入经 8 线程、100 次同路径并发写验证，无碰撞且无临时文件残留。
- 百炼假工具连通性已实测，模型名、默认思考模式和 usage 结论见 `design.md` 已验证的连接参数。该实验仅使用诊断文本，未发送保存结果的原文。
- 用户明确授权后，两份真实保存结果已完成百炼重放，输出到 `data/agent_replays/20261009_authorized/`，SHA256 与修改时间均未变。两份状态均为部分完成，共 32 次更正；模型规划解析失败触发规则回退，12 个任务达到步数上限。详见 `replay-report.md`。

## 重放入口

`scripts/replay-agent-verification.py` 只运行核查，不启动 OCR 或 Redis。独立模型配置来自 `AGENT_BASE_URL`、`AGENT_API_KEY`、`AGENT_MODEL`。默认开启核查只限于本次重放，不写 `.env`。

```bash
backend/.venv/bin/python scripts/replay-agent-verification.py \
  <已保存结果/result.json> --output-dir <独立输出目录>

# 多篇结果可以同时列出；evaluation.json 包含逐文档指标和汇总。
# 标注格式：单篇为 head/relation/tail 对象数组，多篇为 task_id 到该数组的映射。
backend/.venv/bin/python scripts/replay-agent-verification.py \
  <结果1/result.json> <结果2/result.json> \
  --output-dir <独立输出目录> --annotations <标注.json>

# 输出目录中每篇有 result.json、metrics.json、changes.csv。
# judgment 列支持 正确/错误、true/false、1/0；未填行不计入正确率分母。
backend/.venv/bin/python scripts/replay-agent-verification.py \
  --judgments-csv <填好判定的changes.csv>
```

## 本轮评估与条件任务

- **8.5 已完成**：按用户要求生成五领域虚构材料和预标注，真实调用用户已启动的 PaddleOCR-VL 服务、指定百炼模型，保存抽取结果并重放。五份输入 SHA256 和修改时间未变；六处改动由助手逐条对照原文审阅，整体 5/6 正确，实体合并 0/1。详见 `five-domain-evaluation.md`，不声称独立人工盲审。
- **8.6 已完成**：README 已列出全部 agent 配置、开启/回滚/重放/判定步骤，建议百炼 batch 并发 4。保持默认关闭：五篇核查均部分完成，删除门槛无样本，发现自指关系合并错误。
- **9.1–9.3 不适用**：只在 1.1 出现依赖冲突时启用，当前依赖无冲突。未执行 OCR HTTP 独立服务拆分，也未将三个条件任务勾选为已验证。OpenSpec 统计仍为 40/43，适用任务全部完成。
