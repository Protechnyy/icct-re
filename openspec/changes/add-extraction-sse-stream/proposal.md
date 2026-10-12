# Proposal

## Why

抽取阶段目前对用户是完全的黑盒：LLM 调用写死 `stream=False`（skill4re/backends.py），一个批次要等模型完整生成才返回，前端只能靠进度轮询干等，多页文档抽取动辄数分钟没有任何过程反馈。需要把模型的思考（think）与回答过程实时流式展示出来，让用户看到"正在发生什么"，同时为后续排查抽取质量提供直接的过程观测手段。

## What Changes

- skill4re 的 OpenAI 兼容调用支持流式模式（`stream=True`），逐 delta 上报 `think`（reasoning_content）与 `answer`（content）两类内容，通过 `on_delta` 回调注入，不改变抽取提示词与解析逻辑。
- 开启 `enable_thinking=true` 使模型输出思考过程，通过 `.env` 开关 `STREAM_THINKING` 控制（默认开），可随时退回纯回答流。
- Worker 侧每个并发 LLM 调用作为一个命名流（批次号 + 技能名），delta 事件实时写入 Redis Stream（`task:{id}:llm`），带 `seq` 支持增量拉取与断线重放。
- 新增 SSE 接口 `GET /api/tasks/{id}/stream`，基于 Redis Stream `XREAD BLOCK` 实现，含心跳保活与 `Last-Event-ID`/`after_seq` 续传。
- 前端任务详情新增"实时输出"控制台（GPT 桌面版风格）：
  - 折叠态：仅显示最近活跃流的输出尾巴（20-50 字滚动），并显示并发计数徽标（如"4 路并发 · 已完成 2"）；
  - 展开态：按流分组展示全部 think 与 answer 内容，每流可折叠，自动跟随最新活跃流。
- SSE 连接管理：EventSource + after_seq 续传，任务结束/页面离开时关闭连接。

### Non-goals（第一版不做）

- agent 校验阶段的 token 级流（保持现有粗粒度 trace 事件，第二期接入）。
- OCR / 版面解析阶段的内容流（该阶段无 LLM 输出）。
- 输出内容的持久化分析、导出、检索。

## Capabilities

### New Capabilities

- `llm-stream-relay`: LLM 输出的流式采集与中继能力——skill4re 调用产生 token delta，经 Redis Stream 缓冲，由 SSE 接口对外推送，支持多并发流标识、增量序号、历史重放与心跳。
- `stream-console-ui`: 前端实时输出控制台能力——订阅 SSE，按流分组渲染 think/answer，折叠态滚动显示最近活跃流尾巴，展开态查看全部输出，自动跟随与并发计数。

### Modified Capabilities

- 无（现有抽取流水线行为不变，仅新增旁路观测通道；`enable_thinking` 对抽取结果的影响由开关控制并在 design 中说明风险）。
