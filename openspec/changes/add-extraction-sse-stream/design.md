# Design

## Context

现状与约束（对应 proposal）：

- 模型调用集中在 `skill4re/skill4re/backends.py` 的 OpenAI 兼容路径，`request_kwargs` 写死 `"stream": False`（约 L234），返回完整 response 后由上层解析 JSON。
- 并发两层：`backend/app/pipeline.py` 批次线程池（`batch_concurrency`）与 `backend/app/skill4re_client.py` 的 `skill4re_max_workers`（默认 4）。worker 与 API 是**两个独立进程**，事件必须跨进程传递。
- 已有可借力的模式：agent 校验的 trace 事件（worker 写 Redis、前端按 `after_seq` 增量轮询）。本设计将传输层升级为 SSE，事件结构保持同构心智。
- 模型为百炼兼容端点上的 `qwen3.8-27b`：支持 `stream=True` 与 `extra_body={"enable_thinking": true}`；思考内容出现在 `delta.reasoning_content`，回答在 `delta.content`。

## Goal / Non-Goal

- Goal：抽取阶段 LLM delta 的采集、跨进程中继、SSE 推送与前端控制台（第一版）。
- Non-Goal：agent 阶段 token 流、OCR 阶段流、输出内容导出与持久化分析（见 proposal Non-goals）。

## Decisions

### D1. delta 采集点：backends.py 增加 stream 分支 + on_delta 回调

在 `backends.py` 的 OpenAI 兼容路径增加 `stream=True` 分支：迭代 chunk，拼装 `delta.reasoning_content`（kind=think）与 `delta.content`（kind=answer），逐段调用可选的 `on_delta(stream_id, kind, text)` 回调；累积全文后走**原有解析路径**，保证抽取逻辑零改动。`skill4re_client.py` 为每次调用生成 stream_id（`batch-{n}-{skill}`），从 pipeline 透传回调注入。skill4re 命令行与其余 backend 不受影响（仅 openai/api/qwen_api 三条兼容路径生效）。

风险与对策：流式与非流式的最终文本由 SDK 保证一致；若流式中断（网络异常），按现有异常处理重试整批，不做断点续传。

### D2. 思考开关：STREAM_THINKING（默认 true）

`enable_thinking` 属于请求体参数，会改变模型行为（更长推理、更慢、token 增多），与 `stream=True`（仅改变交付方式、不改变内容）不同。因此两个开关分离：

- `STREAM_ENABLED`（默认 true）：关闭则回到 `stream=False` 的一次性模式，完全不产生 delta。
- `STREAM_THINKING`（默认 true）：控制 `enable_thinking=true` 是否携带。

已知风险：开启 thinking 后抽取 JSON 可能因推理变长而变化（好坏均可能），该影响通过开关可随时退回；第一版目的是观测，接受此权衡。

### D3. 跨进程中继：Redis Stream（XADD/XREAD），键 `task:{id}:llm`

选 Redis Stream 而非 Pub/Sub：需要「中途订阅补历史 + 断线重连续传」，Stream 的 `XREAD` + 序号语义天然满足，且与 agent trace 事件的 `after_seq` 心智一致。

- 事件字段：`seq`（用 Stream 自身 ID 派生或字段内单调计数）、`stream_id`、`kind`（think/answer/stage/done）、`text`。
- 裁剪：`XADD` 配合 `XTRIM MAXLEN ~2000`，保留最近事件；任务终态后**不删除**，供完成后回看（数据量小，单任务几十 KB 量级）。
- stage/done 事件：pipeline 在抽取阶段开始/结束写入 `stage` 事件（含并发数），便于前端徽标与生命周期判断；终态写 `done`。
- 心跳不写入 Stream，仅 SSE 层周期发送（见 D4）。

### D4. SSE 接口：`GET /api/tasks/{id}/stream`

Flask `Response(generator, mimetype="text/event-stream")`，生成器内循环 `XREAD BLOCK 15000`：

- 无新事件时每 15 秒发送 SSE 注释行（`: ping\n\n`）保活。
- 请求参数 `after_seq`（或 `Last-Event-ID` 头）决定起始位置：缺省 0 表示从头重放（中途打开页面的场景）。
- 事件格式：`event: delta\ndata: {json}\nid: {seq}\n\n`，`kind=done` 后发送 `event: end` 并返回以关闭连接。
- 响应头带 `X-Accel-Buffering: no` 与 `Cache-Control: no-cache`，防止代理缓冲；Vite dev 代理透传 SSE 无需额外配置。
- 需确认 API 进程以 threaded 方式运行（当前 Flask 启动方式即多线程，SSE 长连接占用一个线程，单人使用场景无并发压力；若后续多人使用再考虑 gevent/异步化）。

### D5. 前端：EventSource + 控制台组件

- `useLlmStream(taskId)` hook：`EventSource(`/api/tasks/${id}/stream?after_seq=${n}`)`；`onerror` 时不无限重连（EventSource 自动重连保留，设 `retry` 上限），失败置 `degraded` 状态；事件按 `stream_id` 归并进 `Map<streamId, {think, answer, status}>`，另维护 `activeStreamId`（最近 delta 所在流）与并发计数（由 stage 事件 + 各流 status 派生）。
- 组件 `StreamConsole.jsx`：折叠态为等宽两行区，渲染 `activeStream` 文本尾巴（`slice(-40)`，约 20-50 字），delta 到达即更新；展开态按流分组渲染（think 灰色斜体、answer 常规），自动跟随实现为「用户 scrollTop 接近底部才 auto-scroll」，向上滚动即暂停跟随。
- 折叠态挂载位置：任务详情的抽取阶段区块内（与现有进度条并列），不遮挡结果表格。
- 降级：`degraded` 时折叠视图显示占位文案（如「实时输出不可用」），不弹全局报错（沿用 Task not found 的静默处理风格）。

## Risks / Trade-offs

- `enable_thinking` 改变抽取行为 → 开关可退回（D2）；后续可做同文档 A/B 对比。
- SSE 长连接占用 Flask 线程 → 单人使用无压力，多人场景二期迁移 ASGI。
- Stream 裁剪后早期 delta 不可回放 → MAXLEN 2000 覆盖整个抽取过程（每流数百 delta 量级），实际不会触达；触达时折叠态语义不受影响（尾巴仍是最新）。
- Redis Stream 键残留 → 任务归档时顺带清理（pipeline 终态已有清理点可挂）。

## Migration Plan

纯增量：不改现有接口与数据结构，`STREAM_ENABLED=false` 时系统行为与现状完全一致。上线顺序：后端（backends 分支 → worker 中继 → SSE 接口）→ 前端 hook → 控制台组件，每步可独立验证。

## Open Questions

- 无（三个决策点已与用户确认：thinking 默认开、折叠态跟随最新活跃流、agent 阶段二期）。
