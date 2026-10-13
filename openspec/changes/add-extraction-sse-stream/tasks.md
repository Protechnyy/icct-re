# Tasks

## 1. 后端：流式调用与 delta 采集

- [x] 1.1 `skill4re/skill4re/backends.py`：OpenAI 兼容路径（openai/api/qwen_api）增加 `stream=True` 分支，迭代 chunk 上报 `delta.reasoning_content`（kind=think）与 `delta.content`（kind=answer），累积全文后复用原有解析逻辑
- [x] 1.2 调用链透传可选 `on_delta` 回调与 `stream_id`：`backend/app/skill4re_client.py` 为每个并发调用生成 `batch-{n}-{skill}` 标识并注入回调；`pipeline.py` 不感知细节
- [x] 1.3 配置项：`STREAM_ENABLED`（默认 true，关闭回退 `stream=False`）与 `STREAM_THINKING`（默认 true，控制 `enable_thinking`）加入 `config.py` 与 `.env.example`
- [x] 1.4 验证：单批抽取在流式与非流式下解析出的三元组一致；关闭 `STREAM_ENABLED` 后行为与现状一致

## 2. 后端：Redis Stream 中继与 SSE 接口

- [x] 2.1 worker 侧中继：delta 事件 `XADD task:{id}:llm`（字段 seq/stream_id/kind/text）并 `XTRIM MAXLEN ~2000`；pipeline 在抽取开始/结束写入 stage 事件（含并发数），任务终态写 done 事件
- [x] 2.2 SSE 接口 `GET /api/tasks/{id}/stream`：`XREAD BLOCK 15000` 循环，支持 `after_seq` 参数与 `Last-Event-ID` 头续传，15 秒心跳注释，`X-Accel-Buffering: no`，终态发 end 事件后关闭
- [x] 2.3 异常路径：Redis 不可用时 SSE 返回 503；流中断时整批重试逻辑不产生重复 stream_id 混淆（重试用新 stream_id 或带 attempt 标记）
- [x] 2.4 验证：curl 消费 SSE 观察到 delta 实时到达、心跳保活、中途断开用 after_seq 重连无重复无遗漏

## 3. 前端：订阅 hook 与实时输出控制台

- [x] 3.1 `useLlmStream(taskId)` hook：EventSource 订阅 + after_seq 续传，事件按 stream_id 归并为 `{think, answer, status}` 映射，维护 activeStreamId 与并发计数，失败进入 degraded 态（静默不弹全局报错）
- [x] 3.2 `StreamConsole.jsx` 折叠态：等宽两行区显示活跃流文本尾巴（slice(-40)），并发徽标（如"4 路并发 · 已完成 2"），展开入口
- [x] 3.3 `StreamConsole.jsx` 展开态：按流分组渲染（think 与 answer 分区、流标识与状态标注、单流可折叠），自动跟随仅在滚动接近底部时生效，手动上滚暂停跟随
- [x] 3.4 生命周期与降级：抽取阶段挂载、终态/页面离开断开；SSE 失败显示占位文案，进度与结果功能不受影响；任务完成后可回看通道保留的全部输出
- [x] 3.5 样式对齐 GPT 桌面版观感：等宽字体、思考区灰斜、delta 渐入动画
- [x] 3.6 验证：浏览器实测三场景——折叠尾巴滚动与活跃流切换、展开态跟随与手动滚动暂停、任务完成后回看

## 4. 端到端验证与收尾

- [x] 4.1 真实多页文档完整流程：OCR → 并发抽取（多流交错）→ agent 校验（不受影响）→ 结果一致
- [x] 4.2 开关回归：`STREAM_THINKING=false` 仅 answer 流；`STREAM_ENABLED=false` 全链路与现状一致
- [x] 4.3 Vite 代理与刷新场景：SSE 经 dev 代理正常；任务进行中刷新页面后重连补历史
