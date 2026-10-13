# Spec Delta

## Purpose

为抽取阶段的 LLM 调用提供流式采集与中继能力：将每个并发调用产生的 think/answer delta 实时写入按任务隔离的有序事件通道，并通过 SSE 对外推送，支持多流标识、增量序号、历史重放与心跳保活。

## ADDED Requirements

### Requirement: 流式调用上报 delta

skill4re 的 OpenAI 兼容调用 SHALL 支持流式模式：模型生成的每个增量片段（reasoning_content 与 content）SHALL 通过回调实时上报，携带流标识（stream_id）与内容类别（think 或 answer）。流式模式 SHALL NOT 改变提示词内容、解析逻辑与最终抽取结果的解析入口。思考过程 SHALL 由配置开关控制（默认开启）。

#### Scenario: 开启思考流
- **WHEN** 开关为开启状态且模型返回 reasoning_content 增量
- **THEN** 回调收到 stream_id 与 kind=think 的 delta，内容与模型原始输出一致
- **AND** 该次调用的最终解析结果与非流式调用的解析流程一致

#### Scenario: 关闭思考流
- **WHEN** 开关为关闭状态
- **THEN** 不请求思考过程，回调仅收到 kind=answer 的 delta

### Requirement: delta 事件写入有序通道

每个任务 SHALL 拥有独立的有序事件通道。每条 delta 事件 SHALL 包含单调递增序号（seq）、stream_id、kind 与文本内容。并发的多个 LLM 调用 SHALL 各自使用稳定的 stream_id（含批次与技能标识），同一 stream_id 的事件按生成顺序排列。通道容量 SHALL 有上限（裁剪最旧事件），任务完成后事件 SHALL 保留供查看。

#### Scenario: 并发流交错
- **WHEN** 多个并发调用同时产生 delta
- **THEN** 所有事件写入同一通道且 seq 全局递增，事件间不丢失、不重复
- **AND** 每条事件可通过 stream_id 归属到具体调用

### Requirement: SSE 推送接口

系统 SHALL 提供任务级 SSE 接口，实时推送该任务的 delta 事件与任务阶段变化。连接 SHALL 支持通过客户端携带的序号（Last-Event-ID 或查询参数）从指定位置续传。空闲期 SHALL 周期性发送心跳注释保持连接。任务到达终态后 SHALL 发送结束事件并关闭连接。

#### Scenario: 中途订阅
- **WHEN** 任务抽取进行中，客户端新建立 SSE 连接并携带已有序号
- **THEN** 服务端从该序号之后开始补发历史事件，再继续实时推送

#### Scenario: 断线重连
- **WHEN** 客户端连接中断后以 Last-Event-ID 重连
- **THEN** 服务端从断点之后继续推送，不重复、不遗漏

#### Scenario: 任务结束
- **WHEN** 任务进入终态
- **THEN** SSE 发送结束事件后关闭连接，客户端停止重连

### Requirement: 配置可退回

流式与思考过程 SHALL 均可通过环境变量独立开关。任一开关关闭时 SHALL 平滑回退：关闭流式时调用退回一次性返回模式，行为与现状一致；关闭思考时仅保留回答流。

#### Scenario: 关闭流式开关
- **WHEN** 流式开关为关闭状态
- **THEN** LLM 调用为非流式一次性返回，不产生 delta 事件，SSE 接口对无事件任务正常返回心跳
