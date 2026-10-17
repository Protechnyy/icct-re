# Spec Delta

## Purpose

让关系抽取、skills 路由和校验智能体使用统一的模型连接配置，并通过文档开始处理时读取的完整配置快照，保证全部 batch 与后续校验请求一致。前端保存的模型设置影响后续开始处理的文档，校验运行限制使用能够明确表达校验用途的配置名称。

## ADDED Requirements

### Requirement: 三个阶段使用公共连接配置

系统 SHALL 使用 `LLM_BACKEND`、`LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL` 配置关系抽取、skills 路由和校验智能体的公共模型服务。三个阶段 MUST 使用同一组生效值，且不提供阶段独立的服务商、地址、Key 或模型覆盖。

#### Scenario: 仅配置公共连接
- **WHEN** 用户配置一组有效的公共连接参数并处理启用校验的文档
- **THEN** 路由、抽取及校验请求都使用该连接的地址、Key 和模型，并按照公共 backend 的要求组织请求

#### Scenario: 校验使用相同模型
- **WHEN** 文档进入校验规划和工具循环
- **THEN** 所有校验模型请求使用该文档抽取和路由阶段使用的公共连接配置

### Requirement: 前端设置覆盖公共默认值

系统 SHALL 以环境变量提供公共默认连接，并允许前端已保存的地址、Key 和模型覆盖相应默认值。前端模型设置 SHALL 对三个阶段统一生效，读取接口 SHALL 返回当前生效模型、默认模型、backend 和 Key 配置状态。

#### Scenario: 未保存前端设置
- **WHEN** 当前没有已保存的模型设置，文档开始处理
- **THEN** 三个阶段使用公共环境变量提供的默认连接

#### Scenario: 保存新的公共连接
- **WHEN** 用户通过模型设置保存有效的新地址、Key 和模型，随后开始处理另一篇文档
- **THEN** 该文档三个阶段均使用新连接，且无需重启 API 或 Worker

### Requirement: 文档开始处理时读取完整配置快照

系统 SHALL 在 Worker 开始处理文档、执行 OCR 之前读取一次完整的生效模型配置。路由、所有抽取 batch、校验规划和工具循环 MUST 共用该不可变配置快照，后续请求 MUST 不重新读取已保存设置。文档的开始时间 SHALL 以开始执行处理为准。

#### Scenario: 多个 batch 使用同一配置
- **WHEN** 一篇文档包含串行或并发执行的多个抽取 batch
- **THEN** 所有 batch 的路由和抽取请求均使用该文档开始处理时取得的连接配置

#### Scenario: 排队期间修改配置
- **WHEN** 文档已经进入队列但尚未开始处理，用户保存新模型配置
- **THEN** 该文档开始处理时读取新配置

#### Scenario: 校验开始前修改配置
- **WHEN** 文档已经开始处理，用户在其进入校验之前保存新配置
- **THEN** 该文档的校验继续使用开始处理时读取的配置

### Requirement: 配置更新与进行中的文档隔离

系统 SHALL 使保存设置后的新配置作用于后续开始处理的文档。进行中的文档 MUST 保持自身配置快照，包括已开始、排队等待执行及尚未创建的后续 batch。不同文档的模型客户端 MUST 不相互改变连接配置。

#### Scenario: OCR 期间保存设置
- **WHEN** 文档正在执行 OCR，用户保存另一组连接配置
- **THEN** 该文档的后续模型请求继续使用 OCR 开始之前确定的连接配置

#### Scenario: 不同文档处理时间重叠
- **WHEN** 一篇文档使用原配置处理中，另一篇文档在新配置保存后开始处理
- **THEN** 两篇文档各自的全部模型请求使用各自开始处理时取得的配置

### Requirement: 保持模型设置的保存语义

系统 SHALL 继续读取已有的单组模型设置文件。更新时省略模型字段 SHALL 保留当前模型，模型名称为空或等于默认模型 SHALL 恢复默认；省略 Key 或提交空白 Key SHALL 保留现有 Key。配置保存 MUST 原子更新整组设置，使读取方取得完整的一组值。

#### Scenario: 读取已有设置文件
- **WHEN** 已保存文件包含地址、Key 和可选模型名称
- **THEN** 系统将其作为三个阶段的公共设置，缺少模型字段时使用公共默认模型

#### Scenario: 恢复默认模型并保留 Key
- **WHEN** 用户清空模型名称并保存，Key 没有提供有效新值
- **THEN** 系统使用公共默认模型并保留当前 Key，后续开始处理的文档使用该组合

#### Scenario: 保存与开始处理同时发生
- **WHEN** 用户保存整组设置的同时，文档开始读取配置
- **THEN** 文档取得完整的保存前配置或完整的保存后配置

### Requirement: 校验运行参数明确表达用途

系统 SHALL 使用 `VERIFICATION_AGENT_ENABLED`、`VERIFICATION_AGENT_CONCURRENCY`、`VERIFICATION_AGENT_MAX_TASKS`、`VERIFICATION_AGENT_MAX_STEPS`、`VERIFICATION_AGENT_MAX_LLM_CALLS`、`VERIFICATION_AGENT_TIMEOUT_SECONDS` 和 `VERIFICATION_AGENT_MAX_ADDED_PER_TASK` 配置校验运行。运行限制的含义与默认值 SHALL 保持一致。

#### Scenario: 使用校验运行参数
- **WHEN** 用户配置 `VERIFICATION_AGENT_*` 并处理文档
- **THEN** 校验使用指定的开关和资源限制，模型连接仍取文档公共配置

#### Scenario: 保持任务级极速模式
- **WHEN** 上传任务显式指定 `fast_mode`
- **THEN** 系统继续按照该任务的 `fast_mode` 决定是否运行校验，环境变量提供未指定时的默认行为

### Requirement: 单次请求超时与校验总时限分别生效

系统 SHALL 使用 `LLM_REQUEST_TIMEOUT_SECONDS` 设置公共单次请求超时，使用 `VERIFICATION_AGENT_TIMEOUT_SECONDS` 设置校验阶段总时限。校验单次请求的等待时长 MUST 同时受到公共请求超时及剩余校验时限约束。各阶段 SHALL 保留符合其任务的输出限制、streaming、thinking 和工具调用行为。

#### Scenario: 剩余校验时间较短
- **WHEN** 校验阶段剩余时间小于公共单次请求超时
- **THEN** 随后校验模型请求的等待时间不超过剩余校验时限

#### Scenario: 三个阶段使用不同请求结构
- **WHEN** 抽取需要结构化关系输出，校验需要工具调用
- **THEN** 两者使用同一组连接配置，并分别发送各自所需的请求参数

### Requirement: 公共配置错误明确终止处理

系统 SHALL 在文档开始处理时校验公共连接配置；不支持的 backend、无效 URL、空模型及无效请求超时 MUST 导致明确错误，系统 MUST 不改用其他阶段或其他服务商的连接。已保存设置格式错误 MUST 报告错误。配置读取失败 MUST 在 OCR 和模型请求之前终止该文档处理。

#### Scenario: 保存文件损坏
- **WHEN** 文档开始处理时无法解析已保存的设置文件
- **THEN** 文档报告配置错误，且不发送 OCR 或模型请求

#### Scenario: 连接配置无效
- **WHEN** 文档开始处理时公共配置包含不支持的 backend 或无效模型连接
- **THEN** 文档报告明确的配置错误

### Requirement: 密钥保持私密

系统 MUST 不在模型设置读取响应、任务结果、事件、普通日志或配置对象的普通显示内容中包含 API Key。配置快照中的 Key SHALL 仅在运行期间供请求认证使用；持久设置 SHALL 继续采用受限权限保存。

#### Scenario: 查看任务或配置摘要
- **WHEN** 用户读取任务结果、模型设置或运行日志
- **THEN** 内容不包含 API Key，设置响应仅表达 Key 是否配置

### Requirement: 配置迁移与相关入口保持一致

系统 SHALL 在配置示例和使用说明中提供公共连接及 `VERIFICATION_AGENT_*` 的完整配置方法。后端若发现已停用的阶段连接配置或校验运行名称，MUST 给出相应的新名称并要求更新。连接检查、校验重放和文档评估入口 SHALL 使用相同的公共配置解析规则。

#### Scenario: 部署仍使用阶段连接配置
- **WHEN** 后端启动配置包含已停用的阶段连接变量
- **THEN** 系统给出明确迁移错误，说明需要使用的公共配置名称

#### Scenario: 单独运行校验重放
- **WHEN** 用户使用公共配置对文档结果执行校验重放
- **THEN** 重放为该文档读取一次公共配置，并将其用于全部校验请求
