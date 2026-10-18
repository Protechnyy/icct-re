# Design

## Context

动机与范围见 `proposal.md`。

- `AppConfig.from_env()` 读取公共 `LLM_*` 配置。`main_model_settings_defaults()` 提供所选 backend 的模型设置默认值。
- `ModelSettingsStore` 将前端设置保存在 `STORAGE_ROOT/settings/model.json`，保存时在同一目录写入临时文件并原子替换。已有文件包含地址、Key 和可选模型名称。
- `DocumentPipeline._extract_relations()` 通过 `ThreadPoolExecutor` 调用共享的 `Skill4ReClient`。每次 `extract_document()` 都执行 `_reload_if_changed()`，模型设置可能在不同 batch 之间更新。
- 模型设置接口提供地址、Key 和模型更新，backend 由进程环境选择。连接检查脚本也包含配置读取逻辑。
- 项目当前没有主规格；相关界面行为记录在 `improve-model-api-settings-ui` 的变更文档中。本变更确立路由与抽取共用配置及文档快照的要求，保留已有单组模型设置的保存语义。

## Goals / Non-Goals

**Goals:**

- 在明确的文档处理边界构造不可变模型配置，并通过参数传递到全部请求位置。
- 让 API、Worker、健康检查和独立脚本使用相同的公共设置解析规则。
- 保持同一文档的模型连接稳定，并保持 batch 并发执行。
- 在配置解析时报告错误，确保 Key 不进入公开结果或普通日志。

**Non-Goals:**

- 本次保留模型设置弹窗的单组配置表单及 OCR 服务配置。

## Decisions

### 1. 公共环境变量

`AppConfig` 统一持有 `llm_backend`、`llm_base_url`、`llm_api_key`、`llm_model`、`llm_request_timeout_seconds` 和 `llm_enable_thinking`。对应环境变量为：

| 环境变量 | 含义与默认行为 |
| --- | --- |
| `LLM_BACKEND` | 请求适配方式，沿用现有远程取值 `vllm`、`qwen_api`、`openai`、`api`，默认 `vllm` |
| `LLM_BASE_URL` | 公共模型服务地址，统一规整 `/chat/completions` 后缀 |
| `LLM_API_KEY` | 公共认证 Key，支持现有无认证兼容服务的使用方式 |
| `LLM_MODEL` | 公共默认模型，默认 `qwen3.8-27b` |
| `LLM_REQUEST_TIMEOUT_SECONDS` | 单次请求超时，默认 `180` 秒 |
| `LLM_ENABLE_THINKING` | 原非流式请求 thinking 开关，默认 `false`，实际参数形式由 backend 决定 |

`STREAM_ENABLED`、`STREAM_THINKING` 及任务上传的 thinking 参数继续表达现有阶段行为。现有请求重试策略保持其行为；相关 `VLLM_MAX_RETRIES`、`VLLM_RETRY_BACKOFF_SECONDS` 及 Skill4RE 同类名称统一迁移为 `LLM_MAX_RETRIES`、`LLM_RETRY_BACKOFF_SECONDS`，并从公共设置显式传递，沿用现有计数含义。

在后端部署配置中检测已停用的明确名称并给出迁移错误，检查范围使用具体变量清单。`OPENAI_API_KEY`、`DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY` 等进程级服务商变量停止作为后端连接来源，其存在本身不触发错误。

### 2. 公共设置解析入口与文件兼容

扩展 `model_settings.py`，由公共解析入口组合 `AppConfig` 默认值和 `ModelSettingsStore.get()` 的保存结果，返回完整的生效请求设置。以 frozen dataclass 表达快照，包含 backend、地址、Key、模型、请求超时及公共请求选项；Key 字段使用 `repr=False`。

保留 `settings/model.json` 的现有地址、Key 和可选模型结构。模型名称省略、清空恢复默认及 Key 留空保留的行为保持一致。backend 和公共请求选项来自统一环境配置，前端继续编辑地址、Key、模型三个字段。修改 backend 需要更新部署环境并重启相应进程；通过前端保存三个字段无需重启。

保存继续使用同目录临时文件和原子替换。解析一次取得整组保存值并构造快照，保证文件在更新期间读取也得到完整配置。解析后校验 backend、URL、模型及数值范围；无效配置在读取位置报告错误。

健康检查每次操作读取当前公共设置；文档任务按自身边界读取快照。健康检查中取得的配置不会修改文档客户端。

### 3. 在开始处理文档时绑定快照

在 `DocumentPipeline.process_task()` 完成任务参数校验后、调用 OCR 之前读取一次公共快照。排队时保存的上传参数不包含 Key，也不包含预先确定的连接配置。Worker 开始执行处理时才确定该文档使用的模型设置。

为该文档创建绑定快照的抽取客户端，将其作为局部变量显式传入 `_extract_relations()`。batch 闭包调用该局部客户端。

文档内部的后续 batch、分块重路由、coref 及抽取后的 Skill4RE 模型调用均使用该客户端持有的配置。运行期间不执行模型设置文件的 fingerprint 检查，也不重新解析环境变量。Skills 文件和路由缓存的现有管理行为与模型配置读取分别处理。

每篇文档绑定自己的客户端配置，不通过修改共享 `self.config`、共享 extractor 或 SDK client 的连接参数实现切换。这样，多个处理时间重叠的文档能够分别持有保存前后的配置，并继续执行 batch 并发。

### 4. 适配 SDK 和 requests

`Skill4ReClient` 的创建入口接受完整快照，`build_backend()` 与 requests 调用显式接收地址、Key、模型和公共请求选项。SDK 的服务商默认环境变量不得覆盖快照。`RelationExtractor` 的调用接口及 pipeline 的 batch 参数传递同步更新。

`VllmClient` 的健康检查和独立请求也使用公共解析结果，确保界面显示的当前模型与后续文档默认使用的模型一致。

单次请求使用 `LLM_REQUEST_TIMEOUT_SECONDS`。无认证服务的 Key 处理按适配器要求在创建客户端时完成，路由与抽取使用相同认证语义。

现有 `local_qwen3` 独立本地调用方式保持原有范围。独立 Skill4RE CLI 的参数体系保持独立。

### 5. API、界面和脚本入口

`GET /api/settings/model` 与 `PUT /api/settings/model` 保留现有请求与响应字段，`default_model` 改取公共默认模型，`backend` 改取公共 backend。保存成功表示公共设置已经保存，后续开始处理的文档使用更新后的值。

模型设置界面的说明表达“关系抽取和 skills 路由共用；保存后对后续开始处理的文档生效”，沿用现有弹窗交互。

连接检查脚本使用公共解析入口。独立脚本已有的显式地址或模型参数仅覆盖该次独立运行的公共配置，并在读取每份文档快照时完成合并。

独立运行在更改结果输出目录时保留公共设置文件的位置，将同一个 `ModelSettingsStore` 显式传入 pipeline，确保前端已保存设置仍然生效。

### 6. 验证方法与证据

使用真实配置文件和真实配置保存接口检查默认值、旧设置文件、模型恢复默认、Key 保留及原子保存；纯配置检查直接调用实际解析对象。

在隔离存储目录中启动真实 API 和 Worker，使用实际可访问的模型服务与 OCR。选用包含多个 batch 的文档并确保能够观察路由请求。验证中途通过实际设置接口保存另一组有效连接或模型，确认进行中的文档保持原配置，后续开始处理的文档使用新配置。增加 OCR 期间更新及排队期间更新两种时序检查。

只记录 task_id、阶段、batch、非敏感模型配置和实际请求结果；请求证据从实际请求发出位置的诊断信息或服务侧记录取得。Key 的一致性通过实际认证及内存中的值比较核验，不写入诊断文件。验证失败直接保留真实错误并继续处理原因，验收使用实际服务返回。验证材料写入已被 Git 忽略的 `data/` 下的专用目录。

## Risks / Trade-offs

- [batch 引用共享可变客户端导致连接变化] → 每个文档持有绑定快照的客户端，batch 闭包引用该文档的局部对象。
- [SDK 自动读取环境变量导致连接来源不一致] → 所有连接参数从快照显式传入，检查 SDK 和 requests 两条请求路径。
- [已有部署配置继续使用停用名称] → 提供完整名称映射，并对明确的停用变量报告迁移错误。
- [诊断输出包含认证信息] → 隐藏 Key 的对象显示内容，公开序列化使用明确允许字段，检查实际日志与结果。

## Migration Plan

1. 更新后端公共字段、解析入口和文档客户端，完成实际配置文件检查及请求超时检查。
2. 将部署配置中的 `SKILL4RE_BACKEND` 迁移为 `LLM_BACKEND`；将当前选中服务商使用的地址、Key、模型迁移为 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`；将请求超时、thinking 和重试配置改为公共名称。停止设置阶段独立连接字段。
3. 更新配置示例、README、项目说明和相关脚本。
4. 保留现有 `settings/model.json`。说明已有保存值优先于公共环境默认值，沿用现有模型名称与 Key 的编辑语义。
5. 完成真实服务的保存、切换和多 batch 验证，更新前端说明并完成构建及 DOM 操作检查。
6. 部署 API 与 Worker 的匹配版本并重启进程。恢复先前运行版本时，同时恢复相应部署配置名称，保留已保存模型文件；代码恢复使用文件编辑操作。
