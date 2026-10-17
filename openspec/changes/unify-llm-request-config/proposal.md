# Proposal

## Why

关系抽取与 skills 路由使用的模型设置能够从前端更新，校验智能体仍使用独立连接配置，用户需要重复维护服务地址、Key 和模型名称。用户要求三个阶段始终使用同一组配置，并在每篇文档开始处理时确定完整配置，保证全部 batch 和校验过程使用一致的模型服务。

## What Changes

- 使用公共 `LLM_BACKEND`、`LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL` 配置三个阶段的服务连接，统一单次请求超时的配置名称。
- 统一环境变量默认值、前端已保存设置及健康检查的配置解析入口；前端保存地址、Key 或模型后，后续开始处理的文档自动使用更新后的配置。
- 每篇文档开始处理时读取一次完整配置并创建不可变快照；skills 路由、全部抽取 batch、校验规划和校验工具循环共用快照。
- 将校验智能体的开关、并发、任务数、步数、调用预算、总时限及新增关系上限统一命名为 `VERIFICATION_AGENT_*`。
- **BREAKING**：后端连接配置采用公共环境变量，校验运行参数采用明确前缀；部署配置、校验脚本和文档需要按迁移说明更新，阶段独立连接配置停止生效。
- 保留已有模型设置文件、API Key 留空保留及模型名称留空恢复默认的行为；各阶段继续根据任务需要使用 streaming、thinking、输出限制和工具调用参数。

## Capabilities

### New Capabilities

- `shared-llm-configuration`：公共模型连接配置、前端模型设置的统一作用范围、文档级配置快照和校验智能体运行配置名称。

### Modified Capabilities

无。

## Impact

- 后端：`backend/app/config.py`、`model_settings.py`、`pipeline.py`、`skill4re_client.py`、`vllm_client.py`、`api.py` 及 `agent/` 中的配置传递和模型创建。
- 模型请求：`skill4re/skill4re/service.py`、`backends.py` 中的参数传递及 SDK、requests 调用；确保请求使用任务快照。
- 运维与说明：`backend/.env.example`、README、项目说明，以及连接检查、校验重放和文档评估脚本。
- 前端：模型设置继续维护一组地址、Key 和模型，说明其作用范围及后续文档生效的时间；沿用现有接口与表单结构。
- 验证：公共配置解析、模型设置保存、跨 batch 一致性、处理期间切换配置、真实校验工具调用及密钥保护检查。
