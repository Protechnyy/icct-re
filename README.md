# ICCT-RE（文档级关系抽取）

ICCT-RE 是一个文档级关系抽取工作台。上传 PDF 或图片后，后端依次完成 OCR 版面解析、结构化重排、Skill4RE 技能路由和关系抽取，最终返回 `relation_list` JSON。

可选的文档核查阶段在规则去重之后运行，通过 LangGraph 调用原文检索与关系修改工具；结果包含核查前的 `pre_agent_relations` 和核查轨迹 `agent_result`。

## 项目结构

- [backend/](backend/)：Flask API、Redis 任务队列、Worker、OCR / LLM 流水线
- [frontend/](frontend/)：React + Vite + Ant Design 前端工作台
- [skill4re/](skill4re/)：关系抽取框架、技能路由和领域 skill 定义
- [data/](data/)：上传文件、任务结果和临时缓存目录

## 服务依赖

默认开发环境使用：

- Redis（本地）：`redis://localhost:6379/0`
- OCR 版面解析服务（本地）：PaddleOCR-VL-1.6-0.9B，默认地址为 `http://127.0.0.1:8118/v1`
- 关系抽取 Qwen OpenAI 兼容服务：远程 API，默认模型为 `qwen3.8-27b`

## 环境准备

首次使用请先完成本节；随后按本文顺序启动 Redis、PaddleOCR-VL、关系抽取服务，最后启动 ICCT-RE。

创建后端虚拟环境、安装依赖并生成配置文件：

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
cp .env.example .env
```

按需修改 `backend/.env`。后续章节中的默认地址均已写入 `.env.example`；前端依赖会在首次运行启动脚本时自动安装。

## 启动 Redis

首次创建并启动 Redis 7：

```bash
sudo docker run -d \
  --name docre-redis \
  --restart unless-stopped \
  -p 6379:6379 \
  m.daocloud.io/docker.io/library/redis:7
```

停止后再次启动：

```bash
sudo docker start docre-redis
```

## 启动 PaddleOCR-VL

OCR 版面解析默认使用本地部署的 PaddleOCR-VL（`PaddleOCR-VL-1.6-0.9B`）。

首次创建并启动 `paddlex_genai_server`：

```bash
sudo docker run -d \
  --name docre-paddleocr \
  --restart unless-stopped \
  --gpus all \
  --network host \
  -e PADDLE_PDX_MODEL_SOURCE=BOS \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server:latest \
  paddlex_genai_server \
    --model_name PaddleOCR-VL-1.6-0.9B \
    --host 0.0.0.0 \
    --port 8118 \
    --backend vllm
```

停止后再次启动：

```bash
sudo docker start docre-paddleocr
```

后端默认通过 PaddleOCR Python API 连接本机 `8118/v1`：

```env
PADDLE_OCR_MODE=python_api
PADDLE_OCR_BASE_URL=http://127.0.0.1:8118
PADDLE_OCR_SERVER_URL=http://127.0.0.1:8118/v1
```

## 公共模型配置

关系抽取、skills 路由和校验智能体共用一组 OpenAI 兼容模型连接。复制 `backend/.env.example` 为 `backend/.env` 后，填写有效的公共地址、Key 和模型。

```env
LLM_BACKEND=vllm
LLM_BASE_URL=https://llm-bln22h7lns8wvuub.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=你的_api_key
LLM_MODEL=qwen3.8-27b
LLM_REQUEST_TIMEOUT_SECONDS=180
LLM_ENABLE_THINKING=false
LLM_MAX_RETRIES=3
LLM_RETRY_BACKOFF_SECONDS=2
```

`LLM_BASE_URL` 支持 `/v1` 地址或完整 `/v1/chat/completions` 地址。`LLM_BACKEND` 支持 `vllm`、`qwen_api`、`openai`、`api`，选择相应的请求参数形式；三个阶段均使用公共地址、Key 和模型。百炼部署可设置 `LLM_BACKEND=qwen_api`，`RELATION_BATCH_CONCURRENCY=4`。

`LLM_REQUEST_TIMEOUT_SECONDS` 限制单次请求；`VERIFICATION_AGENT_TIMEOUT_SECONDS` 限制整篇文档的校验总时长，校验请求同时受两个时限约束。`LLM_MAX_RETRIES=3` 表示最多尝试三次，包含首次请求；requests 重试等待按 `LLM_RETRY_BACKOFF_SECONDS × 当前尝试次数` 递增，SDK 使用其内置退避，校验限流采用指数退避并计入调用预算。

顶部模型设置编辑公共地址、Key 和模型，保存值优先于环境变量默认值。Key 留空保留现有值，模型名称留空恢复 `LLM_MODEL`。已有 `STORAGE_ROOT/settings/model.json` 可以继续使用。每篇文档在 Worker 开始处理、执行 OCR 之前读取一次配置；路由、全部 batch、校验规划和工具循环使用该快照。排队期间保存设置，文档开始处理时使用新配置；OCR、抽取或校验期间保存设置，进行中的文档继续使用原配置，后续开始处理的文档使用新配置。通过前端保存无需重启，修改部署环境中的 backend 或运行限制需要重启相应进程。

### 配置名称迁移

部署配置使用公共连接字段及 `VERIFICATION_AGENT_*` 运行字段。更新名称后删除对应的旧字段，后端会对停用名称报告迁移错误。

| 配置来源 | 公共配置名称 |
| --- | --- |
| `SKILL4RE_BACKEND` | `LLM_BACKEND` |
| `VLLM_BASE_URL`、`SKILL4RE_BASE_URL`、`AGENT_BASE_URL` | `LLM_BASE_URL` |
| 当前服务使用的 Key：`VLLM_API_KEY`、`DASHSCOPE_API_KEY`、`OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`AGENT_API_KEY` | `LLM_API_KEY` |
| `VLLM_MODEL`、`SKILL4RE_MODEL`、`AGENT_MODEL` | `LLM_MODEL` |
| `VLLM_TIMEOUT_SECONDS` | `LLM_REQUEST_TIMEOUT_SECONDS` |
| `VLLM_ENABLE_THINKING` | `LLM_ENABLE_THINKING` |
| `VLLM_MAX_RETRIES`、`SKILL4RE_API_MAX_RETRIES` | `LLM_MAX_RETRIES` |
| `VLLM_RETRY_BACKOFF_SECONDS`、`SKILL4RE_API_RETRY_BACKOFF_SECONDS` | `LLM_RETRY_BACKOFF_SECONDS` |
| 校验运行参数 `AGENT_*` | 对应的 `VERIFICATION_AGENT_*` |

服务商通用 Key 环境变量可以供其他程序使用，ICCT-RE 的连接认证统一读取 `LLM_API_KEY`。

## 可选：启动本地 vLLM

如需改用本地模型，可启动 vLLM：

```bash
source ~/venvs/vllm-qwen/bin/activate

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3.5-27b \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key EMPTY \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --kv-cache-dtype fp8_e5m2 \
  --enable-prefix-caching
```

验证：

```bash
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer EMPTY"
```

显存不足时可更换更小模型，并修改 `backend/.env` 的 `LLM_MODEL`。

## 启动 ICCT-RE

在依赖服务就绪后，在仓库根目录打开两个终端：

终端一：

```bash
./scripts/start-backend.sh
```

终端二：

```bash
./scripts/start-frontend.sh
```

后端脚本同时启动 API 和 Worker，前端脚本首次会自动安装依赖。访问 [http://127.0.0.1:5173](http://127.0.0.1:5173)；按 `Ctrl+C` 停止后端服务。

健康检查：

```bash
curl http://127.0.0.1:5000/api/health
```

## 关系抽取粒度

上传时可选择以下粒度，默认 `small_section`：

- `small_section`：按 `1.1`、`4.4` 等小节切分，默认
- `chapter`：按 `一、`、`二、` 等大章切分
- `paragraph`：按 Markdown 段落切分
- `fixed_sections`：每 N 个小节一批

其他批处理和并发参数见 [backend/.env.example](backend/.env.example)。

## 常用接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/upload` | 上传文件，表单字段为 `files` |
| `GET` | `/api/status/<task_id>` | 查询任务状态和进度 |
| `GET` | `/api/result/<task_id>` | 获取 OCR、抽取阶段和最终关系结果 |
| `GET` | `/api/agent/<task_id>/events` | 分页读取智能体执行事件 |
| `GET` | `/api/agent/<task_id>/events/<seq>` | 读取指定事件的完整结构化详情 |
| `GET` | `/api/health` | 检查 Redis、PaddleOCR-VL 和 vLLM |
| `GET` | `/api/skills` | 获取 Skill4RE skill 列表 |
| `POST` | `/api/skills` | 新增 skill |
| `PUT` | `/api/skills/<name>` | 修改 skill |

## 文档核查 agent

校验默认关闭。公共模型配置就绪后，修改 `backend/.env` 并重启 Worker：

```dotenv
VERIFICATION_AGENT_ENABLED=true
```

校验使用该文档的公共模型快照，模型需支持 tool calling。校验请求按 backend 显式关闭 thinking，执行记录包含任务规划、请求和工具调用的状态、参数、原文依据及任务结论。上传任务的 `fast_mode` 控制是否执行校验，环境开关提供默认值。

| 配置项 | 默认值 | 含义 |
| --- | --- | --- |
| `VERIFICATION_AGENT_ENABLED` | `false` | 是否运行校验 |
| `VERIFICATION_AGENT_CONCURRENCY` | `1` | 校验任务串行执行，设置大于 1 时按 1 执行 |
| `VERIFICATION_AGENT_MAX_TASKS` | `20` | 每篇最多规划的任务数 |
| `VERIFICATION_AGENT_MAX_STEPS` | `6` | 单任务 LangGraph 节点执行上限，包含模型和工具节点 |
| `VERIFICATION_AGENT_MAX_LLM_CALLS` | `80` | 每篇模型请求预算，限流重试计入 |
| `VERIFICATION_AGENT_TIMEOUT_SECONDS` | `600` | 校验阶段总时限，单位秒 |
| `VERIFICATION_AGENT_MAX_ADDED_PER_TASK` | `5` | 单任务补充关系上限 |

核查期间任务状态为 `verifying`，任务详情标题下方展示智能体过程，结束后仍然可以查看。左侧列出任务，右侧显示执行时间线，包含以下四类信息：

- **规划任务**：任务类型、目标、理由、来源和执行状态。
- **执行步骤**：模型等待、工具名称、请求参数和耗时，同一次调用的开始与结束显示为一个步骤。
- **查看依据**：展开步骤读取完整详情，显示原文、小节、页码、精确命中或候选原文；位置缺失和历史详情缺失均有说明。
- **形成结论**：任务结论、关系或实体的具体改动、原文依据及最终采用情况。关系表中的核查状态可以打开逐关系详情。

过程自动跟随当前任务。选择历史任务、滚动时间线或使用键盘阅读后暂停跟随，点击“跟随当前任务”恢复。展开状态按文档保留；“折叠过程”可以增加结果区域的空间。任务刷新后从首条事件恢复，切换任务时取消此前的读取请求。事件读取和详情读取失败分别提示，可以重新读取，已经读取的内容继续保留。

阶段失败时保留核查前的抽取结果，文档任务仍可成功；部分完成只保留已生效的合法改动。核查状态与文档处理状态分别展示。关闭校验时设置 `VERIFICATION_AGENT_ENABLED=false` 并重启 Worker。