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

下面命令会直接从镜像源拉取 Redis 7 并启动后台容器：

```bash
sudo docker run -d \
  --name docre-redis \
  --restart unless-stopped \
  -p 6379:6379 \
  m.daocloud.io/docker.io/library/redis:7
```

## 启动 PaddleOCR-VL

OCR 版面解析默认使用本地部署的 PaddleOCR-VL（`PaddleOCR-VL-1.6-0.9B`）。先通过 Docker 启动 `paddlex_genai_server`：

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

后端默认通过 PaddleOCR Python API 连接本机 `8118/v1`：

```env
PADDLE_OCR_MODE=python_api
PADDLE_OCR_BASE_URL=http://127.0.0.1:8118
PADDLE_OCR_SERVER_URL=http://127.0.0.1:8118/v1
```

## 关系抽取 Qwen API

默认配置通过远程 OpenAI 兼容 API 调用 `qwen3.8-27b`。复制 `backend/.env.example` 为 `backend/.env` 后，填写有效的 `VLLM_API_KEY` 即可，无需启动本地 vLLM。

```env
VLLM_BASE_URL=https://llm-bln22h7lns8wvuub.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
VLLM_API_KEY=你的_api_key
VLLM_MODEL=qwen3.8-27b
VLLM_MAX_RETRIES=3
VLLM_RETRY_BACKOFF_SECONDS=2
SKILL4RE_BACKEND=vllm
SKILL4RE_MODEL=qwen3.8-27b
VLLM_ENABLE_THINKING=false
```

`VLLM_BASE_URL` 既可以填写完整的 `/v1/chat/completions` 地址，也可以填写去掉 `/chat/completions` 后的 `/v1` base URL；后端会在请求关系抽取时自动规整。

使用 `SKILL4RE_BACKEND=qwen_api` 时，密钥来自 `DASHSCOPE_API_KEY`，自定义部署地址请设置 `SKILL4RE_BASE_URL`；SDK 和 requests 两条调用路径均遵循这个地址。示例默认使用 `vllm` 适配器调用兼容 API，密钥来自 `VLLM_API_KEY`。切到百炼部署时建议设置 `RELATION_BATCH_CONCURRENCY=4`，遇到限流再调低。

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

显存不足时可更换更小模型，并同步修改 `backend/.env` 的 `VLLM_MODEL` 和 `SKILL4RE_MODEL`。

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

核查默认关闭（`AGENT_ENABLED=false`）。启用时修改 `backend/.env`，填写独立的 `AGENT_API_KEY`，然后重启 worker：

```dotenv
AGENT_ENABLED=true
AGENT_BASE_URL=https://llm-bln22h7lns8wvuub.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
AGENT_API_KEY=your_bailian_api_key
AGENT_MODEL=qwen3.8-27b
```

核查使用独立密钥配置，不会自动读取抽取阶段的密钥。即使两个阶段使用同一部署，也须分别配置。模型调用显式关闭思考，执行记录包含任务规划、请求和工具调用的状态、参数、原文依据及任务结论。

| 配置项 | 默认值 | 含义 |
| --- | --- | --- |
| `AGENT_ENABLED` | `false` | 是否运行核查 |
| `AGENT_BASE_URL` | 上述北京百炼地址 | OpenAI 兼容地址，可带 `/chat/completions` 后缀 |
| `AGENT_API_KEY` | 空 | 独立核查密钥 |
| `AGENT_MODEL` | `qwen3.8-27b` | 核查模型 |
| `AGENT_CONCURRENCY` | `1` | 首版串行执行，设大于 1 会收敛为 1 |
| `AGENT_MAX_TASKS` | `20` | 每篇最多规划的任务数 |
| `AGENT_MAX_STEPS` | `6` | 单任务 LangGraph 图节点执行上限，包含模型和工具节点 |
| `AGENT_MAX_LLM_CALLS` | `80` | 每篇模型请求预算，限流重试计入 |
| `AGENT_TIMEOUT_SECONDS` | `600` | 核查阶段总时限，单位秒 |
| `AGENT_MAX_ADDED_PER_TASK` | `5` | 单任务补充关系上限 |

核查期间任务状态为 `verifying`，任务详情标题下方展示智能体过程，结束后仍然可以查看。左侧列出任务，右侧显示执行时间线，包含以下四类信息：

- **规划任务**：任务类型、目标、理由、来源和执行状态。
- **执行步骤**：模型等待、工具名称、请求参数和耗时，同一次调用的开始与结束显示为一个步骤。
- **查看依据**：展开步骤读取完整详情，显示原文、小节、页码、精确命中或候选原文；位置缺失和历史详情缺失均有说明。
- **形成结论**：任务结论、关系或实体的具体改动、原文依据及最终采用情况。关系表中的核查状态可以打开逐关系详情。

过程自动跟随当前任务。选择历史任务、滚动时间线或使用键盘阅读后暂停跟随，点击“跟随当前任务”恢复。展开状态按文档保留；“折叠过程”可以增加结果区域的空间。任务刷新后从首条事件恢复，切换任务时取消此前的读取请求。事件读取和详情读取失败分别提示，可以重新读取，已经读取的内容继续保留。

阶段失败时保留核查前的抽取结果，文档任务仍可成功；部分完成只保留已生效的合法改动。核查状态与文档处理状态分别展示。关闭核查时设置 `AGENT_ENABLED=false` 并重启 Worker。

### 执行事件查询

```bash
curl --noproxy '*' 'http://127.0.0.1:5000/api/agent/<task_id>/events?after_seq=0&limit=100'
curl --noproxy '*' 'http://127.0.0.1:5000/api/agent/<task_id>/events/<seq>'
```

`after_seq` 为已经读取的最后序号，从 `0` 开始；`limit` 默认为 `100`，允许 `1` 至 `200`。返回 `events`、`next_seq`、`last_seq`、`has_more`、阶段状态和任务清单。`has_more=true` 时立即使用 `next_seq` 请求下一页，之后每 2.5 秒继续读取。参数无效返回 HTTP 400，任务或详情不存在返回 HTTP 404。

新记录使用 `version: 2`。每篇文档的 `seq` 从 `1` 连续增加，包含 `document_task_id`、核查 `task_id`、UTC `timestamp`、`kind`、`phase`、`status`、`call_id`、`args`、`elapsed_seconds` 和最多 500 个字符的 `summary`。原有 `result` 摘要及 `truncated` 保留；完整 `result_data` 通过详情接口读取，`detail_available` 表示是否存在结构化结果。

事件类型包括 `phase_start`、`plan_start`、`plan`、`task_start`、`model_start`、`model_end`、`tool_start`、`tool`、`conclusion`、`task_end` 和 `phase_end`。同一次模型或工具调用使用相同 `call_id`；改动使用稳定 `change_id`，`phase_end` 中的 `accepted_change_ids` 和 `discarded_change_ids` 说明最终采用情况。

Redis 按文档保存事件摘要 List、详情 Hash 和元数据 Hash，同一次 transaction 完成写入后发布进度快照。`agent_progress.recent_events` 保留最近 20 条，完整事件可通过上述接口读取。旧结果读取其已保存的轨迹，缺失结构化详情时显示“结构化详情未记录”。

### 真实数据与服务检查

下面的检查连接已配置的真实 Redis，创建独立的 `agent-event-check-*` 任务，使用已有结果中的原文执行工具，检查并发序号、完整详情、拒绝、未命中、分页和 HTTP 状态。`--history-result` 可指定已保存的旧核查结果，检查材料保存到 `--output-dir`：

```bash
STORAGE_ROOT="$PWD/data" backend/.venv/bin/python scripts/check-agent-execution-events.py \
  data/results/<task_id>/result.json \
  --history-result data/agent_replays/<run>/<task_id>/result.json \
  --output-dir .impeccable/checks/agent-events
```

增加 `--live --max-calls 24` 会将输入原文发送到配置的真实核查模型，并保存核查结果与阶段结束事件；设置 `--max-calls 1` 可以检查预算受限状态。执行前须确认原文允许发送到该模型服务。

只重放已保存的结果，无需 OCR、Redis 或前端，输出目录必须与输入目录独立：

```bash
backend/.venv/bin/python scripts/replay-agent-verification.py \
  data/results/<task_id>/result.json \
  --output-dir data/agent_replays/my-run
```

输出包含 `result.json`、`metrics.json`、`changes.csv` 和多文档 `evaluation.json`。可以用 `--annotations annotations.json` 提供标注；单篇为关系列表，多篇为 task_id 到关系列表的映射。精确率、召回率和 F1 按规范化后的头实体、关系、尾实体严格匹配，关系词同义表达不会自动等同。无标注时明确跳过这些指标。

在 `changes.csv` 的 `judgment` 列填写 `正确` 或 `错误` 后，可以重新计算整体和分类型改动正确率：

```bash
backend/.venv/bin/python scripts/replay-agent-verification.py \
  --judgments-csv data/agent_replays/my-run/<task_id>/changes.csv
```

本次五领域材料与评估可通过以下脚本复现。第二个命令会实际调用本地 OCR 和远程模型，需要先配置有效的抽取及核查密钥；它会复用已有输出，重新运行请选择新输出目录。

```bash
backend/.venv/bin/python scripts/generate-agent-evaluation-documents.py \
  --output-dir data/agent_evaluation/my-run/documents
backend/.venv/bin/python scripts/run-agent-document-evaluation.py \
  --documents-dir data/agent_evaluation/my-run/documents \
  --output-dir data/agent_evaluation/my-run
```

2026-10-09 的五篇合成文档实测核查为 10.8–29.7 秒，中位数 17.7 秒，五篇均部分完成；这不是完整核查的耗时承诺。助手逐条审阅的改动正确率为 5/6（83.33%），发现别名合并产生自指关系，且没有删除样本，未能证明全部开启门槛。建议保持默认关闭。较长的真实项目文档此前用旧模型约 175 秒，也为部分完成；结果不能直接外推到新模型或其他长度文档。详细指标与逐条判定见 [五领域评估报告](openspec/changes/add-document-agent-verification/five-domain-evaluation.md)。
