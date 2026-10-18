# ICCT-RE（文档级关系抽取）

ICCT-RE 是一个文档级实体与关系抽取工作台。上传 PDF、图片、TXT 或 Markdown 后，后端识别网页配置范围内的全部实体及实体类型，并抽取具有原文证据的指定关系。结果使用独立的 `entities`、`relations` 集合。

结果格式为版本 4。实体表展示名称、别名和类型，关系表展示头实体、关系和尾实体及类型。JSON、实体 CSV 和关系 CSV 使用业务字段；原文引用仅参与处理期间的身份判断与校验。历史结果读取时转换，已有文件保持原样。字段示例见 [抽取配置与结果](docs/extraction-schema.md)。

结果页面提供文档预览、实体、关系和运行日志。PDF 和图片完成 OCR 后即可预览原文；Markdown 和 TXT 直接读取原文并跳过 OCR。预览在抽取期间保持可用。

## 项目结构

- [backend/](backend/)：Flask API、Redis 任务队列、Worker、OCR / LLM 流水线
- [frontend/](frontend/)：React、Vite、Arco Design 前端工作台
- [skill4re/](skill4re/)：模型请求适配、类型定义、实体识别与关系抽取
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

实体识别、批量实体归并和关系抽取共用一组 OpenAI 兼容模型连接。每个原文批次分别执行一次实体识别和关系抽取，返回结果执行程序校验。复制 `backend/.env.example` 为 `backend/.env` 后，填写有效的公共地址、Key 和模型。

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

`LLM_BASE_URL` 支持 `/v1` 地址或完整 `/v1/chat/completions` 地址。`LLM_BACKEND` 支持 `vllm`、`qwen_api`、`openai`、`api`，选择相应的请求参数形式；全部实体与关系阶段使用公共地址、Key 和模型。远程服务需要支持 JSON Schema 结构化输出。百炼部署可设置 `LLM_BACKEND=qwen_api`，`RELATION_BATCH_CONCURRENCY=4`。

`LLM_REQUEST_TIMEOUT_SECONDS` 限制单次请求。`LLM_MAX_RETRIES=3` 表示最多尝试三次，包含首次请求。网页抽取的重试等待按 `LLM_RETRY_BACKOFF_SECONDS × 当前尝试次数` 递增，SDK 自身重试关闭。

顶部模型设置编辑公共地址、Key 和模型，保存值优先于环境变量默认值。Key 留空保留现有值，模型名称留空恢复 `LLM_MODEL`。已有 `STORAGE_ROOT/settings/model.json` 可以继续使用。每篇文档在 Worker 开始处理、执行 OCR 之前读取一次模型设置与抽取配置；全部实体与关系 batch 使用文档自己的快照。排队期间保存设置，文档开始处理时使用新配置；OCR 或抽取期间保存设置，进行中的文档继续使用原配置，后续开始处理的文档使用新配置。通过前端保存无需重启，修改部署环境中的 backend 或运行限制需要重启相应进程。

顶部“抽取配置”维护实体类型、父类型、关系定义与逐行头尾类型组合，保存在 `STORAGE_ROOT/settings/extraction-schema.json`。默认提供指挥、协同、情报保障，以及武器平台的四个子类型。保存冲突返回 409，进行中文档保持原版本。实体包含未参与关系的独立实体，保存名称、类型和别名。字段、示例、提示词与导出说明见 [项目说明](docs/ICCT-RE%20项目说明.md) 和 [抽取配置与结果](docs/extraction-schema.md)。

### 配置名称迁移

部署配置使用公共连接字段。更新名称后删除对应的旧字段，后端会对停用名称报告迁移错误。

| 配置来源 | 公共配置名称 |
| --- | --- |
| `SKILL4RE_BACKEND` | `LLM_BACKEND` |
| `VLLM_BASE_URL`、`SKILL4RE_BASE_URL` | `LLM_BASE_URL` |
| 当前服务使用的 Key：`VLLM_API_KEY`、`DASHSCOPE_API_KEY`、`OPENAI_API_KEY`、`DEEPSEEK_API_KEY` | `LLM_API_KEY` |
| `VLLM_MODEL`、`SKILL4RE_MODEL` | `LLM_MODEL` |
| `VLLM_TIMEOUT_SECONDS` | `LLM_REQUEST_TIMEOUT_SECONDS` |
| `VLLM_ENABLE_THINKING` | `LLM_ENABLE_THINKING` |
| `VLLM_MAX_RETRIES`、`SKILL4RE_API_MAX_RETRIES` | `LLM_MAX_RETRIES` |
| `VLLM_RETRY_BACKOFF_SECONDS`、`SKILL4RE_API_RETRY_BACKOFF_SECONDS` | `LLM_RETRY_BACKOFF_SECONDS` |

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
| `GET` | `/api/health` | 检查 Redis、PaddleOCR-VL 和 vLLM |
| `GET`、`PUT` | `/api/settings/extraction-schema` | 读取或保存完整实体与关系配置 |
| `GET` | `/api/result/<task_id>/entities.csv` | 下载包含全部独立实体的实体 CSV |
| `GET` | `/api/tasks/<task_id>/preview` | 获取已读取或完成 OCR 的文档原文，尚未就绪时返回 409 |
