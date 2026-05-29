# Document-Level RE

文档级关系抽取工作台：上传 PDF / 图片后，后端完成 OCR、结构化重排、Skill4RE 技能路由和关系抽取，最终返回 `relation_list` JSON。

## 项目结构

- [backend/](backend/)：Flask API、Redis 任务队列、Worker、OCR / LLM 流水线
- [frontend/](frontend/)：React + Vite + Ant Design 前端工作台
- [skill4re/](skill4re/)：关系抽取框架、技能路由和领域 skill 定义
- [data/](data/)：上传文件、任务结果和临时缓存目录

## 服务依赖

默认开发环境使用：

- Redis：`redis://localhost:6379/0`
- OCR 版面解析服务：默认使用远程 `http://47.108.239.169:31583/layout-parsing`；也可切换为本地 PaddleOCR-VL `http://127.0.0.1:8118/v1`
- 关系抽取 Qwen OpenAI 兼容服务：默认使用远程 `https://api.asukalangely.top/v1/chat/completions`；也可切换为本地 vLLM `http://127.0.0.1:8000/v1`

基本环境要求：Linux、Python 3.10+、Node.js 18+、Redis。若本机启动 PaddleOCR-VL / vLLM，还需要 NVIDIA GPU、Docker 和 NVIDIA Container Toolkit。

## 快速启动

按下面顺序手动启动服务。首次准备后端配置：

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
cp .env.example .env
```

按需修改 `backend/.env`，然后依次启动：

- Redis
- PaddleOCR-VL
- 远程 Qwen API key，或本地 vLLM
- 后端 API
- 后端 Worker
- 前端

默认访问：[http://127.0.0.1:5173](http://127.0.0.1:5173)。

## 启动 Redis

下面命令会直接从镜像源拉取 Redis 7 并启动后台容器：

```bash
sudo docker run -d \
  --name docre-redis \
  --restart unless-stopped \
  -p 6379:6379 \
  m.daocloud.io/docker.io/library/redis:7
```

常用维护命令：

```bash
sudo docker ps
sudo docker logs -f docre-redis
sudo docker restart docre-redis
```

## 启动 PaddleOCR-VL

默认配置会请求远程 OCR 服务，不需要在本机启动 PaddleOCR-VL。如果需要改回本地 Docker 部署的 PaddleOCR-VL，先运行 `genai_server`：

```bash
sudo docker run -d \
  --name docre-paddleocr \
  --restart unless-stopped \
  --gpus all \
  --network host \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
  paddleocr genai_server \
    --model_name PaddleOCR-VL-1.5-0.9B \
    --host 0.0.0.0 \
    --port 8118 \
    --backend vllm
```

默认使用本地 Docker 启动的 PaddleOCR-VL 服务，由后端 PaddleOCR Python API 连接本机 `8118/v1`：

```env
PADDLE_OCR_MODE=python_api
PADDLE_OCR_BASE_URL=http://127.0.0.1:8118
PADDLE_OCR_SERVER_URL=http://127.0.0.1:8118/v1
```

如需临时切回远程 OCR HTTP 接口，可在 `backend/.env` 或启动环境中覆盖：

```env
PADDLE_OCR_MODE=http_api
PADDLE_OCR_BASE_URL=http://47.108.239.169:31583
```

## 关系抽取 Qwen API

默认交付配置使用已经部署好的远程 Qwen3-32B-BF16 OpenAI 兼容接口。复制 `backend/.env.example` 为 `backend/.env` 后，只需要把 `VLLM_API_KEY` 改成有效 key；不需要启动本地 vLLM。

```env
VLLM_BASE_URL=https://api.asukalangely.top/v1/chat/completions
VLLM_API_KEY=你的_api_key
VLLM_MODEL=Qwen3-32B-BF16
VLLM_MAX_RETRIES=3
VLLM_RETRY_BACKOFF_SECONDS=2
SKILL4RE_BACKEND=vllm
SKILL4RE_MODEL=Qwen3-32B-BF16
VLLM_ENABLE_THINKING=false
```

`VLLM_BASE_URL` 既可以填写完整的 `/v1/chat/completions` 地址，也可以填写去掉 `/chat/completions` 后的 `/v1` base URL；后端会在请求关系抽取时自动规整。

远程服务偶发 `502`、`503`、`504` 或 `429` 时，关系抽取请求会按 `VLLM_MAX_RETRIES` 和 `VLLM_RETRY_BACKOFF_SECONDS` 做短重试；如果重试后仍失败，说明远程 API 上游服务不可用，需要稍后重跑任务或联系 API 服务提供方。

## 可选：启动本地 vLLM

如果不使用远程 API，也可以本地启动 vLLM。示例：

```bash
source ~/venvs/vllm-qwen/bin/activate

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-32B-AWQ \
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

## 关系抽取粒度

上传时可选择关系抽取粒度，后端会把配置保存到任务 payload 并在结果中返回 `relation_split_config`、`relation_sections` 和 `relation_batches`。

支持模式：

- `small_section`：按 `1.1`、`4.4` 等小节切分，默认
- `chapter`：按 `一、`、`二、` 等大章切分
- `paragraph`：按 Markdown 段落切分
- `fixed_sections`：每 N 个小节一批

默认配置在 [backend/.env.example](backend/.env.example)：

```env
RELATION_SPLIT_MODE=small_section
RELATION_BATCH_SIZE=1
RELATION_CHAPTER_BATCH_SIZE=2
RELATION_PARAGRAPH_BATCH_SIZE=5
RELATION_FIXED_SECTION_BATCH_SIZE=1
RELATION_MAX_BATCH_TOKENS=2500
RELATION_INCLUDE_PARENT_TITLE=true
RELATION_BATCH_CONCURRENCY=10
```

`RELATION_CHAPTER_BATCH_SIZE` 控制 `chapter` 模式下每批合并几个大章；`RELATION_PARAGRAPH_BATCH_SIZE` 控制 `paragraph` 模式下每批合并几个段落；`RELATION_FIXED_SECTION_BATCH_SIZE` 控制 `fixed_sections` 模式下每批合并几个小节。旧的 `RELATION_BATCH_SIZE` 保留为 fixed sections 的兼容别名。

`RELATION_BATCH_CONCURRENCY` 控制同一文档内关系抽取 batch 的并发数。默认保留 `small_section` 细粒度以保证召回，同时用并发 10 降低远程 API 的总等待时间；如果上游服务不稳定，可以临时改为 `8`、`6` 或更低。

## 手动启动后端

```bash
cd backend
source .venv/bin/activate
python run_api.py
```

另开终端启动 Worker：

```bash
cd backend
source .venv/bin/activate
python run_worker.py
```

健康检查：

```bash
curl http://127.0.0.1:5000/api/health
```

## 手动启动前端

```bash
cd frontend
npm install
npm run dev
```

前端默认使用 `/api`，Vite 会把请求代理到 `http://127.0.0.1:5000`，因此默认不需要 `.env` 或 `.env.example`。

如需指向远程后端：

```bash
VITE_API_BASE_URL=http://your-host:5000/api npm run dev
```

## 常用接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/upload` | 上传文件，表单字段为 `files` |
| `GET` | `/api/status/<task_id>` | 查询任务状态和进度 |
| `GET` | `/api/result/<task_id>` | 获取 OCR、抽取阶段和最终关系结果 |
| `GET` | `/api/health` | 检查 Redis、PaddleOCR-VL 和 vLLM |
| `GET` | `/api/skills` | 获取 Skill4RE skill 列表 |
| `POST` | `/api/skills` | 新增 skill |
| `PUT` | `/api/skills/<name>` | 修改 skill |
