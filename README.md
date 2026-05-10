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
- PaddleOCR-VL OpenAI 兼容服务：`http://127.0.0.1:8118/v1`
- vLLM / Qwen OpenAI 兼容服务：`http://127.0.0.1:8000/v1`

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
- vLLM
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

Worker 启动前需要先运行 PaddleOCR-VL `genai_server`：

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

后端通过 `PADDLE_OCR_SERVER_URL` 访问该服务。如需使用远程 OCR 服务，修改 `backend/.env` 或启动时传入环境变量：

```env
PADDLE_OCR_BASE_URL=http://your-host:8118
PADDLE_OCR_SERVER_URL=http://your-host:8118/v1
```

## 启动 vLLM

关系抽取阶段默认使用 OpenAI 兼容接口。示例：

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
RELATION_MAX_BATCH_TOKENS=2500
RELATION_INCLUDE_PARENT_TITLE=true
```

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
