# ICCT-RE（文档级关系抽取）

ICCT-RE 是一个文档级关系抽取工作台。上传 PDF 或图片后，后端依次完成 OCR 版面解析、结构化重排、Skill4RE 技能路由和关系抽取，最终返回 `relation_list` JSON。

## 项目结构

- [backend/](backend/)：Flask API、Redis 任务队列、Worker、OCR / LLM 流水线
- [frontend/](frontend/)：React + Vite + Ant Design 前端工作台
- [skill4re/](skill4re/)：关系抽取框架、技能路由和领域 skill 定义
- [data/](data/)：上传文件、任务结果和临时缓存目录

## 服务依赖

默认开发环境使用：

- Redis（本地）：`redis://localhost:6379/0`
- OCR 版面解析服务（本地）：PaddleOCR-VL-1.6-0.9B，默认地址为 `http://127.0.0.1:8118/v1`
- 关系抽取 Qwen OpenAI 兼容服务：远程 API，默认模型为 `Qwen3-32B-AWQ`

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

默认配置通过远程 OpenAI 兼容 API 调用 `Qwen3-32B-AWQ`。复制 `backend/.env.example` 为 `backend/.env` 后，填写有效的 `VLLM_API_KEY` 即可，无需启动本地 vLLM。

```env
VLLM_BASE_URL=https://api.asukalangely.top/v1/chat/completions
VLLM_API_KEY=你的_api_key
VLLM_MODEL=Qwen3-32B-AWQ
VLLM_MAX_RETRIES=3
VLLM_RETRY_BACKOFF_SECONDS=2
SKILL4RE_BACKEND=vllm
SKILL4RE_MODEL=Qwen3-32B-AWQ
VLLM_ENABLE_THINKING=false
```

`VLLM_BASE_URL` 既可以填写完整的 `/v1/chat/completions` 地址，也可以填写去掉 `/chat/completions` 后的 `/v1` base URL；后端会在请求关系抽取时自动规整。

## 可选：启动本地 vLLM

如需改用本地模型，可启动 vLLM：

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
| `GET` | `/api/skills` | 获取 Skill4RE skill 列表 |
| `POST` | `/api/skills` | 新增 skill |
| `PUT` | `/api/skills/<name>` | 修改 skill |
