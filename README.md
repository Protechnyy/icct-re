# Document-Level RE

文档级关系抽取工作台：上传 PDF / 图片后，后端完成 OCR、结构化重排、Skill4RE 技能路由和关系抽取，最终返回 `relation_list` JSON。

## 项目结构

- [backend/](backend/)：Flask API、Redis 任务队列、Worker、OCR / LLM 流水线
- [frontend/](frontend/)：React + Vite + Ant Design 前端工作台
- [skill4re/](skill4re/)：关系抽取框架、技能路由和领域 skill 定义
- [data/](data/)：上传文件、任务结果和临时缓存目录

## 服务依赖

本项目本地开发默认使用以下服务：

- Redis：任务队列和状态存储，默认 `redis://localhost:6379/0`
- PaddleOCR-VL genai_server：OCR 和版面解析，默认 `http://127.0.0.1:8118/v1`
- vLLM / Qwen：关系抽取推理，默认 `http://127.0.0.1:8000/v1`

基本环境要求：Linux、NVIDIA GPU、Python 3.10+、Node.js 18+、Docker、NVIDIA Container Toolkit。

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

后端通过 `PADDLE_OCR_SERVER_URL` 访问该服务。如需调整端口或模型，同步修改 [backend/.env.example](backend/.env.example) 中的配置。

## 启动 vLLM

关系抽取阶段默认使用 Qwen3-32B-AWQ。建议把 vLLM 安装在独立虚拟环境中，避免与后端依赖冲突：

```bash
source ~/venvs/vllm-qwen/bin/activate

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-32B-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key EMPTY \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192 \
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

显存不足时可更换更小模型，并同步修改 `backend/.env` 的 `VLLM_MODEL`。

## 启动后端

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
cp .env.example .env
```

按需修改 `backend/.env` 中的服务地址、模型名和并发参数。常用项：

- `REDIS_URL`
- `PADDLE_OCR_SERVER_URL`
- `PADDLE_OCR_API_MODEL_NAME`
- `VLLM_BASE_URL`
- `VLLM_MODEL`
- `OCR_CONCURRENCY`
- `LLM_CONCURRENCY`

启动 API：

```bash
cd backend
source .venv/bin/activate
python run_api.py
```

启动 Worker：

```bash
cd backend
source .venv/bin/activate
python run_worker.py
```

健康检查：

```bash
curl http://127.0.0.1:5000/api/health
```

## 启动前端

```bash
cd frontend
npm install
npm run dev
```

默认访问地址：[http://localhost:5173](http://localhost:5173)。Vite 会把 `/api` 代理到 `http://127.0.0.1:5000`，因此前端默认不需要 `.env` 或 `.env.example`。

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

## 推荐启动顺序

1. Redis：`sudo docker start docre-redis`
2. PaddleOCR-VL：`sudo docker start docre-paddleocr`
3. vLLM：启动 `vllm serve`
4. 后端 API：`python run_api.py`
5. 后端 Worker：`python run_worker.py`
6. 前端：`npm run dev`

全部启动后，在 [http://localhost:5173](http://localhost:5173) 上传文档并查看抽取结果。
