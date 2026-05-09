# Document-Level RE

## 当前本地启动方式

当前仓库已改成一键启动脚本，优先按本节操作。下面旧的 Docker OCR / 本地 vLLM 章节是原项目说明，当前本地测试不按那套启动。

### 依赖服务

启动前确认这些服务可访问：

```bash
redis-cli ping
curl http://192.168.1.2:8080/v1/models -H "Authorization: Bearer EMPTY"
curl http://192.168.1.2:9000/v1/models -H "Authorization: Bearer EMPTY"
```

当前约定：

- Redis 使用本机服务：`redis://localhost:6379/0`，不用 Docker。
- OCR 远程服务：`http://192.168.1.2:8080/v1`，模型 `PaddleOCR-VL-1.5-0.9B`。
- 关系抽取 LLM 远程服务：`http://192.168.1.2:9000/v1`，模型 `Qwen3-8B`。
- 后端 conda 环境：`/home/ybxy/下载/enter/envs/icct-re`。
- 后端环境不安装 `vllm`，不安装 `paddlepaddle-gpu`。

### 一键启动

```bash
cd /mnt/code3/recctt/icct-re
./start.sh
```

启动后打开：

```text
http://127.0.0.1:5173
```

停止：

```bash
cd /mnt/code3/recctt/icct-re
./start.sh stop
```

日志：

```text
data/tmp/api-5011.log
data/tmp/worker.log
data/tmp/frontend-5173.log
```

结果保存位置：

```text
data/results/<task_id>/result.json
data/results/<task_id>/document_text.md
data/results/<task_id>/ocr_paragraphs.json
```

关系抽取结果在：

```json
final_relations
```

### start.sh 配置怎么改

`start.sh` 里所有关键配置都可以通过环境变量覆盖，不需要直接改 Python 代码。

改 OCR 服务地址或 OCR 模型：

```bash
cd /mnt/code3/recctt/icct-re
PADDLE_OCR_SERVER_URL=http://192.168.1.2:8080/v1 \
PADDLE_OCR_API_MODEL_NAME=PaddleOCR-VL-1.5-0.9B \
./start.sh
```

对应默认值在 [start.sh](start.sh)：

```bash
PADDLE_OCR_BASE_URL=http://192.168.1.2:8080
PADDLE_OCR_SERVER_URL=http://192.168.1.2:8080/v1
PADDLE_OCR_API_MODEL_NAME=PaddleOCR-VL-1.5-0.9B
PADDLE_OCR_MAX_CONCURRENCY=32
```

改关系抽取 LLM，也就是原项目里说的 vLLM/OpenAI 兼容接口：

```bash
cd /mnt/code3/recctt/icct-re
VLLM_BASE_URL=http://192.168.1.2:9000/v1 \
VLLM_MODEL=Qwen3-8B \
SKILL4RE_MODEL=Qwen3-8B \
./start.sh
```

对应默认值在 [start.sh](start.sh)：

```bash
VLLM_BASE_URL=http://192.168.1.2:9000/v1
VLLM_MODEL=Qwen3-8B
SKILL4RE_MODEL=Qwen3-8B
```

改关系抽取每批 section 数：

```bash
cd /mnt/code3/recctt/icct-re
RELATION_SECTION_BATCH_SIZE=1 ./start.sh
```

当前默认是：

```bash
RELATION_SECTION_BATCH_SIZE=2
```

改 API / 前端端口：

```bash
cd /mnt/code3/recctt/icct-re
API_PORT=5012 FRONTEND_PORT=5174 ./start.sh
```

改 Python 环境：

```bash
cd /mnt/code3/recctt/icct-re
PYTHON_BIN=/path/to/python ./start.sh
```

当前默认是：

```bash
PYTHON_BIN=/home/ybxy/下载/enter/envs/icct-re/bin/python
```

更多当前改动说明见：

- [CHANGES_OCR_RELATION_BACKEND.md](CHANGES_OCR_RELATION_BACKEND.md)
- [TODO_RELATION_BATCHING.md](TODO_RELATION_BATCHING.md)

---

本项目实现一个文档级关系抽取（Document-level Relation Extraction）：

- **Flask API**：上传文档、查询任务状态、获取抽取结果
- **Worker**：从 Redis 队列消费任务，串联 OCR → 分块 → 关系抽取流水线
- **PaddleOCR-VL**：通过官方 Python API（`PaddleOCRVL`）调用本机 `genai_server` 完成版面解析与表格还原
- **Skill4RE**：作为关系抽取框架，提供技能路由、长文档分块、结果合并和 `relation_list` JSON 输出
- **vLLM / Qwen**：作为 Skill4RE 的 LLM 后端，提供 OpenAI 兼容的 `chat/completions` 接口
- **React + Vite + Ant Design**：前端工作台，支持批量上传、轮询任务状态、查看分阶段抽取结果

## 目录结构

- [backend/](backend/)：Flask API、Redis 任务存储、OCR/LLM 流水线、Worker
  - [backend/app/](backend/app/)：核心模块（[api.py](backend/app/api.py)、[worker.py](backend/app/worker.py)、[pipeline.py](backend/app/pipeline.py)、[skill4re_client.py](backend/app/skill4re_client.py)、[paddle_ocr.py](backend/app/paddle_ocr.py)、[vllm_client.py](backend/app/vllm_client.py)、[config.py](backend/app/config.py)）
  - [backend/run_api.py](backend/run_api.py)、[backend/run_worker.py](backend/run_worker.py)：API / Worker 启动入口
- [skill4re/](skill4re/)：技能路由关系抽取框架与领域技能定义
- [frontend/](frontend/)：React 单页工作台
- [data/](data/)：上传文件与结果落盘目录（由 `STORAGE_ROOT` 控制）

## 服务拓扑

```
浏览器 ──► 前端 (Vite, :5173) ──► Flask API (:5000)
                                       │
                                       ▼
                                   Redis (:6379)
                                       │
                                       ▼
                                   Worker
                                  ┌──┴──┐
                                  ▼     ▼
                  PaddleOCR-VL genai_server (:8118)   vLLM / Qwen (:8000)
```

需要在本机或可达网络上分别准备：**Redis**、**PaddleOCR-VL genai_server**、**vLLM**。

## 环境准备

- Linux + NVIDIA GPU（当前依赖按 CUDA 13.0 / cu130 PaddlePaddle 轮子配置，详见 [backend/requirements.txt](backend/requirements.txt)）
- Python 3.10+
- Node.js 18+
- Redis 5+

本机使用 Docker 在后台运行 Redis（容器名 `docre-redis`，宿主端口 6379，开机自启）：

```bash
sudo docker run -d \
  --name docre-redis \
  --restart unless-stopped \
  -p 6379:6379 \
  redis:7
```

如果拉取官方镜像受限，可改用国内镜像源 `m.daocloud.io/docker.io/library/redis:7`。容器启动后默认监听 `redis://localhost:6379/0`，与 [.env.example](backend/.env.example) 中的 `REDIS_URL` 一致。常用维护命令：

```bash
sudo docker ps                    # 确认 docre-redis 在运行
sudo docker logs -f docre-redis   # 查看日志
sudo docker restart docre-redis   # 重启
```

## 启动 PaddleOCR-VL genai_server（Docker）

参考 [PaddleOCR-VL 官方文档 §3.1.1 使用 Docker 镜像启动](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html#311-docker)，本项目使用 Paddle 官方提供的 `paddleocr-genai-vllm-server` 镜像在本机加载 VL 识别模型（`PaddleOCR-VL-1.5-0.9B`），并在 8118 端口暴露 OpenAI 兼容接口供后端 OCR 流水线调用。该服务必须先于 Worker 启动。

> 前置条件：已安装 NVIDIA 驱动与 [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，使 Docker 可以使用 `--gpus all`。

```bash
sudo docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
    paddleocr genai_server \
        --model_name PaddleOCR-VL-1.5-0.9B \
        --host 0.0.0.0 \
        --port 8118 \
        --backend vllm
```

要让该容器也常驻后台，可把 `-it --rm` 换成 `-d --restart unless-stopped --name docre-paddleocr`。

启动完成后，服务在 `http://127.0.0.1:8118/v1` 暴露 OpenAI 兼容接口；OCR 流水线（[paddle_ocr.py](backend/app/paddle_ocr.py)）会以 `vl_rec_backend=vllm-server` + `vl_rec_server_url=$PADDLE_OCR_SERVER_URL` 调用它。如需更换模型或端口，请同步修改 `backend/.env` 的 `PADDLE_OCR_API_MODEL_NAME` 与 `PADDLE_OCR_SERVER_URL`。

## 启动 vLLM / Qwen3-32B-AWQ 推理服务

vLLM 用于关系抽取阶段的 LLM 推理。本项目把 vLLM 安装在独立的虚拟环境 `~/venvs/vllm-qwen` 中（与后端 `.venv` 隔离，避免依赖冲突），从该环境直接 `vllm serve` 启动 OpenAI 兼容服务，监听 8000 端口：

```bash
# 激活独立 venv
source ~/venvs/vllm-qwen/bin/activate

# 在 GPU 0 上加载 Qwen3-32B-AWQ
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

启动后可用以下命令验证：

```bash
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer EMPTY"
```

参数说明：

- `--api-key EMPTY` 与 `backend/.env` 的 `VLLM_API_KEY=EMPTY` 对应
- `--default-chat-template-kwargs '{"enable_thinking": false}'` 默认关闭 Qwen3 思考模式；后端可通过 `VLLM_ENABLE_THINKING=true` 在每次请求中按需开启（详见 [vllm_client.py](backend/app/vllm_client.py)）
- 显存不足可改用更小的 Qwen 变体（如 `Qwen/Qwen3-8B`）并同步更新 `backend/.env` 的 `VLLM_MODEL`，或多卡时通过 `CUDA_VISIBLE_DEVICES=0,1` 配合 `--tensor-parallel-size 2` 分布式加载

## 后端启动

1. 创建虚拟环境并安装依赖

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

说明：

- [requirements.txt](backend/requirements.txt) 面向 CUDA 13.0 主机，包含 API/Worker 依赖以及 PaddlePaddle GPU 运行时
- `vllm` 不在该 requirements 中，请在独立环境运行（见上一节）

2. 配置环境变量

```bash
cp .env.example .env
# 按需修改 .env：Redis 地址、PaddleOCR-VL / vLLM 服务地址、模型名等
```

关键变量（完整列表见 [config.py](backend/app/config.py)）：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `5000` | Flask API 监听地址 |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis 任务队列与状态存储 |
| `STORAGE_ROOT` | `../data` | 上传文件与结果落盘根目录 |
| `PADDLE_OCR_MODE` | `python_api` | OCR 调用模式（推荐保持 `python_api`） |
| `PADDLE_OCR_SERVER_URL` | `http://127.0.0.1:8118/v1` | `genai_server` 地址 |
| `PADDLE_OCR_API_MODEL_NAME` | `PaddleOCR-VL-1.5-0.9B` | VL 识别模型名 |
| `VLLM_BASE_URL` | `http://127.0.0.1:8000/v1` | vLLM OpenAI 兼容地址 |
| `VLLM_MODEL` | `Qwen/Qwen3-32B-AWQ` | 关系抽取使用的模型名 |
| `VLLM_ENABLE_THINKING` | `false` | 是否开启 Qwen 思考模式 |
| `OCR_CONCURRENCY` / `LLM_CONCURRENCY` | `1` / `4` | OCR 与 LLM 并发度 |
| `MAX_CHUNK_CHARS` | `2400` | 多页分块字符上限 |
| `SKILL4RE_BACKEND` | `vllm` | Skill4RE 推理后端，默认调用本项目的 vLLM OpenAI 兼容服务 |
| `SKILL4RE_MODEL` | 同 `VLLM_MODEL` | Skill4RE 抽取模型名 |
| `SKILL4RE_SKILLS_DIR` | `skill4re/skill4re/skills` | 技能定义 JSON 目录 |
| `SKILL4RE_CHUNK_TRIGGER` / `SKILL4RE_CHUNK_BUDGET` | `1200` / `900` | Skill4RE 长文档分块阈值与预算 |
| `SKILL4RE_FAST_MODE` / `SKILL4RE_SKIP_COREF` | `false` / `false` | 是否跳过 LLM 语义合并 / 是否跳过长文档共指消解 |

3. 启动 API（开启另一个终端）

```bash
cd backend
source .venv/bin/activate
python run_api.py
```

API 启动后可通过健康检查确认依赖是否就绪：

```bash
curl http://127.0.0.1:5000/api/health
# {"status": "ok", "redis": true, "paddle_ocr": true, "vllm": true}
```

4. 启动 Worker（再开启一个终端）

```bash
cd backend
source .venv/bin/activate
python run_worker.py
```

Worker 会持续从 Redis 拉取任务并调用 [DocumentPipeline](backend/app/pipeline.py)：版面解析 → 结构化重排 → Skill4RE 技能路由关系抽取 → `relation_list` JSON 输出。Worker 会在每次抽取前检测 `SKILL4RE_SKILLS_DIR` 下的 skill JSON 是否变更，前端新增或修改 skill 后，后续任务会自动使用最新配置。

## API 速览

所有接口位于 [api.py](backend/app/api.py)：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/upload` | 多文件上传，表单字段 `files`，返回创建的任务列表 |
| `GET` | `/api/status/<task_id>` | 查询任务状态、阶段、进度 |
| `GET` | `/api/result/<task_id>` | 任务完成后返回 OCR / 抽取完整结果，最终关系位于 `final_relations` 与 `final_relation_list.relation_list` |
| `GET` | `/api/health` | Redis / PaddleOCR / vLLM 依赖健康检查 |
| `GET` | `/api/skills` | 扫描并返回当前 Skill4RE skills |
| `GET` | `/api/skills/<name>` | 读取单个 skill |
| `POST` | `/api/skills` | 新增 skill，写入 `<name>.json` |
| `PUT` | `/api/skills/<name>` | 修改 skill，若 `name` 变更会重命名 JSON 文件 |

## 前端启动

```bash
cd frontend
npm install
npm run dev
```

默认开发服务器运行在 [http://localhost:5173](http://localhost:5173)。前端无需任何 `.env` 文件即可工作：[api.js](frontend/src/lib/api.js) 把 `API_BASE_URL` 默认设为 `/api`，[vite.config.js](frontend/vite.config.js) 又把 `/api` 代理到 `http://127.0.0.1:5000`。

如需指向远程后端，临时通过环境变量覆盖即可，无需新建 `.env`：

```bash
VITE_API_BASE_URL=http://your-host:5000/api npm run dev
```

构建产物：

```bash
npm run build      # 生成 dist/
npm run preview    # 预览构建结果
```

## 推荐启动顺序

1. Redis 容器：`sudo docker start docre-redis`（首次创建用上文 `docker run` 命令）
2. PaddleOCR-VL `genai_server` 容器（监听 8118）
3. vLLM 推理服务（`source ~/venvs/vllm-qwen/bin/activate` 后 `vllm serve ...`，监听 8000）
4. 后端 API：`python run_api.py`
5. 后端 Worker：`python run_worker.py`
6. 前端：`npm run dev`

启动完成后访问 [http://localhost:5173](http://localhost:5173)，上传 PDF / 图片即可在右侧面板查看 Skill4RE 路由、分块与最终 JSON 抽取结果。
