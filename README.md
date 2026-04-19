# Document-Level RE MVP

本项目实现一个文档级关系抽取 MVP：

- 本地 Flask API 负责上传、任务状态、结果查询
- 本地 Worker 通过 PaddleOCR-VL 官方 Python API 调用本机 `genai_server`
- 本地或远程 vLLM/Qwen 提供 OpenAI 兼容推理
- React + Vite + Ant Design 前端提供批量上传和结果查看

## 目录

- `backend/`: Flask API、Redis 任务存储、OCR/LLM 流水线、worker
- `frontend/`: React 单页工作台

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

- `requirements.txt` 面向当前这台 CUDA 12.6 主机，包含后端 API/worker 依赖和 PaddlePaddle GPU 运行时
- `vllm` 建议放在单独环境或单独服务中启动，不要与后端 `.venv` 混装

2. 配置环境变量

```bash
cp .env.example .env
```

3. 启动 API

```bash
python run_api.py
```

4. 启动 Worker

```bash
python run_worker.py
```

## 前端启动

```bash
cd frontend
npm install
npm run dev
```

默认前端通过 `VITE_API_BASE_URL` 调用 `http://localhost:5000/api`。
