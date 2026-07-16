#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/backend"
PYTHON="$BACKEND_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "未找到后端虚拟环境：$PYTHON" >&2
  echo "请先按 README 创建 backend/.venv 并安装依赖。" >&2
  exit 1
fi

if [[ ! -f "$BACKEND_DIR/.env" ]]; then
  echo "提示：未找到 backend/.env，将使用代码默认配置。" >&2
  echo "可执行：cp backend/.env.example backend/.env" >&2
fi

cd "$BACKEND_DIR"

"$PYTHON" run_worker.py &
worker_pid=$!

cleanup() {
  if kill -0 "$worker_pid" 2>/dev/null; then
    kill "$worker_pid" 2>/dev/null || true
    wait "$worker_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Worker 已启动（PID: $worker_pid）"
echo "启动后端 API；按 Ctrl+C 将同时停止 API 和 Worker。"
"$PYTHON" run_api.py
