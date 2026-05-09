#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================
# 可改配置
# =============================
#
# 后端 Python 环境：
#   PYTHON_BIN=/path/to/python ./start.sh
#
# 页面/API 端口：
#   API_PORT=5012 FRONTEND_PORT=5174 ./start.sh
#
# OCR 远程服务和模型：
#   PADDLE_OCR_SERVER_URL=http://host:port/v1 \
#   PADDLE_OCR_API_MODEL_NAME=PaddleOCR-VL-1.5-0.9B \
#   ./start.sh
#
# 关系抽取 LLM 服务和模型：
#   VLLM_BASE_URL=http://host:port/v1 \
#   VLLM_MODEL=Qwen3-8B \
#   SKILL4RE_MODEL=Qwen3-8B \
#   ./start.sh
#
# 关系抽取每批 section 数：
#   RELATION_SECTION_BATCH_SIZE=1 ./start.sh
#
# 如果外网下载慢，默认会使用 127.0.0.1:7890 代理；如不需要：
#   HTTP_PROXY= HTTPS_PROXY= ./start.sh

PYTHON_BIN="${PYTHON_BIN:-/home/ybxy/下载/enter/envs/icct-re/bin/python}"
API_PORT="${API_PORT:-5011}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
STORAGE_ROOT="${STORAGE_ROOT:-$ROOT_DIR/data}"
TMP_DIR="$STORAGE_ROOT/tmp"
PID_DIR="$TMP_DIR/pids"

mkdir -p "$TMP_DIR" "$PID_DIR"

stop_pid() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
}

if [[ "${1:-}" == "stop" ]]; then
  stop_pid "$PID_DIR/api.pid"
  stop_pid "$PID_DIR/worker.pid"
  stop_pid "$PID_DIR/frontend.pid"
  echo "Stopped icct-re services."
  exit 0
fi

stop_pid "$PID_DIR/api.pid"
stop_pid "$PID_DIR/worker.pid"
stop_pid "$PID_DIR/frontend.pid"

export PYTHONPATH="$ROOT_DIR/backend"
export STORAGE_ROOT
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7890}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,192.168.1.2}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-icct-re}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"

export PADDLE_OCR_MODE="${PADDLE_OCR_MODE:-python_api}"
export PADDLE_OCR_BASE_URL="${PADDLE_OCR_BASE_URL:-http://192.168.1.2:8080}"
export PADDLE_OCR_SERVER_URL="${PADDLE_OCR_SERVER_URL:-http://192.168.1.2:8080/v1}"
export PADDLE_OCR_API_MODEL_NAME="${PADDLE_OCR_API_MODEL_NAME:-PaddleOCR-VL-1.5-0.9B}"
export PADDLE_OCR_API_KEY="${PADDLE_OCR_API_KEY:-EMPTY}"
export PADDLE_OCR_MAX_CONCURRENCY="${PADDLE_OCR_MAX_CONCURRENCY:-32}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://192.168.1.2:9000/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen3-8B}"
export VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING:-false}"
export SKILL4RE_MODEL="${SKILL4RE_MODEL:-Qwen3-8B}"
export RELATION_SECTION_BATCH_SIZE="${RELATION_SECTION_BATCH_SIZE:-2}"

cd "$ROOT_DIR"

API_PORT="$API_PORT" "$PYTHON_BIN" backend/run_api.py > "$TMP_DIR/api-$API_PORT.log" 2>&1 &
echo "$!" > "$PID_DIR/api.pid"

"$PYTHON_BIN" backend/run_worker.py > "$TMP_DIR/worker.log" 2>&1 &
echo "$!" > "$PID_DIR/worker.pid"

echo "API: http://127.0.0.1:$API_PORT"
echo "Frontend: http://127.0.0.1:$FRONTEND_PORT"
echo "Logs:"
echo "  $TMP_DIR/api-$API_PORT.log"
echo "  $TMP_DIR/worker.log"
echo "Stop:"
echo "  $ROOT_DIR/start.sh stop"

cd "$ROOT_DIR/frontend"
if [[ ! -x node_modules/.bin/vite ]]; then
  npm install
fi
VITE_API_BASE_URL="http://127.0.0.1:$API_PORT/api" npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT" > "$TMP_DIR/frontend-$FRONTEND_PORT.log" 2>&1 &
echo "$!" > "$PID_DIR/frontend.pid"

echo "Frontend log:"
echo "  $TMP_DIR/frontend-$FRONTEND_PORT.log"
echo
echo "Services started. Press Ctrl+C to leave this script; services keep running in background."
