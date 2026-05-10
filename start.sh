#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Override any value at launch, for example:
#   API_PORT=5012 FRONTEND_PORT=5174 ./start.sh
#   PADDLE_OCR_SERVER_URL=http://host:8118/v1 ./start.sh
#   VLLM_BASE_URL=http://host:8000/v1 VLLM_MODEL=Qwen/Qwen3-8B ./start.sh

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/backend/.venv/bin/python" ]]; then
  PYTHON_CMD="$ROOT_DIR/backend/.venv/bin/python"
else
  PYTHON_CMD="python3"
fi

API_PORT="${API_PORT:-5000}"
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
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-icct-re}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"

export API_PORT
export PADDLE_OCR_MODE="${PADDLE_OCR_MODE:-python_api}"
export PADDLE_OCR_BASE_URL="${PADDLE_OCR_BASE_URL:-http://127.0.0.1:8118}"
export PADDLE_OCR_SERVER_URL="${PADDLE_OCR_SERVER_URL:-http://127.0.0.1:8118/v1}"
export PADDLE_OCR_API_MODEL_NAME="${PADDLE_OCR_API_MODEL_NAME:-PaddleOCR-VL-1.5-0.9B}"
export PADDLE_OCR_API_KEY="${PADDLE_OCR_API_KEY:-EMPTY}"
export PADDLE_OCR_MAX_CONCURRENCY="${PADDLE_OCR_MAX_CONCURRENCY:-4}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-32B-AWQ}"
export VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING:-false}"
export SKILL4RE_MODEL="${SKILL4RE_MODEL:-$VLLM_MODEL}"
export RELATION_SPLIT_MODE="${RELATION_SPLIT_MODE:-small_section}"
export RELATION_BATCH_SIZE="${RELATION_BATCH_SIZE:-1}"
export RELATION_MAX_BATCH_TOKENS="${RELATION_MAX_BATCH_TOKENS:-2500}"
export RELATION_INCLUDE_PARENT_TITLE="${RELATION_INCLUDE_PARENT_TITLE:-true}"

cd "$ROOT_DIR"

"$PYTHON_CMD" backend/run_api.py > "$TMP_DIR/api-$API_PORT.log" 2>&1 &
echo "$!" > "$PID_DIR/api.pid"

"$PYTHON_CMD" backend/run_worker.py > "$TMP_DIR/worker.log" 2>&1 &
echo "$!" > "$PID_DIR/worker.pid"

cd "$ROOT_DIR/frontend"
if [[ ! -x node_modules/.bin/vite ]]; then
  npm install
fi
VITE_API_BASE_URL="http://127.0.0.1:$API_PORT/api" \
  npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT" > "$TMP_DIR/frontend-$FRONTEND_PORT.log" 2>&1 &
echo "$!" > "$PID_DIR/frontend.pid"

echo "API: http://127.0.0.1:$API_PORT"
echo "Frontend: http://127.0.0.1:$FRONTEND_PORT"
echo "Logs:"
echo "  $TMP_DIR/api-$API_PORT.log"
echo "  $TMP_DIR/worker.log"
echo "  $TMP_DIR/frontend-$FRONTEND_PORT.log"
echo "Stop:"
echo "  $ROOT_DIR/start.sh stop"
