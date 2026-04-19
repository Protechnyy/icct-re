from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)

    LOGGER.debug("Loaded environment variables from %s", env_path)


@dataclass(frozen=True)
class AppConfig:
    api_host: str
    api_port: int
    redis_url: str
    storage_root: Path
    paddle_ocr_mode: str
    paddle_ocr_base_url: str
    paddle_ocr_server_url: str
    paddle_ocr_api_model_name: str
    paddle_ocr_api_key: str
    paddle_ocr_timeout_seconds: int
    paddle_ocr_file_mode: str
    vllm_base_url: str
    vllm_api_key: str
    vllm_model: str
    vllm_timeout_seconds: int
    vllm_enable_thinking: bool
    ocr_concurrency: int
    llm_concurrency: int
    max_chunk_chars: int
    sentence_stage_limit: int
    page_stage_limit: int
    worker_poll_seconds: int
    debug: bool

    @classmethod
    def from_env(cls) -> "AppConfig":
        _load_dotenv()
        storage_root = Path(os.getenv("STORAGE_ROOT", "../data")).expanduser().resolve()
        return cls(
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "5000")),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            storage_root=storage_root,
            paddle_ocr_mode=os.getenv("PADDLE_OCR_MODE", "python_api"),
            paddle_ocr_base_url=os.getenv("PADDLE_OCR_BASE_URL", "http://127.0.0.1:8080").rstrip("/"),
            paddle_ocr_server_url=os.getenv("PADDLE_OCR_SERVER_URL", "http://127.0.0.1:8118/v1").rstrip("/"),
            paddle_ocr_api_model_name=os.getenv("PADDLE_OCR_API_MODEL_NAME", "PaddlePaddle/PaddleOCR-VL-1.5"),
            paddle_ocr_api_key=os.getenv("PADDLE_OCR_API_KEY", "EMPTY"),
            paddle_ocr_timeout_seconds=int(os.getenv("PADDLE_OCR_TIMEOUT_SECONDS", "180")),
            paddle_ocr_file_mode=os.getenv("PADDLE_OCR_FILE_MODE", "base64"),
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/"),
            vllm_api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
            vllm_model=os.getenv("VLLM_MODEL", "qwen"),
            vllm_timeout_seconds=int(os.getenv("VLLM_TIMEOUT_SECONDS", "180")),
            vllm_enable_thinking=_as_bool(os.getenv("VLLM_ENABLE_THINKING"), False),
            ocr_concurrency=int(os.getenv("OCR_CONCURRENCY", "1")),
            llm_concurrency=int(os.getenv("LLM_CONCURRENCY", "4")),
            max_chunk_chars=int(os.getenv("MAX_CHUNK_CHARS", "2400")),
            sentence_stage_limit=int(os.getenv("SENTENCE_STAGE_LIMIT", "12")),
            page_stage_limit=int(os.getenv("PAGE_STAGE_LIMIT", "12")),
            worker_poll_seconds=int(os.getenv("WORKER_POLL_SECONDS", "2")),
            debug=_as_bool(os.getenv("FLASK_ENV"), False),
        )

    def ensure_storage_dirs(self) -> None:
        for relative in ("uploads", "results", "tmp"):
            (self.storage_root / relative).mkdir(parents=True, exist_ok=True)

    def safe_summary(self) -> dict[str, str | int | bool]:
        return {
            "api_host": self.api_host,
            "api_port": self.api_port,
            "redis_url": self.redis_url,
            "storage_root": str(self.storage_root),
            "paddle_ocr_mode": self.paddle_ocr_mode,
            "paddle_ocr_base_url": self.paddle_ocr_base_url,
            "paddle_ocr_server_url": self.paddle_ocr_server_url,
            "vllm_base_url": self.vllm_base_url,
            "vllm_model": self.vllm_model,
            "vllm_enable_thinking": self.vllm_enable_thinking,
            "ocr_concurrency": self.ocr_concurrency,
            "llm_concurrency": self.llm_concurrency,
        }
