from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

RELATION_SPLIT_MODES = {"small_section", "chapter", "paragraph", "fixed_sections"}
DEFAULT_RELATION_SPLIT_MODE = "small_section"
DEFAULT_RELATION_BATCH_SIZE = 1
DEFAULT_RELATION_MAX_BATCH_TOKENS = 2500
DEFAULT_RELATION_INCLUDE_PARENT_TITLE = True


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _as_relation_split_mode(value: str | None, default: str = DEFAULT_RELATION_SPLIT_MODE) -> str:
    mode = str(value or default).strip()
    return mode if mode in RELATION_SPLIT_MODES else default


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
    skill4re_backend: str
    skill4re_model: str
    skill4re_skills_dir: Path
    skill4re_route_cache_path: Path
    skill4re_chunk_trigger: int
    skill4re_chunk_budget: int
    skill4re_max_workers: int
    skill4re_fast_mode: bool
    skill4re_skip_coref: bool
    worker_poll_seconds: int
    debug: bool
    relation_split_mode: str = DEFAULT_RELATION_SPLIT_MODE
    relation_batch_size: int = DEFAULT_RELATION_BATCH_SIZE
    relation_max_batch_tokens: int = DEFAULT_RELATION_MAX_BATCH_TOKENS
    relation_include_parent_title: bool = DEFAULT_RELATION_INCLUDE_PARENT_TITLE

    @classmethod
    def from_env(cls) -> "AppConfig":
        _load_dotenv()
        storage_root = Path(os.getenv("STORAGE_ROOT", "../data")).expanduser().resolve()
        repo_root = Path(__file__).resolve().parents[2]
        default_skill4re_skills_dir = repo_root / "skill4re" / "skill4re" / "skills"
        default_skill4re_route_cache = storage_root / "tmp" / "skill4re_route_cache.json"
        vllm_model = os.getenv("VLLM_MODEL", "Qwen3-8B")
        return cls(
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "5000")),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            storage_root=storage_root,
            paddle_ocr_mode=os.getenv("PADDLE_OCR_MODE", "python_api"),
            paddle_ocr_base_url=os.getenv("PADDLE_OCR_BASE_URL", "http://192.168.1.2:8080").rstrip("/"),
            paddle_ocr_server_url=os.getenv("PADDLE_OCR_SERVER_URL", "http://192.168.1.2:8080/v1").rstrip("/"),
            paddle_ocr_api_model_name=os.getenv("PADDLE_OCR_API_MODEL_NAME", "PaddleOCR-VL-1.5-0.9B"),
            paddle_ocr_api_key=os.getenv("PADDLE_OCR_API_KEY", "EMPTY"),
            paddle_ocr_timeout_seconds=int(os.getenv("PADDLE_OCR_TIMEOUT_SECONDS", "180")),
            paddle_ocr_file_mode=os.getenv("PADDLE_OCR_FILE_MODE", "base64"),
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://192.168.1.2:9000/v1").rstrip("/"),
            vllm_api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
            vllm_model=vllm_model,
            vllm_timeout_seconds=int(os.getenv("VLLM_TIMEOUT_SECONDS", "180")),
            vllm_enable_thinking=_as_bool(os.getenv("VLLM_ENABLE_THINKING"), False),
            ocr_concurrency=int(os.getenv("OCR_CONCURRENCY", "1")),
            llm_concurrency=int(os.getenv("LLM_CONCURRENCY", "4")),
            max_chunk_chars=int(os.getenv("MAX_CHUNK_CHARS", "2400")),
            sentence_stage_limit=int(os.getenv("SENTENCE_STAGE_LIMIT", "12")),
            page_stage_limit=int(os.getenv("PAGE_STAGE_LIMIT", "12")),
            skill4re_backend=os.getenv("SKILL4RE_BACKEND", "vllm"),
            skill4re_model=os.getenv("SKILL4RE_MODEL", vllm_model),
            skill4re_skills_dir=Path(
                os.getenv("SKILL4RE_SKILLS_DIR", str(default_skill4re_skills_dir))
            ).expanduser().resolve(),
            skill4re_route_cache_path=Path(
                os.getenv("SKILL4RE_ROUTE_CACHE_PATH", str(default_skill4re_route_cache))
            ).expanduser().resolve(),
            skill4re_chunk_trigger=int(os.getenv("SKILL4RE_CHUNK_TRIGGER", "1200")),
            skill4re_chunk_budget=int(os.getenv("SKILL4RE_CHUNK_BUDGET", "900")),
            skill4re_max_workers=int(os.getenv("SKILL4RE_MAX_WORKERS", os.getenv("LLM_CONCURRENCY", "4"))),
            skill4re_fast_mode=_as_bool(os.getenv("SKILL4RE_FAST_MODE"), False),
            skill4re_skip_coref=_as_bool(os.getenv("SKILL4RE_SKIP_COREF"), False),
            worker_poll_seconds=int(os.getenv("WORKER_POLL_SECONDS", "2")),
            debug=_as_bool(os.getenv("FLASK_ENV"), False),
            relation_split_mode=_as_relation_split_mode(os.getenv("RELATION_SPLIT_MODE")),
            relation_batch_size=_as_int(
                os.getenv("RELATION_BATCH_SIZE", os.getenv("RELATION_SECTION_BATCH_SIZE")),
                DEFAULT_RELATION_BATCH_SIZE,
            ),
            relation_max_batch_tokens=_as_int(
                os.getenv("RELATION_MAX_BATCH_TOKENS"),
                DEFAULT_RELATION_MAX_BATCH_TOKENS,
            ),
            relation_include_parent_title=_as_bool(
                os.getenv("RELATION_INCLUDE_PARENT_TITLE"),
                DEFAULT_RELATION_INCLUDE_PARENT_TITLE,
            ),
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
            "skill4re_backend": self.skill4re_backend,
            "skill4re_model": self.skill4re_model,
            "skill4re_skills_dir": str(self.skill4re_skills_dir),
            "skill4re_chunk_trigger": self.skill4re_chunk_trigger,
            "skill4re_chunk_budget": self.skill4re_chunk_budget,
            "skill4re_max_workers": self.skill4re_max_workers,
            "skill4re_fast_mode": self.skill4re_fast_mode,
            "skill4re_skip_coref": self.skill4re_skip_coref,
            "relation_split_mode": self.relation_split_mode,
            "relation_batch_size": self.relation_batch_size,
            "relation_max_batch_tokens": self.relation_max_batch_tokens,
            "relation_include_parent_title": self.relation_include_parent_title,
        }
