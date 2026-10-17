from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "qwen3.8-27b"
DEFAULT_LLM_BASE_URL = "https://llm-bln22h7lns8wvuub.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

RELATION_SPLIT_MODES = {"small_section", "chapter", "paragraph", "fixed_sections"}
DEFAULT_RELATION_SPLIT_MODE = "small_section"
DEFAULT_RELATION_BATCH_SIZE = 1
DEFAULT_RELATION_CHAPTER_BATCH_SIZE = 2
DEFAULT_RELATION_PARAGRAPH_BATCH_SIZE = 5
DEFAULT_RELATION_FIXED_SECTION_BATCH_SIZE = DEFAULT_RELATION_BATCH_SIZE
DEFAULT_RELATION_MAX_BATCH_TOKENS = 2500
DEFAULT_RELATION_INCLUDE_PARENT_TITLE = True
DEFAULT_RELATION_BATCH_CONCURRENCY = 10
DEFAULT_PADDLE_OCR_MAX_CONCURRENCY = 4
DEFAULT_PADDLE_OCR_MODE = "python_api"
DEFAULT_PADDLE_OCR_BASE_URL = "http://127.0.0.1:8118"
DEFAULT_PADDLE_OCR_SERVER_URL = "http://127.0.0.1:8118/v1"

RETIRED_MODEL_VARIABLES = {
    "SKILL4RE_BACKEND": "LLM_BACKEND",
    "SKILL4RE_BASE_URL": "LLM_BASE_URL",
    "SKILL4RE_MODEL": "LLM_MODEL",
    "VLLM_BASE_URL": "LLM_BASE_URL",
    "VLLM_API_KEY": "LLM_API_KEY",
    "VLLM_MODEL": "LLM_MODEL",
    "VLLM_TIMEOUT_SECONDS": "LLM_REQUEST_TIMEOUT_SECONDS",
    "VLLM_ENABLE_THINKING": "LLM_ENABLE_THINKING",
    "VLLM_MAX_RETRIES": "LLM_MAX_RETRIES",
    "VLLM_RETRY_BACKOFF_SECONDS": "LLM_RETRY_BACKOFF_SECONDS",
    "SKILL4RE_API_MAX_RETRIES": "LLM_MAX_RETRIES",
    "SKILL4RE_API_RETRY_BACKOFF_SECONDS": "LLM_RETRY_BACKOFF_SECONDS",
    "AGENT_BASE_URL": "LLM_BASE_URL",
    "AGENT_API_KEY": "LLM_API_KEY",
    "AGENT_MODEL": "LLM_MODEL",
    **{f"AGENT_{name}": f"VERIFICATION_AGENT_{name}" for name in (
        "ENABLED", "CONCURRENCY", "MAX_TASKS", "MAX_STEPS", "MAX_LLM_CALLS",
        "TIMEOUT_SECONDS", "MAX_ADDED_PER_TASK",
    )},
}


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} 必须是正整数")
    return value


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


def normalize_openai_base_url(value: str) -> str:
    url = value.strip().rstrip("/")
    suffix = "/chat/completions"
    if url.endswith(suffix):
        return url[: -len(suffix)].rstrip("/")
    return url


def _load_dotenv(env_path: Path | None = None) -> None:
    env_path = env_path or Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=False)
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
    paddle_ocr_api_key: str = field(repr=False)
    paddle_ocr_timeout_seconds: int
    paddle_ocr_file_mode: str
    llm_base_url: str
    llm_api_key: str = field(repr=False)
    llm_model: str
    llm_request_timeout_seconds: int
    llm_enable_thinking: bool
    ocr_concurrency: int
    llm_concurrency: int
    max_chunk_chars: int
    sentence_stage_limit: int
    page_stage_limit: int
    llm_backend: str
    skill4re_skills_dir: Path
    skill4re_route_cache_path: Path
    skill4re_chunk_trigger: int
    skill4re_chunk_budget: int
    skill4re_max_workers: int
    skill4re_fast_mode: bool
    skill4re_skip_coref: bool
    worker_poll_seconds: int
    debug: bool
    paddle_ocr_max_concurrency: int = DEFAULT_PADDLE_OCR_MAX_CONCURRENCY
    relation_split_mode: str = DEFAULT_RELATION_SPLIT_MODE
    relation_batch_size: int = DEFAULT_RELATION_BATCH_SIZE
    relation_chapter_batch_size: int = DEFAULT_RELATION_CHAPTER_BATCH_SIZE
    relation_paragraph_batch_size: int = DEFAULT_RELATION_PARAGRAPH_BATCH_SIZE
    relation_fixed_section_batch_size: int = DEFAULT_RELATION_FIXED_SECTION_BATCH_SIZE
    relation_max_batch_tokens: int = DEFAULT_RELATION_MAX_BATCH_TOKENS
    relation_include_parent_title: bool = DEFAULT_RELATION_INCLUDE_PARENT_TITLE
    relation_batch_concurrency: int = DEFAULT_RELATION_BATCH_CONCURRENCY
    verification_agent_enabled: bool = False
    verification_agent_concurrency: int = 1
    verification_agent_max_tasks: int = 20
    verification_agent_max_steps: int = 6
    verification_agent_max_llm_calls: int = 80
    verification_agent_timeout_seconds: int = 600
    verification_agent_max_added_per_task: int = 5
    llm_max_retries: int = 3
    llm_retry_backoff_seconds: float = 2.0
    stream_enabled: bool = True
    stream_thinking: bool = True

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "AppConfig":
        _load_dotenv(env_path)
        retired = [f"{old} → {new}" for old, new in RETIRED_MODEL_VARIABLES.items() if old in os.environ]
        if retired:
            raise ValueError("请更新模型配置名称：" + "；".join(retired))
        storage_root = Path(os.getenv("STORAGE_ROOT", "../data")).expanduser().resolve()
        repo_root = Path(__file__).resolve().parents[2]
        default_skill4re_skills_dir = repo_root / "skill4re" / "skill4re" / "skills"
        default_skill4re_route_cache = storage_root / "tmp" / "skill4re_route_cache.json"
        retry_backoff = float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", "2"))
        if not math.isfinite(retry_backoff) or retry_backoff < 0:
            raise ValueError("LLM_RETRY_BACKOFF_SECONDS 必须是有限的非负数")
        return cls(
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "5000")),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            storage_root=storage_root,
            paddle_ocr_mode=os.getenv("PADDLE_OCR_MODE", DEFAULT_PADDLE_OCR_MODE),
            paddle_ocr_base_url=os.getenv("PADDLE_OCR_BASE_URL", DEFAULT_PADDLE_OCR_BASE_URL).rstrip("/"),
            paddle_ocr_server_url=os.getenv("PADDLE_OCR_SERVER_URL", DEFAULT_PADDLE_OCR_SERVER_URL).rstrip("/"),
            paddle_ocr_api_model_name=os.getenv("PADDLE_OCR_API_MODEL_NAME", "PaddleOCR-VL-1.5-0.9B"),
            paddle_ocr_api_key=os.getenv("PADDLE_OCR_API_KEY", "EMPTY"),
            paddle_ocr_max_concurrency=_as_int(
                os.getenv("PADDLE_OCR_MAX_CONCURRENCY"),
                DEFAULT_PADDLE_OCR_MAX_CONCURRENCY,
            ),
            paddle_ocr_timeout_seconds=int(os.getenv("PADDLE_OCR_TIMEOUT_SECONDS", "180")),
            paddle_ocr_file_mode=os.getenv("PADDLE_OCR_FILE_MODE", "base64"),
            llm_base_url=normalize_openai_base_url(os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL).strip(),
            llm_request_timeout_seconds=_positive_int("LLM_REQUEST_TIMEOUT_SECONDS", 180),
            llm_enable_thinking=_as_bool(os.getenv("LLM_ENABLE_THINKING"), False),
            llm_max_retries=_positive_int("LLM_MAX_RETRIES", 3),
            llm_retry_backoff_seconds=retry_backoff,
            ocr_concurrency=int(os.getenv("OCR_CONCURRENCY", "1")),
            llm_concurrency=int(os.getenv("LLM_CONCURRENCY", "4")),
            max_chunk_chars=int(os.getenv("MAX_CHUNK_CHARS", "2400")),
            sentence_stage_limit=int(os.getenv("SENTENCE_STAGE_LIMIT", "12")),
            page_stage_limit=int(os.getenv("PAGE_STAGE_LIMIT", "12")),
            llm_backend=os.getenv("LLM_BACKEND", "vllm").strip(),
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
            relation_chapter_batch_size=_as_int(
                os.getenv("RELATION_CHAPTER_BATCH_SIZE"),
                DEFAULT_RELATION_CHAPTER_BATCH_SIZE,
            ),
            relation_paragraph_batch_size=_as_int(
                os.getenv("RELATION_PARAGRAPH_BATCH_SIZE"),
                DEFAULT_RELATION_PARAGRAPH_BATCH_SIZE,
            ),
            relation_fixed_section_batch_size=_as_int(
                os.getenv(
                    "RELATION_FIXED_SECTION_BATCH_SIZE",
                    os.getenv("RELATION_BATCH_SIZE", os.getenv("RELATION_SECTION_BATCH_SIZE")),
                ),
                DEFAULT_RELATION_FIXED_SECTION_BATCH_SIZE,
            ),
            relation_max_batch_tokens=_as_int(
                os.getenv("RELATION_MAX_BATCH_TOKENS"),
                DEFAULT_RELATION_MAX_BATCH_TOKENS,
            ),
            relation_include_parent_title=_as_bool(
                os.getenv("RELATION_INCLUDE_PARENT_TITLE"),
                DEFAULT_RELATION_INCLUDE_PARENT_TITLE,
            ),
            relation_batch_concurrency=_as_int(
                os.getenv("RELATION_BATCH_CONCURRENCY"),
                DEFAULT_RELATION_BATCH_CONCURRENCY,
            ),
            verification_agent_enabled=_as_bool(os.getenv("VERIFICATION_AGENT_ENABLED"), False),
            verification_agent_concurrency=_positive_int("VERIFICATION_AGENT_CONCURRENCY", 1),
            verification_agent_max_tasks=_positive_int("VERIFICATION_AGENT_MAX_TASKS", 20),
            verification_agent_max_steps=_positive_int("VERIFICATION_AGENT_MAX_STEPS", 6),
            verification_agent_max_llm_calls=_positive_int("VERIFICATION_AGENT_MAX_LLM_CALLS", 80),
            verification_agent_timeout_seconds=_positive_int("VERIFICATION_AGENT_TIMEOUT_SECONDS", 600),
            verification_agent_max_added_per_task=_positive_int("VERIFICATION_AGENT_MAX_ADDED_PER_TASK", 5),
            stream_enabled=_as_bool(os.getenv("STREAM_ENABLED"), True),
            stream_thinking=_as_bool(os.getenv("STREAM_THINKING"), True),
        )

    def ensure_storage_dirs(self) -> None:
        for relative in ("uploads", "results", "tmp"):
            (self.storage_root / relative).mkdir(parents=True, exist_ok=True)

    def safe_summary(self) -> dict[str, str | int | float | bool]:
        return {
            "api_host": self.api_host,
            "api_port": self.api_port,
            "redis_url": self.redis_url,
            "storage_root": str(self.storage_root),
            "paddle_ocr_mode": self.paddle_ocr_mode,
            "paddle_ocr_base_url": self.paddle_ocr_base_url,
            "paddle_ocr_server_url": self.paddle_ocr_server_url,
            "paddle_ocr_max_concurrency": self.paddle_ocr_max_concurrency,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_backend": self.llm_backend,
            "llm_request_timeout_seconds": self.llm_request_timeout_seconds,
            "llm_enable_thinking": self.llm_enable_thinking,
            "llm_max_retries": self.llm_max_retries,
            "llm_retry_backoff_seconds": self.llm_retry_backoff_seconds,
            "ocr_concurrency": self.ocr_concurrency,
            "llm_concurrency": self.llm_concurrency,
            "skill4re_skills_dir": str(self.skill4re_skills_dir),
            "skill4re_chunk_trigger": self.skill4re_chunk_trigger,
            "skill4re_chunk_budget": self.skill4re_chunk_budget,
            "skill4re_max_workers": self.skill4re_max_workers,
            "skill4re_fast_mode": self.skill4re_fast_mode,
            "skill4re_skip_coref": self.skill4re_skip_coref,
            "relation_split_mode": self.relation_split_mode,
            "relation_batch_size": self.relation_batch_size,
            "relation_chapter_batch_size": self.relation_chapter_batch_size,
            "relation_paragraph_batch_size": self.relation_paragraph_batch_size,
            "relation_fixed_section_batch_size": self.relation_fixed_section_batch_size,
            "relation_max_batch_tokens": self.relation_max_batch_tokens,
            "relation_include_parent_title": self.relation_include_parent_title,
            "relation_batch_concurrency": self.relation_batch_concurrency,
            "verification_agent_enabled": self.verification_agent_enabled,
            "verification_agent_concurrency": self.verification_agent_concurrency,
            "verification_agent_max_tasks": self.verification_agent_max_tasks,
            "verification_agent_max_steps": self.verification_agent_max_steps,
            "verification_agent_max_llm_calls": self.verification_agent_max_llm_calls,
            "verification_agent_timeout_seconds": self.verification_agent_timeout_seconds,
            "verification_agent_max_added_per_task": self.verification_agent_max_added_per_task,
            "stream_enabled": self.stream_enabled,
            "stream_thinking": self.stream_thinking,
        }
