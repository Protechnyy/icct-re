from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from .config import AppConfig
from .model_settings import LlmRequestSettings

LOGGER = logging.getLogger(__name__)


class RelationExtractor(Protocol):
    def for_document(self, settings: LlmRequestSettings) -> "RelationExtractor":
        ...

    def close(self) -> None:
        ...

    def extract_document(
        self,
        document_text: str,
        on_delta: Optional[Callable[[str, str, str], None]] = ...,
        stream_prefix: str = ...,
        stream_thinking: Optional[bool] = ...,
        thinking_budget: Optional[int] = ...,
    ) -> dict[str, Any]:
        ...


class Skill4ReClientError(RuntimeError):
    pass


class Skill4ReClient:
    def __init__(self, config: AppConfig, settings: LlmRequestSettings | None = None) -> None:
        self.config = config
        self.settings = settings
        self._ensure_import_path()

        from skill4re.backends import build_backend
        from skill4re.loader import load_skills
        from skill4re.routing import load_route_cache, save_route_cache
        from skill4re.service import SkillRouterExtractor

        self._build_backend = build_backend
        self._load_skills = load_skills
        self._load_route_cache = load_route_cache
        self._save_route_cache = save_route_cache
        self._extractor_class = SkillRouterExtractor
        self._client = None
        self._local_generator = None
        self.skills = []
        self.route_cache = {}
        self.extractor = None
        if settings is not None:
            self._reload()

    def for_document(self, settings: LlmRequestSettings) -> "Skill4ReClient":
        return Skill4ReClient(self.config, settings)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def _reload(self) -> None:
        settings = self.settings
        if settings is None:
            raise Skill4ReClientError("文档模型配置尚未绑定")
        self.skills = self._load_skills(self.config.skill4re_skills_dir)
        self.route_cache = self._load_route_cache(self.config.skill4re_route_cache_path)
        self._client, self._local_generator = self._build_backend(
            backend=settings.backend, api_key=settings.api_key,
            model_path=os.getenv("SKILL4RE_LOCAL_MODEL_PATH", ""),
            base_url=settings.base_url, timeout=settings.request_timeout_seconds,
            max_retries=settings.max_retries,
        )

        self.extractor = self._extractor_class(
            skills=self.skills,
            backend=settings.backend,
            model=settings.model,
            client=self._client,
            local_generator=self._local_generator,
            route_cache=self.route_cache,
            api_key=settings.api_key,
            base_url=settings.base_url,
            request_timeout=settings.request_timeout_seconds,
            enable_thinking=settings.enable_thinking,
            max_retries=settings.max_retries,
            retry_backoff_seconds=settings.retry_backoff_seconds,
            stream_thinking=self.config.stream_thinking,
            fast_mode=self.config.skill4re_fast_mode,
            skip_coref=self.config.skill4re_skip_coref,
        )

    def extract_document(
        self,
        document_text: str,
        on_delta: Optional[Callable[[str, str, str], None]] = None,
        stream_prefix: str = "",
        stream_thinking: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
    ) -> dict[str, Any]:
        if self.settings is None:
            raise Skill4ReClientError("文档模型配置尚未绑定")
        if not document_text.strip():
            return {
                "preprocess": {
                    "doc_token_count": 0,
                    "used_chunking": False,
                    "chunk_count": 0,
                    "chunk_trigger": self.config.skill4re_chunk_trigger,
                    "chunk_budget": self.config.skill4re_chunk_budget,
                    "coref_resolved": False,
                    "coref_entity_groups": 0,
                },
                "routing": {
                    "selected_skills": [],
                    "document_selected_skills": [],
                    "scores": {},
                    "router_reason": "空文档，跳过抽取。",
                    "router_mode": "skipped",
                    "cache_hit": False,
                    "chunk_routes": [],
                },
                "timing": {
                    "routing_seconds": 0.0,
                    "coref_seconds": 0.0,
                    "chunk_routing_seconds": 0.0,
                    "extraction_seconds": 0.0,
                    "summarize_seconds": 0.0,
                    "proofreading_seconds": 0.0,
                    "refinement_seconds": 0.0,
                    "domain_reflection_seconds": 0.0,
                    "total_seconds": 0.0,
                },
                "proofreading": {
                    "input_relation_count": 0,
                    "high_confidence_count": 0,
                    "low_confidence_count": 0,
                    "output_relation_count": 0,
                },
                "domain_reflection": {
                    "input_relation_count": 0,
                    "high_confidence_count": 0,
                    "low_confidence_count": 0,
                    "output_relation_count": 0,
                },
                "chunk_predictions": [],
                "prediction": {"relation_list": []},
            }

        if self.extractor is None:
            raise Skill4ReClientError("Skill4RE extractor is not initialized.")
        result = self.extractor.extract_document(
            doc_text=document_text,
            chunk_trigger=self.config.skill4re_chunk_trigger,
            chunk_budget=self.config.skill4re_chunk_budget,
            max_workers=self.config.skill4re_max_workers,
            on_delta=on_delta,
            stream_prefix=stream_prefix,
            stream_thinking=stream_thinking,
            thinking_budget=thinking_budget,
        )
        self.config.skill4re_route_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_route_cache(self.config.skill4re_route_cache_path, self.route_cache)
        return result

    @staticmethod
    def _ensure_import_path() -> None:
        skill4re_root = Path(__file__).resolve().parents[2] / "skill4re"
        if not skill4re_root.exists():
            raise Skill4ReClientError(f"skill4re directory not found: {skill4re_root}")
        path = str(skill4re_root)
        if path not in sys.path:
            sys.path.insert(0, path)
