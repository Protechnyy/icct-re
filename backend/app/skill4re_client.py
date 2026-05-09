from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Protocol

from .config import AppConfig

LOGGER = logging.getLogger(__name__)


class RelationExtractor(Protocol):
    def extract_document(self, document_text: str) -> dict[str, Any]:
        ...


class Skill4ReClientError(RuntimeError):
    pass


class Skill4ReClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._ensure_import_path()

        try:
            from skill4re.backends import build_backend
            from skill4re.loader import load_skills
            from skill4re.routing import load_route_cache, save_route_cache
            from skill4re.service import SkillRouterExtractor
        except ImportError as exc:
            raise Skill4ReClientError(f"Unable to import skill4re: {exc}") from exc

        self._build_backend = build_backend
        self._load_skills = load_skills
        self._load_route_cache = load_route_cache
        self._save_route_cache = save_route_cache
        self._extractor_class = SkillRouterExtractor
        self._skills_fingerprint: tuple[tuple[str, int, int], ...] = ()
        self._client = None
        self._local_generator = None
        self.skills = []
        self.route_cache = {}
        self.extractor = None
        self._reload()

    def _reload(self) -> None:
        self.skills = self._load_skills(self.config.skill4re_skills_dir)
        self.route_cache = self._load_route_cache(self.config.skill4re_route_cache_path)

        try:
            self._client, self._local_generator = self._build_backend(
                backend=self.config.skill4re_backend,
                api_key=self._api_key(),
                model_path=os.getenv("SKILL4RE_LOCAL_MODEL_PATH", ""),
            )
        except ImportError:
            LOGGER.info("OpenAI SDK is not installed; skill4re will use requests fallback.")
            self._client, self._local_generator = None, None

        self.extractor = self._extractor_class(
            skills=self.skills,
            backend=self.config.skill4re_backend,
            model=self.config.skill4re_model,
            client=self._client,
            local_generator=self._local_generator,
            route_cache=self.route_cache,
            api_key=self._api_key(),
            base_url=self._base_url(),
            request_timeout=self.config.vllm_timeout_seconds,
            enable_thinking=self.config.vllm_enable_thinking,
            fast_mode=self.config.skill4re_fast_mode,
            skip_coref=self.config.skill4re_skip_coref,
        )
        self._skills_fingerprint = self._fingerprint_skills()

    def extract_document(self, document_text: str) -> dict[str, Any]:
        self._reload_if_changed()
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
        )
        self.config.skill4re_route_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_route_cache(self.config.skill4re_route_cache_path, self.route_cache)
        return result

    def _reload_if_changed(self) -> None:
        fingerprint = self._fingerprint_skills()
        if fingerprint == self._skills_fingerprint:
            return
        LOGGER.info("Skill4RE skills changed; reloading %s", self.config.skill4re_skills_dir)
        self._reload()

    def _fingerprint_skills(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (path.name, path.stat().st_mtime_ns, path.stat().st_size)
            for path in sorted(self.config.skill4re_skills_dir.glob("*.json"))
        )

    def _api_key(self) -> str:
        if self.config.skill4re_backend == "vllm":
            return self.config.vllm_api_key
        if self.config.skill4re_backend == "qwen_api":
            return os.getenv("DASHSCOPE_API_KEY", "")
        if self.config.skill4re_backend == "openai":
            return os.getenv("OPENAI_API_KEY", "")
        if self.config.skill4re_backend == "api":
            return os.getenv("DEEPSEEK_API_KEY", "")
        return ""

    def _base_url(self) -> str | None:
        if self.config.skill4re_backend == "vllm":
            return self.config.vllm_base_url
        return os.getenv("SKILL4RE_BASE_URL")

    @staticmethod
    def _ensure_import_path() -> None:
        skill4re_root = Path(__file__).resolve().parents[2] / "skill4re"
        if not skill4re_root.exists():
            raise Skill4ReClientError(f"skill4re directory not found: {skill4re_root}")
        path = str(skill4re_root)
        if path not in sys.path:
            sys.path.insert(0, path)
