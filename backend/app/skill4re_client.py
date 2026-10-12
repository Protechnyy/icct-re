from __future__ import annotations

import logging
import os
import sys
from threading import Lock
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from .config import AppConfig, normalize_openai_base_url
from .model_settings import ModelSettings, ModelSettingsStore, main_model_settings_defaults

LOGGER = logging.getLogger(__name__)


class RelationExtractor(Protocol):
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
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model_settings_store = ModelSettingsStore(config.storage_root)
        self._reload_lock = Lock()
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
        self._model_settings_fingerprint: tuple[int, int, int] | None = None
        self._client = None
        self._local_generator = None
        self.skills = []
        self.route_cache = {}
        self.extractor = None
        self._reload()

    def _reload(self) -> None:
        defaults = main_model_settings_defaults(self.config)
        model_settings = self.model_settings_store.get(*defaults) if defaults is not None else None
        self.skills = self._load_skills(self.config.skill4re_skills_dir)
        self.route_cache = self._load_route_cache(self.config.skill4re_route_cache_path)

        try:
            self._client, self._local_generator = self._build_backend(
                backend=self.config.skill4re_backend,
                api_key=self._api_key(model_settings),
                model_path=os.getenv("SKILL4RE_LOCAL_MODEL_PATH", ""),
            )
        except ImportError:
            LOGGER.info("OpenAI SDK is not installed; skill4re will use requests fallback.")
            self._client, self._local_generator = None, None

        # SDK backends also need the configured deployment URL; otherwise qwen_api
        # always connects to its library default even when SKILL4RE_BASE_URL is set.
        base_url = self._base_url(model_settings)
        api_key = self._api_key(model_settings)
        if self._client is not None and base_url:
            self._client = self._client.with_options(
                base_url=normalize_openai_base_url(base_url))

        self.extractor = self._extractor_class(
            skills=self.skills,
            backend=self.config.skill4re_backend,
            model=self.config.skill4re_model,
            client=self._client,
            local_generator=self._local_generator,
            route_cache=self.route_cache,
            api_key=api_key,
            base_url=base_url,
            request_timeout=self.config.vllm_timeout_seconds,
            enable_thinking=self.config.vllm_enable_thinking,
            stream_thinking=self.config.stream_thinking,
            fast_mode=self.config.skill4re_fast_mode,
            skip_coref=self.config.skill4re_skip_coref,
        )
        self._skills_fingerprint = self._fingerprint_skills()
        self._model_settings_fingerprint = (
            self.model_settings_store.fingerprint() if defaults is not None else None
        )

    def extract_document(
        self,
        document_text: str,
        on_delta: Optional[Callable[[str, str, str], None]] = None,
        stream_prefix: str = "",
        stream_thinking: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
    ) -> dict[str, Any]:
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
            on_delta=on_delta,
            stream_prefix=stream_prefix,
            stream_thinking=stream_thinking,
            thinking_budget=thinking_budget,
        )
        self.config.skill4re_route_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_route_cache(self.config.skill4re_route_cache_path, self.route_cache)
        return result

    def _reload_if_changed(self) -> None:
        with self._reload_lock:
            fingerprint = self._fingerprint_skills()
            defaults = main_model_settings_defaults(self.config)
            settings_fingerprint = self.model_settings_store.fingerprint() if defaults is not None else None
            if (
                fingerprint == self._skills_fingerprint
                and settings_fingerprint == self._model_settings_fingerprint
            ):
                return
            LOGGER.info("Skill4RE skills or model settings changed; reloading %s", self.config.skill4re_skills_dir)
            self._reload()

    def _fingerprint_skills(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (path.name, path.stat().st_mtime_ns, path.stat().st_size)
            for path in sorted(self.config.skill4re_skills_dir.glob("*.json"))
        )

    def _api_key(self, model_settings: ModelSettings | None = None) -> str:
        defaults = main_model_settings_defaults(self.config)
        if defaults is not None:
            if model_settings is None:
                model_settings = self.model_settings_store.get(*defaults)
            return model_settings.api_key
        return ""

    def _base_url(self, model_settings: ModelSettings | None = None) -> str | None:
        defaults = main_model_settings_defaults(self.config)
        if defaults is not None:
            if model_settings is None:
                model_settings = self.model_settings_store.get(*defaults)
            return model_settings.base_url
        return None

    @staticmethod
    def _ensure_import_path() -> None:
        skill4re_root = Path(__file__).resolve().parents[2] / "skill4re"
        if not skill4re_root.exists():
            raise Skill4ReClientError(f"skill4re directory not found: {skill4re_root}")
        path = str(skill4re_root)
        if path not in sys.path:
            sys.path.insert(0, path)
