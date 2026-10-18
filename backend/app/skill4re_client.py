import json
import logging
import os
import time
from pathlib import Path
from typing import Protocol

from openai import APIConnectionError, APIStatusError

from skill4re.backends import build_backend, strip_think_section
from skill4re.config import max_tokens_for
from skill4re.typed_extraction import ConfiguredExtractor

from .config import AppConfig
from .extraction_schema_store import ExtractionSchemaStore
from .model_settings import LlmRequestSettings
from .llm_output_guard import OutputRepetitionGuard


LOGGER = logging.getLogger(__name__)


def provider_schema(schema):
    if isinstance(schema, list):
        return [provider_schema(item) for item in schema]
    if isinstance(schema, dict):
        return {key: provider_schema(value) for key, value in schema.items()
                if key != "uniqueItems" and not (key == "pattern" and value == r"\S")}
    return schema


class RelationExtractor(Protocol):
    def for_document(self, settings, extraction_schema=None): ...

    def close(self): ...

    def extract_document(self, document_text, **options): ...


class Skill4ReClientError(RuntimeError):
    pass


def require_complete_output(finish_reason, stream_id):
    if finish_reason != "stop":
        raise Skill4ReClientError(f"{stream_id}: 模型输出未完整完成，finish_reason={finish_reason}")


class Skill4ReClient:
    def __init__(self, config: AppConfig, settings: LlmRequestSettings | None = None, extraction_schema=None):
        self.config = config
        self.settings = settings
        self.extraction_schema = extraction_schema
        self._client = None
        self._local_generator = None
        self.extractor = None
        self.diagnostic_dir = None
        if settings is not None:
            if extraction_schema is None:
                self.extraction_schema = ExtractionSchemaStore(config.storage_root).snapshot()
            self._client, self._local_generator = build_backend(
                backend="api" if settings.backend == "vllm" else settings.backend,
                api_key=settings.api_key, model_path=os.getenv("SKILL4RE_LOCAL_MODEL_PATH", ""),
                base_url=settings.base_url, timeout=settings.request_timeout_seconds, max_retries=1,
            )
            reserve = max(max_tokens_for(stage, settings.backend) for stage in ("entity_recognition", "entity_proofreading", "entity_merge", "extraction", "proofreading"))
            self.extractor = ConfiguredExtractor(self.extraction_schema, self._call_model,
                                                 input_limit=config.extraction_max_request_tokens - reserve,
                                                 max_workers=config.relation_batch_concurrency)
            self.extractor.model = settings.model
            self.extractor.base_url = settings.base_url

    def for_document(self, settings, extraction_schema=None):
        return Skill4ReClient(self.config, settings, extraction_schema)

    def close(self):
        if self._client is not None:
            self._client.close()

    def _call_model(self, prompt, stage, schema, on_delta, stream_id, stream_thinking, reasoning_effort, cancel_check):
        cancel_check()
        settings = self.settings
        limit = max_tokens_for(stage, settings.backend)
        if self._local_generator is not None:
            raw = self._local_generator.generate(prompt, max_new_tokens=limit)
            self._save_diagnostic(stream_id, prompt, raw, stage)
            return raw
        streaming = on_delta is not None
        options = {"model": settings.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": limit, "stream": streaming}
        if stage == "extraction" and settings.backend == "vllm":
            options["temperature"] = 0.2
        extra = {}
        if settings.backend == "vllm":
            extra["chat_template_kwargs"] = {"enable_thinking": False if streaming else settings.enable_thinking}
        elif settings.backend == "qwen_api":
            extra["enable_thinking"] = bool(streaming and stream_thinking)
            if streaming and stream_thinking and reasoning_effort:
                extra["reasoning_effort"] = reasoning_effort
        if extra:
            options["extra_body"] = extra
        options["response_format"] = {"type": "json_schema", "json_schema": {"name": stage, "strict": True, "schema": provider_schema(schema)}}
        LOGGER.info("LLM request model=%s backend=%s stream_id=%s revision=%s", settings.model, settings.backend, stream_id, self.extraction_schema.revision)
        for attempt in range(1, settings.max_retries + 1):
            cancel_check()
            identifier = stream_id if attempt == 1 else f"{stream_id}#retry{attempt}"
            self._save_diagnostic(identifier, prompt, None, stage, options)
            try:
                if streaming:
                    parts = []
                    repetition_guard = OutputRepetitionGuard()
                    finish_reason = None
                    with self._client.chat.completions.create(**options) as stream:
                        for chunk in stream:
                            if not chunk.choices:
                                continue
                            choice = chunk.choices[0]
                            if choice.finish_reason is not None:
                                finish_reason = choice.finish_reason
                            if choice.finish_reason in {"length", "content_filter"}:
                                self._save_diagnostic(identifier, prompt, "".join(parts), stage, options)
                                raise Skill4ReClientError(f"{identifier}: 模型输出未完整完成，finish_reason={choice.finish_reason}")
                            reasoning = getattr(choice.delta, "reasoning_content", None)
                            if reasoning:
                                on_delta(identifier, "think", reasoning)
                            if choice.delta.content:
                                parts.append(choice.delta.content)
                                on_delta(identifier, "answer", choice.delta.content)
                                violation = repetition_guard.feed(choice.delta.content)
                                if violation:
                                    self._save_diagnostic(identifier, prompt, "".join(parts), stage, options)
                                    raise Skill4ReClientError(f"{identifier}: {violation}")
                    self._save_diagnostic(identifier, prompt, "".join(parts), stage, options)
                    require_complete_output(finish_reason, identifier)
                    raw = strip_think_section("".join(parts))
                else:
                    response = self._client.chat.completions.create(**options)
                    choice = response.choices[0]
                    self._save_diagnostic(identifier, prompt, choice.message.content, stage, options)
                    require_complete_output(choice.finish_reason, identifier)
                    raw = strip_think_section(choice.message.content)
                self._save_diagnostic(identifier, prompt, raw, stage, options)
                return raw
            except APIStatusError as exc:
                if exc.status_code not in {408, 429, 500, 502, 503, 504} or attempt == settings.max_retries:
                    raise Skill4ReClientError(f"{identifier}: 模型服务返回 HTTP {exc.status_code}") from exc
            except APIConnectionError as exc:
                if attempt == settings.max_retries:
                    raise Skill4ReClientError(f"{identifier}: 无法连接模型服务") from exc
            time.sleep(settings.retry_backoff_seconds * attempt)
        raise Skill4ReClientError("模型请求未返回结果")

    def _save_diagnostic(self, stream_id, prompt, raw, stage, request_options=None):
        if self.diagnostic_dir is None:
            return
        self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
        path = self.diagnostic_dir / (stream_id + ".json")
        request = {key: value for key, value in (request_options or {}).items()
                   if key in {"model", "temperature", "max_tokens", "stream", "extra_body"}}
        path.write_text(json.dumps({"stage": stage, "revision": self.extraction_schema.revision,
                                    "model": self.settings.model, "request": request,
                                    "prompt": prompt, "output": raw}, ensure_ascii=False, indent=2), encoding="utf-8")

    def extract_document(self, document_text, *, batches=None, sources=None, diagnostic_dir=None, **options):
        if self.settings is None or self.extractor is None:
            raise Skill4ReClientError("文档模型配置尚未绑定")
        self.diagnostic_dir = Path(diagnostic_dir) if diagnostic_dir is not None else None
        if batches is None or sources is None:
            raise Skill4ReClientError("实体与关系抽取需要完整的文档来源和批次")
        return self.extractor.extract_document(document_text, batches, sources, **options)
