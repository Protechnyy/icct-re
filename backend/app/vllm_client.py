from __future__ import annotations

import requests
from typing import Any

from .config import AppConfig
from .model_settings import LlmRequestSettings, ModelSettingsStore, resolve_model_settings
from .utils import extract_json_array


class VllmClientError(RuntimeError):
    pass


class VllmClient:
    def __init__(
        self,
        config: AppConfig,
        session: requests.Session | None = None,
        settings_store: ModelSettingsStore | None = None,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()
        storage_root = getattr(config, "storage_root", None)
        self.settings_store = settings_store or (
            ModelSettingsStore(storage_root) if storage_root is not None else None
        )

    def healthcheck(self) -> bool:
        settings = self._settings()
        try:
            response = self.session.get(
                f"{settings.base_url}/models", timeout=5, headers=self._headers(settings)
            )
            if response.ok:
                if settings.backend == "qwen_api":
                    return self._chat_completion_healthcheck(settings)
                return True
            if response.status_code not in {404, 405}:
                return False
            return self._chat_completion_healthcheck(settings)
        except requests.RequestException:
            return False

    def _chat_completion_healthcheck(self, settings: LlmRequestSettings | None = None) -> bool:
        settings = settings or self._settings()
        payload = {
            "model": settings.model,
            "messages": [{"role": "user", "content": "ping"}],
            "temperature": 0,
            "max_tokens": 1,
            "stream": False,
            **settings.thinking_options(settings.enable_thinking),
        }
        try:
            response = self.session.post(
                f"{settings.base_url}/chat/completions",
                json=payload,
                headers=self._headers(settings),
                timeout=min(10, settings.request_timeout_seconds),
            )
            return response.ok
        except requests.RequestException:
            return False

    def extract_relations(self, text: str, context: str) -> list[dict[str, Any]]:
        system_prompt = (
            "你是文档级关系抽取助手。"
            "请从给定文本中抽取关系三元组，只返回 JSON 数组。"
            '每个元素格式为 {"subject": "...", "relation": "...", "object": "..."}。'
            "如果没有关系，返回空数组。"
        )
        user_prompt = f"任务上下文：{context}\n\n文本：\n{text}"
        settings = self._settings()
        payload = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            **settings.thinking_options(settings.enable_thinking),
        }
        try:
            response = self.session.post(
                f"{settings.base_url}/chat/completions",
                json=payload,
                headers=self._headers(settings),
                timeout=settings.request_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise VllmClientError(f"vLLM request failed: {exc}") from exc
        except ValueError as exc:
            raise VllmClientError("vLLM returned invalid JSON") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VllmClientError("vLLM returned an unexpected response shape") from exc
        return extract_json_array(content if isinstance(content, str) else "")

    def _headers(self, settings: LlmRequestSettings | None = None) -> dict[str, str]:
        settings = settings or self._settings()
        headers = {"Content-Type": "application/json"}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        return headers

    def _thinking_options(self) -> dict[str, Any]:
        settings = self._settings()
        return settings.thinking_options(settings.enable_thinking)

    def _base_url(self) -> str:
        return self._settings().base_url

    def _settings(self) -> LlmRequestSettings:
        return resolve_model_settings(self.config, self.settings_store)
