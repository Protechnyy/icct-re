from __future__ import annotations

import requests
from typing import Any

from .config import AppConfig
from .utils import extract_json_array


class VllmClientError(RuntimeError):
    pass


class VllmClient:
    def __init__(self, config: AppConfig, session: requests.Session | None = None) -> None:
        self.config = config
        self.session = session or requests.Session()

    def healthcheck(self) -> bool:
        try:
            response = self.session.get(f"{self.config.vllm_base_url}/models", timeout=5, headers=self._headers())
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
        payload = {
            "model": self.config.vllm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "chat_template_kwargs": {
                "enable_thinking": self.config.vllm_enable_thinking,
            },
        }
        try:
            response = self.session.post(
                f"{self.config.vllm_base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.config.vllm_timeout_seconds,
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

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.vllm_api_key}",
            "Content-Type": "application/json",
        }
