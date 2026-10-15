from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .config import normalize_openai_base_url

DEFAULT_QWEN_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class ModelSettingsError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSettings:
    base_url: str
    api_key: str
    model: str = ""
    default_model: str = ""

    @property
    def api_key_configured(self) -> bool:
        return bool(self.api_key.strip())


class ModelSettingsStore:
    def __init__(self, storage_root: Path) -> None:
        self.path = storage_root / "settings" / "model.json"

    def get(
        self,
        default_base_url: str,
        default_api_key: str,
        default_model: str = "",
    ) -> ModelSettings:
        if not self.path.exists():
            return ModelSettings(
                normalize_openai_base_url(default_base_url),
                default_api_key,
                default_model,
                default_model,
            )

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelSettingsError("模型设置文件格式无效") from exc
        if not isinstance(payload, dict):
            raise ModelSettingsError("模型设置文件格式无效")
        base_url = payload.get("base_url")
        api_key = payload.get("api_key")
        if not isinstance(base_url, str) or not isinstance(api_key, str):
            raise ModelSettingsError("模型设置文件缺少有效的 API 地址或 Key")
        model_value = payload.get("model")
        if model_value is not None and not isinstance(model_value, str):
            raise ModelSettingsError("模型设置文件中的模型名称无效")
        model_override = model_value.strip() if isinstance(model_value, str) else ""
        effective_model = (
            model_override
            if model_override and model_override != default_model.strip()
            else default_model
        )
        return ModelSettings(
            _validated_base_url(base_url),
            api_key,
            effective_model,
            default_model,
        )

    def save(
        self,
        base_url: str,
        api_key: str,
        model: str = "",
        default_model: str = "",
    ) -> ModelSettings:
        model_override = model.strip()
        if model_override == default_model.strip():
            model_override = ""
        settings = ModelSettings(
            _validated_base_url(base_url),
            api_key.strip(),
            model_override or default_model,
            default_model,
        )
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".model-", suffix=".tmp", dir=self.path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(file_descriptor, 0o600)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
                payload = {"base_url": settings.base_url, "api_key": settings.api_key}
                if model_override:
                    payload["model"] = model_override
                json.dump(payload, output)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, self.path)
            os.chmod(self.path, 0o600)
        finally:
            temporary_path.unlink(missing_ok=True)
        return settings

    def fingerprint(self) -> tuple[int, int, int] | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size, stat.st_ino


def main_model_settings_defaults(config: Any) -> tuple[str, str, str] | None:
    backend = getattr(config, "skill4re_backend", "vllm")
    default_model = getattr(config, "skill4re_model", getattr(config, "vllm_model", ""))
    if backend == "vllm":
        return config.vllm_base_url, config.vllm_api_key, default_model
    if backend == "qwen_api":
        return (
            os.getenv("SKILL4RE_BASE_URL") or DEFAULT_QWEN_API_BASE_URL,
            os.getenv("DASHSCOPE_API_KEY", ""),
            default_model,
        )
    if backend == "openai":
        return (
            os.getenv("SKILL4RE_BASE_URL") or "https://api.openai.com/v1",
            os.getenv("OPENAI_API_KEY", ""),
            default_model,
        )
    if backend == "api":
        return (
            os.getenv("SKILL4RE_BASE_URL") or "https://api.deepseek.com",
            os.getenv("DEEPSEEK_API_KEY", ""),
            default_model,
        )
    return None


def _validated_base_url(value: str) -> str:
    base_url = normalize_openai_base_url(value)
    try:
        parsed = urlsplit(base_url)
        hostname = parsed.hostname
        parsed.port
    except ValueError as exc:
        raise ModelSettingsError("API 地址格式无效") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ModelSettingsError("API 地址必须是有效的 HTTP 或 HTTPS 地址")
    return base_url
