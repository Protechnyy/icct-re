import os
from pathlib import Path

from dotenv import dotenv_values


def configure_environment(source: Path, output: Path) -> None:
    from app.config import RETIRED_MODEL_VARIABLES

    values = {key: value for key, value in dotenv_values(source).items() if value is not None}
    backend = values.get("LLM_BACKEND", values.get("SKILL4RE_BACKEND", "vllm"))
    address_names = {"qwen_api": "SKILL4RE_BASE_URL", "openai": "SKILL4RE_BASE_URL", "api": "SKILL4RE_BASE_URL"}
    key_names = {"qwen_api": "DASHSCOPE_API_KEY", "openai": "OPENAI_API_KEY", "api": "DEEPSEEK_API_KEY"}
    public = {new: values[old] for old, new in RETIRED_MODEL_VARIABLES.items() if old in values}
    public.update({key: value for key, value in values.items() if key.startswith(("LLM_", "VERIFICATION_AGENT_"))})
    public["LLM_BACKEND"] = backend
    public["LLM_BASE_URL"] = values.get("LLM_BASE_URL") or values.get(address_names.get(backend, "VLLM_BASE_URL")) or values.get("VLLM_BASE_URL", "")
    public["LLM_API_KEY"] = values.get("LLM_API_KEY") or values.get(key_names.get(backend, "VLLM_API_KEY"), "")
    public["LLM_MODEL"] = values.get("LLM_MODEL") or values.get("SKILL4RE_MODEL") or values.get("VLLM_MODEL", "qwen3.8-27b")
    for name in RETIRED_MODEL_VARIABLES:
        os.environ.pop(name, None)
    os.environ.update({key: value for key, value in values.items() if key not in RETIRED_MODEL_VARIABLES})
    os.environ.update(public)
    os.environ["STORAGE_ROOT"] = str(output.resolve())
    os.environ["SKILL4RE_SKILLS_DIR"] = str(source.resolve().parent.parent / "skill4re" / "skill4re" / "skills")
    os.environ["SKILL4RE_ROUTE_CACHE_PATH"] = str(output.resolve() / "route_cache.json")
    os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY", ""), "127.0.0.1", "localhost", "::1"]))
