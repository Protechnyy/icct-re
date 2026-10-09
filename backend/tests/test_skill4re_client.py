from dataclasses import replace
from pathlib import Path

from app.skill4re_client import Skill4ReClient
from test_pipeline import build_config


def test_qwen_sdk_uses_configured_deployment_url(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("SKILL4RE_BASE_URL", "https://deployment.example/v1/chat/completions")
    config = replace(build_config(tmp_path), skill4re_backend="qwen_api",
                     skill4re_skills_dir=Path(__file__).resolve().parents[2] / "skill4re/skill4re/skills")
    client = Skill4ReClient(config)
    assert str(client._client.base_url) == "https://deployment.example/v1/"
    assert client.extractor.base_url == "https://deployment.example/v1/chat/completions"


def test_qwen_sdk_keeps_library_default_without_override(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.delenv("SKILL4RE_BASE_URL", raising=False)
    config = replace(build_config(tmp_path), skill4re_backend="qwen_api",
                     skill4re_skills_dir=Path(__file__).resolve().parents[2] / "skill4re/skill4re/skills")
    client = Skill4ReClient(config)
    assert str(client._client.base_url) == "https://dashscope.aliyuncs.com/compatible-mode/v1/"
