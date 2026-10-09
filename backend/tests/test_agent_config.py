import json

from app.config import AppConfig, DEFAULT_LLM_BASE_URL


def test_agent_defaults(monkeypatch):
    monkeypatch.setattr("app.config._load_dotenv", lambda: None)
    for key in tuple(__import__("os").environ):
        if key.startswith("AGENT_") or key in ("VLLM_MODEL", "VLLM_BASE_URL", "SKILL4RE_MODEL"):
            monkeypatch.delenv(key)
    config = AppConfig.from_env()
    assert not config.agent_enabled
    assert config.agent_api_key == ""
    assert config.agent_model == config.vllm_model == config.skill4re_model == "qwen3.8-27b"
    assert config.agent_base_url == config.vllm_base_url == DEFAULT_LLM_BASE_URL
    assert (config.agent_concurrency, config.agent_max_tasks, config.agent_max_steps,
            config.agent_max_llm_calls, config.agent_timeout_seconds,
            config.agent_max_added_per_task) == (1, 20, 6, 80, 600, 5)


def test_agent_environment_and_safe_summary(monkeypatch):
    monkeypatch.setattr("app.config._load_dotenv", lambda: None)
    overrides = {"ENABLED": "true", "BASE_URL": "https://example.test/v1/chat/completions",
                 "API_KEY": "test-agent-secret", "MODEL": "custom-model", "CONCURRENCY": "3",
                 "MAX_TASKS": "8", "MAX_STEPS": "4", "MAX_LLM_CALLS": "10",
                 "TIMEOUT_SECONDS": "30", "MAX_ADDED_PER_TASK": "2"}
    for key, value in overrides.items():
        monkeypatch.setenv("AGENT_" + key, value)
    config = AppConfig.from_env()
    assert config.agent_enabled and config.agent_api_key == "test-agent-secret"
    assert config.agent_base_url == "https://example.test/v1"
    assert config.agent_model == "custom-model"
    assert (config.agent_concurrency, config.agent_max_tasks, config.agent_max_steps,
            config.agent_max_llm_calls, config.agent_timeout_seconds,
            config.agent_max_added_per_task) == (3, 8, 4, 10, 30, 2)
    summary = config.safe_summary()
    assert summary["agent_enabled"] is True
    assert "agent_api_key" not in summary
    assert "test-agent-secret" not in json.dumps(summary)
