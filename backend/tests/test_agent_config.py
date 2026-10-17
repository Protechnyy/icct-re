import json

from test_shared_llm_configuration import read_config


def test_verification_defaults_use_public_connection():
    result = read_config()
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["verification_agent_enabled"] is False
    assert summary["llm_model"] == "qwen3.8-27b"
    assert tuple(summary["verification_agent_" + name] for name in (
        "concurrency", "max_tasks", "max_steps", "max_llm_calls", "timeout_seconds", "max_added_per_task",
    )) == (1, 20, 6, 80, 600, 5)
    assert not any(name in summary for name in ("agent_api_key", "llm_api_key", "agent_model", "agent_base_url"))


def test_verification_environment_and_safe_summary():
    result = read_config(VERIFICATION_AGENT_ENABLED="true", VERIFICATION_AGENT_CONCURRENCY="3",
                         VERIFICATION_AGENT_MAX_TASKS="8", VERIFICATION_AGENT_MAX_STEPS="4",
                         VERIFICATION_AGENT_MAX_LLM_CALLS="10", VERIFICATION_AGENT_TIMEOUT_SECONDS="30",
                         VERIFICATION_AGENT_MAX_ADDED_PER_TASK="2")
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["verification_agent_enabled"] is True
    assert tuple(summary["verification_agent_" + name] for name in (
        "concurrency", "max_tasks", "max_steps", "max_llm_calls", "timeout_seconds", "max_added_per_task",
    )) == (3, 8, 4, 10, 30, 2)
    assert "llm_api_key" not in summary
