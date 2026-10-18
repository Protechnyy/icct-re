import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import pytest

from app.config import AppConfig, RETIRED_MODEL_VARIABLES
from app.model_settings import ModelSettingsError, ModelSettingsStore, resolve_model_settings


ROOT = Path(__file__).resolve().parents[2]
CHECK_ROOT = ROOT / "data" / "shared_llm_config_checks"
CHECK_ROOT.mkdir(parents=True, exist_ok=True)


def read_config(**environment):
    env = {key: value for key, value in os.environ.items()
           if key not in RETIRED_MODEL_VARIABLES and not key.startswith("LLM_")}
    env.update(environment)
    result = subprocess.run(
        [sys.executable, "-c", "import json; from pathlib import Path; from app.config import AppConfig; "
         "c=AppConfig.from_env(Path('data/shared_llm_config_checks/absent.env')); "
         "print(json.dumps(c.safe_summary()))"],
        cwd=ROOT, env={**env, "PYTHONPATH": str(ROOT / "backend")},
        text=True, capture_output=True,
    )
    return result


def base_config(storage):
    result = read_config()
    assert result.returncode == 0, result.stderr
    env_path = CHECK_ROOT / "absent.env"
    config = AppConfig.from_env(env_path)
    return replace(config, storage_root=Path(storage))


def test_public_environment_defaults_and_overrides():
    result = read_config()
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["llm_backend"] == "vllm"
    assert summary["llm_request_timeout_seconds"] == 180
    assert "llm_api_key" not in summary
    result = read_config(LLM_BACKEND="qwen_api", LLM_REQUEST_TIMEOUT_SECONDS="23")
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["llm_backend"] == "qwen_api"
    assert summary["llm_request_timeout_seconds"] == 23


@pytest.mark.parametrize("environment", [
    {"LLM_REQUEST_TIMEOUT_SECONDS": "0"}, {"LLM_MAX_RETRIES": "0"},
    {"LLM_RETRY_BACKOFF_SECONDS": "nan"},
])
def test_invalid_environment_fails(environment):
    result = read_config(**environment)
    assert result.returncode != 0
    assert "ValueError" in result.stderr


def test_retired_environment_reports_replacement():
    result = read_config(VLLM_MODEL="qwen3.8-27b")
    assert result.returncode != 0
    assert "VLLM_MODEL → LLM_MODEL" in result.stderr


def test_saved_settings_snapshot_is_immutable_and_private():
    with TemporaryDirectory(dir=CHECK_ROOT) as directory:
        config = base_config(directory)
        store = ModelSettingsStore(config.storage_root)
        first = resolve_model_settings(config, store)
        store.save(config.llm_base_url, config.llm_api_key, config.llm_model + "-updated", config.llm_model)
        second = resolve_model_settings(config, store)
        assert first.model == config.llm_model
        assert second.model == config.llm_model + "-updated"
        with pytest.raises(FrozenInstanceError):
            first.model = second.model
        assert "api_key=" not in repr(first)
        assert "api_key=" not in repr(config)
        assert "api_key=" not in repr(store.get(config.llm_base_url, config.llm_api_key, config.llm_model))


def test_saved_settings_support_existing_shape_and_default_model():
    with TemporaryDirectory(dir=CHECK_ROOT) as directory:
        config = base_config(directory)
        store = ModelSettingsStore(config.storage_root)
        saved = store.save(config.llm_base_url, config.llm_api_key, "", config.llm_model)
        payload = json.loads(store.path.read_text())
        assert "model" not in payload
        assert resolve_model_settings(config, store).model == config.llm_model
        assert saved.api_key == config.llm_api_key
        assert store.path.stat().st_mode & 0o777 == 0o600
        assert store.path.parent.stat().st_mode & 0o777 == 0o700


def test_atomic_settings_reads_and_writes():
    with TemporaryDirectory(dir=CHECK_ROOT) as directory:
        config = base_config(directory)
        store = ModelSettingsStore(config.storage_root)
        values = [(config.llm_base_url, config.llm_api_key, config.llm_model),
                  (config.llm_base_url + "/alternate", config.llm_api_key, config.llm_model + "-updated")]
        store.save(*values[0])
        def save():
            for index in range(50):
                store.save(*values[index % 2])
        def read():
            for _ in range(100):
                settings = store.get(*values[0])
                assert (settings.base_url, settings.api_key, settings.model) in values
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(save), executor.submit(read), executor.submit(read)]
            for future in futures:
                future.result()


@pytest.mark.parametrize("overrides", [
    {"llm_backend": "unsupported"}, {"llm_base_url": "ftp://localhost"},
    {"llm_model": " "}, {"llm_request_timeout_seconds": 0},
])
def test_invalid_snapshot_fails(overrides):
    with TemporaryDirectory(dir=CHECK_ROOT) as directory:
        config = replace(base_config(directory), **overrides)
        with pytest.raises(ModelSettingsError):
            resolve_model_settings(config)


def test_corrupt_saved_settings_fails():
    with TemporaryDirectory(dir=CHECK_ROOT) as directory:
        config = base_config(directory)
        store = ModelSettingsStore(config.storage_root)
        store.path.parent.mkdir()
        store.path.write_text("[", encoding="utf-8")
        with pytest.raises(ModelSettingsError):
            resolve_model_settings(config, store)


def test_example_and_provider_environment():
    from dotenv import dotenv_values

    values = dotenv_values(ROOT / "backend" / ".env.example")
    assert not (set(values) & set(RETIRED_MODEL_VARIABLES))
    result = read_config(**{key: value for key, value in values.items() if value is not None})
    assert result.returncode == 0, result.stderr
    result = read_config(OPENAI_API_KEY="", DASHSCOPE_API_KEY="", DEEPSEEK_API_KEY="")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["llm_backend"] == "vllm"


def test_bound_sdk_clients_preserve_document_settings():
    from app.skill4re_client import Skill4ReClient

    with TemporaryDirectory(dir=CHECK_ROOT) as directory:
        config = replace(base_config(directory), llm_backend="qwen_api")
        store = ModelSettingsStore(config.storage_root)
        factory = Skill4ReClient(config)
        first = factory.for_document(resolve_model_settings(config, store))
        store.save(config.llm_base_url, config.llm_api_key, config.llm_model + "-updated", config.llm_model)
        second = factory.for_document(resolve_model_settings(config, store))
        try:
            assert first.extractor.model == config.llm_model
            assert second.extractor.model == config.llm_model + "-updated"
            assert first.extractor is not second.extractor
            assert first._client.base_url == second._client.base_url
            assert first._client.timeout == config.llm_request_timeout_seconds
        finally:
            first.close()
            second.close()


def test_pipeline_rejects_configuration_before_ocr():
    from app.pipeline import bootstrap_pipeline

    with TemporaryDirectory(dir=CHECK_ROOT) as directory:
        config = replace(base_config(directory), llm_model="")
        pipeline = bootstrap_pipeline(config)
        with pytest.raises(ModelSettingsError, match="LLM_MODEL"):
            pipeline.process_task("invalid-configuration", {})


def test_evaluation_output_keeps_public_saved_settings():
    from app.pipeline import bootstrap_pipeline

    with TemporaryDirectory(dir=CHECK_ROOT) as directory:
        config = base_config(Path(directory) / "public")
        store = ModelSettingsStore(config.storage_root)
        store.save(config.llm_base_url, config.llm_api_key, "qwen3.8-flash", config.llm_model)
        evaluation_config = replace(config, storage_root=Path(directory) / "evaluation")
        pipeline = bootstrap_pipeline(evaluation_config, store)
        settings = resolve_model_settings(evaluation_config, pipeline.model_settings_store)
        client = pipeline.relation_extractor.for_document(settings)
        try:
            assert client.extractor.model == "qwen3.8-flash"
            assert not (evaluation_config.storage_root / "settings/model.json").exists()
        finally:
            client.close()
