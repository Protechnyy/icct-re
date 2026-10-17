from dataclasses import replace
from tempfile import TemporaryDirectory

import pytest

from app.model_settings import resolve_model_settings
from app.skill4re_client import Skill4ReClient, Skill4ReClientError
from test_shared_llm_configuration import CHECK_ROOT, base_config


def test_qwen_sdk_uses_document_snapshot_url():
    with TemporaryDirectory(dir=CHECK_ROOT) as directory:
        config = replace(base_config(directory), llm_backend="qwen_api")
        config = replace(config, llm_base_url=config.llm_base_url + "/chat/completions")
        settings = resolve_model_settings(config)
        client = Skill4ReClient(config).for_document(settings)
        try:
            assert str(client._client.base_url).rstrip("/") == settings.base_url
            assert client.extractor.base_url == settings.base_url
            assert client.extractor.model == settings.model
        finally:
            client.close()


def test_extraction_requires_bound_document_settings():
    with TemporaryDirectory(dir=CHECK_ROOT) as directory:
        client = Skill4ReClient(base_config(directory))
        with pytest.raises(Skill4ReClientError, match="文档模型配置尚未绑定"):
            client.extract_document("")
