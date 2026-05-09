from __future__ import annotations

from pathlib import Path

import pytest

from app.config import AppConfig
from app.skill_store import SkillStore, SkillStoreError


def build_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        api_host="127.0.0.1",
        api_port=5000,
        redis_url="redis://localhost:6379/0",
        storage_root=tmp_path,
        paddle_ocr_mode="python_api",
        paddle_ocr_base_url="http://localhost:8080",
        paddle_ocr_server_url="http://localhost:8118/v1",
        paddle_ocr_api_model_name="PaddleOCR-VL-1.5-0.9B",
        paddle_ocr_api_key="EMPTY",
        paddle_ocr_timeout_seconds=10,
        paddle_ocr_file_mode="base64",
        vllm_base_url="http://localhost:8001/v1",
        vllm_api_key="EMPTY",
        vllm_model="qwen",
        vllm_timeout_seconds=10,
        vllm_enable_thinking=False,
        ocr_concurrency=1,
        llm_concurrency=2,
        max_chunk_chars=1200,
        sentence_stage_limit=5,
        page_stage_limit=5,
        skill4re_backend="vllm",
        skill4re_model="qwen",
        skill4re_skills_dir=tmp_path / "skills",
        skill4re_route_cache_path=tmp_path / "skill4re_route_cache.json",
        skill4re_chunk_trigger=1200,
        skill4re_chunk_budget=900,
        skill4re_max_workers=2,
        skill4re_fast_mode=False,
        skill4re_skip_coref=False,
        worker_poll_seconds=1,
        debug=False,
    )


def build_skill(name: str = "demo_skill") -> dict:
    return {
        "name": name,
        "description": "领域描述",
        "focus": "关注重点",
        "head_prior": "head 实体类型",
        "tail_prior": "tail 实体类型",
        "relation_style": "关系词风格",
        "negative_scope": "不应抽取的范围",
        "extraction_rules": ["规则1"],
        "keywords": ["关键词1"],
        "fewshot": [
            {
                "text": "示例文本",
                "json": '{"relation_list":[{"head":"A","relation":"关联","tail":"B","evidence":"A关联B","skill":"demo_skill"}]}',
                "is_document_level": True,
            }
        ],
    }


def test_skill_store_creates_and_updates_skill(tmp_path: Path) -> None:
    store = SkillStore(build_config(tmp_path))

    created = store.create_skill(build_skill())
    assert created["name"] == "demo_skill"
    assert (tmp_path / "skills" / "demo_skill.json").exists()

    updated_payload = build_skill("renamed_skill")
    updated_payload["keywords"].append("关键词2")
    updated = store.update_skill("demo_skill", updated_payload)

    assert updated["name"] == "renamed_skill"
    assert updated["keywords"] == ["关键词1", "关键词2"]
    assert not (tmp_path / "skills" / "demo_skill.json").exists()
    assert (tmp_path / "skills" / "renamed_skill.json").exists()


def test_skill_store_rejects_invalid_fewshot_json(tmp_path: Path) -> None:
    store = SkillStore(build_config(tmp_path))
    payload = build_skill()
    payload["fewshot"][0]["json"] = '{"not_relation_list":[]}'

    with pytest.raises(SkillStoreError):
        store.create_skill(payload)
