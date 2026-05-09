from __future__ import annotations

from pathlib import Path

from app.config import AppConfig
from app.pipeline import DocumentPipeline


class FakeTaskStore:
    def __init__(self) -> None:
        self.status_updates = []
        self.result = None

    def update_task(self, task_id: str, **updates):
        self.status_updates.append((task_id, updates))
        return updates

    def set_result(self, task_id: str, result):
        self.result = result


class FakeOcrClient:
    def layout_parse(self, file_path: Path, file_type: int):
        return {
            "layoutParsingResults": [
                {"prunedResult": {"page": 1}, "markdown": {"markdownText": "Alice joined ACME."}},
                {"prunedResult": {"page": 2}, "markdown": {"markdownText": "She manages Project Atlas."}},
            ]
        }

    def extract_pages(self, layout_result):
        from app.types import OcrPage

        return [
            OcrPage(page_index=1, pruned_result={"page": 1}, markdown_text="Alice joined ACME."),
            OcrPage(page_index=2, pruned_result={"page": 2}, markdown_text="She manages Project Atlas."),
        ]

    def build_restructure_payload(self, ocr_pages):
        return [{"prunedResult": page.pruned_result, "markdownImages": None} for page in ocr_pages]

    def restructure_pages(self, pages):
        raise RuntimeError("restructure down")

    def normalize_restructured_document(self, response, fallback_pages):
        from app.types import RestructuredDocument

        return RestructuredDocument(
            markdown_text="Alice joined ACME.\n\nShe manages Project Atlas.",
            layout_parsing_results=[],
        )


class FakeRelationExtractor:
    def extract_document(self, document_text: str):
        return {
            "preprocess": {
                "doc_token_count": len(document_text),
                "used_chunking": False,
                "chunk_count": 1,
                "chunk_trigger": 1200,
                "chunk_budget": 900,
                "coref_resolved": False,
                "coref_entity_groups": 0,
            },
            "routing": {
                "selected_skills": ["technology"],
                "document_selected_skills": ["technology"],
                "scores": {"technology": 1.0},
                "router_reason": "test",
                "router_mode": "test",
                "cache_hit": False,
                "chunk_routes": [],
            },
            "timing": {
                "routing_seconds": 0.0,
                "coref_seconds": 0.0,
                "chunk_routing_seconds": 0.0,
                "extraction_seconds": 0.0,
                "summarize_seconds": 0.0,
                "proofreading_seconds": 0.0,
                "refinement_seconds": 0.0,
                "domain_reflection_seconds": 0.0,
                "total_seconds": 0.0,
            },
            "proofreading": {"input_relation_count": 2, "output_relation_count": 2},
            "domain_reflection": {"input_relation_count": 2, "output_relation_count": 2},
            "chunk_predictions": [],
            "prediction": {
                "relation_list": [
                    {
                        "evidence": "Alice joined ACME.",
                        "head": "Alice",
                        "relation": "works_for",
                        "skill": "technology",
                        "tail": "ACME",
                    },
                    {
                        "head": "Alice",
                        "relation": "manages",
                        "tail": "Project Atlas",
                        "evidence": "She manages Project Atlas.",
                        "skill": "technology",
                    },
                ]
            },
        }


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


def test_pipeline_falls_back_when_restructure_fails(tmp_path: Path) -> None:
    pipeline = DocumentPipeline(build_config(tmp_path), FakeTaskStore(), FakeOcrClient(), FakeRelationExtractor())
    file_path = tmp_path / "demo.pdf"
    file_path.write_text("dummy", encoding="utf-8")

    result = pipeline.process_task(
        "task-1",
        {"file_path": str(file_path), "filename": "demo.pdf", "file_type": 0},
    )

    assert result["ocr_summary"]["ocr_restructure_fallback"] is True
    assert len(result["final_relations"]) == 2
    assert result["final_relation_list"]["relation_list"][0]["head"] == "Alice"
    assert list(result["final_relations"][0].keys()) == ["head", "relation", "tail", "evidence", "skill"]
    assert list(result["stage_outputs"]["prediction"]["relation_list"][0].keys()) == [
        "head",
        "relation",
        "tail",
        "evidence",
        "skill",
    ]
    assert result["document_meta"]["extractor"] == "skill4re"
    assert result["document_text"].startswith("Alice joined ACME")
