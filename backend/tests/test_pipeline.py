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


class FakeVllmClient:
    def extract_relations(self, text: str, context: str):
        if "ACME" in text:
            return [{"subject": "Alice", "relation": "works_for", "object": "ACME"}]
        if "Project Atlas" in text:
            return [{"subject": "Alice", "relation": "manages", "object": "Project Atlas"}]
        return []


def build_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        api_host="127.0.0.1",
        api_port=5000,
        redis_url="redis://localhost:6379/0",
        storage_root=tmp_path,
        paddle_ocr_base_url="http://localhost:8080",
        paddle_ocr_timeout_seconds=10,
        paddle_ocr_file_mode="base64",
        vllm_base_url="http://localhost:8001/v1",
        vllm_api_key="EMPTY",
        vllm_model="qwen",
        vllm_timeout_seconds=10,
        ocr_concurrency=1,
        llm_concurrency=2,
        max_chunk_chars=1200,
        sentence_stage_limit=5,
        page_stage_limit=5,
        worker_poll_seconds=1,
        debug=False,
    )


def test_pipeline_falls_back_when_restructure_fails(tmp_path: Path) -> None:
    pipeline = DocumentPipeline(build_config(tmp_path), FakeTaskStore(), FakeOcrClient(), FakeVllmClient())
    file_path = tmp_path / "demo.pdf"
    file_path.write_text("dummy", encoding="utf-8")

    result = pipeline.process_task(
        "task-1",
        {"file_path": str(file_path), "filename": "demo.pdf", "file_type": 0},
    )

    assert result["ocr_summary"]["ocr_restructure_fallback"] is True
    assert len(result["final_relations"]) == 2
    assert result["document_text"].startswith("Alice joined ACME")

