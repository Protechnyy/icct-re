from __future__ import annotations

from pathlib import Path

from app.config import AppConfig
from app.pipeline import (
    DocumentPipeline,
    _build_relation_batches,
    _build_relation_sections,
    _strip_markdown_image_content,
)


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

    def build_restructure_payload(self, ocr_pages, layout_result=None):
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
    def __init__(self) -> None:
        self.inputs = []

    def extract_document(self, document_text: str):
        self.inputs.append(document_text)
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


class ImageMarkdownOcrClient(FakeOcrClient):
    def extract_pages(self, layout_result):
        from app.types import OcrPage

        return [
            OcrPage(
                page_index=1,
                pruned_result={"page": 1},
                markdown_text=(
                    "Alice joined ACME.\n\n"
                    "![scan](data:image/png;base64,AAAA)\n\n"
                    "<img src=\"data:image/png;base64,BBBB\" />"
                ),
            ),
            OcrPage(page_index=2, pruned_result={"page": 2}, markdown_text="She manages Project Atlas."),
        ]

    def normalize_restructured_document(self, response, fallback_pages):
        from app.types import RestructuredDocument

        return RestructuredDocument(
            markdown_text=(
                "Alice joined ACME.\n\n"
                "![diagram](figures/page-1.png)\n\n"
                "<img alt=\"chart\" src=\"data:image/png;base64,CCCC\" />\n\n"
                "She manages Project Atlas."
            ),
            layout_parsing_results=[],
        )


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
    assert list(result["final_relations"][0].keys())[:5] == ["head", "relation", "tail", "evidence", "skill"]
    assert result["final_relations"][0]["source_sections"] == ["section-001"]
    assert result["relation_split_config"]["split_mode"] == "small_section"
    assert result["relation_split_config"]["batch_size"] == 1
    assert result["ocr_summary"]["relation_batch_count"] == 1
    assert result["relation_batches"][0]["section_ids"] == ["section-001"]
    assert list(result["stage_outputs"]["prediction"]["relation_list"][0].keys())[:5] == [
        "head",
        "relation",
        "tail",
        "evidence",
        "skill",
    ]
    assert result["document_meta"]["extractor"] == "skill4re"
    assert result["document_text"].startswith("Alice joined ACME")


def test_pipeline_strips_markdown_images_before_relation_extraction(tmp_path: Path) -> None:
    extractor = FakeRelationExtractor()
    pipeline = DocumentPipeline(build_config(tmp_path), FakeTaskStore(), ImageMarkdownOcrClient(), extractor)
    file_path = tmp_path / "demo.pdf"
    file_path.write_text("dummy", encoding="utf-8")

    result = pipeline.process_task(
        "task-1",
        {"file_path": str(file_path), "filename": "demo.pdf", "file_type": 0},
    )

    extracted_input = "\n\n".join(extractor.inputs)
    assert "Alice joined ACME." in extracted_input
    assert "She manages Project Atlas." in extracted_input
    assert "![" not in extracted_input
    assert "<img" not in extracted_input
    assert "data:image" not in extracted_input
    assert "figures/page-1.png" not in extracted_input
    assert "![" not in result["document_text"]
    assert "![diagram]" in result["ocr_restructured"]["markdown_text"]


def test_strip_markdown_image_content_removes_common_image_markup() -> None:
    markdown = (
        "甲方签署协议。\n\n"
        "![扫描件](data:image/png;base64,AAAA)\n\n"
        "<img src=\"attachment:image-1.png\" alt=\"scan\" />\n\n"
        "![流程图][flow]\n\n"
        "[flow]: images/flow.png\n\n"
        "乙方负责交付。"
    )

    cleaned = _strip_markdown_image_content(markdown)

    assert cleaned == "甲方签署协议。\n\n乙方负责交付。"


def relation_config(split_mode: str, batch_size: int = 1, max_batch_tokens: int = 2500) -> dict:
    return {
        "split_mode": split_mode,
        "batch_size": batch_size,
        "max_batch_tokens": max_batch_tokens,
        "include_parent_title": True,
    }


def sample_numbered_document() -> str:
    return (
        "一、战区背景与态势\n\n"
        "1.1 第一小节\n\n"
        "A 单位抵达甲地。\n\n"
        "1.2 第二小节\n\n"
        "B 单位支援乙地。\n\n"
        "1.3 第三小节\n\n"
        "C 单位保障丙地。\n\n"
        "二、任务目标\n\n"
        "2.1 第一目标\n\n"
        "D 单位控制丁地。"
    )


def test_relation_small_section_defaults_to_one_section_per_batch() -> None:
    sections = _build_relation_sections(sample_numbered_document())

    batches = _build_relation_batches(sections, relation_config("small_section"))

    assert [batch["section_ids"] for batch in batches] == [["1.1"], ["1.2"], ["1.3"], ["2.1"]]
    assert batches[0]["split_mode"] == "small_section"
    assert batches[0]["parent_title"] == "一、战区背景与态势"
    assert batches[0]["text"].startswith("一、战区背景与态势\n\n1.1 第一小节")


def test_relation_chapter_mode_groups_numbered_sections() -> None:
    sections = _build_relation_sections(sample_numbered_document())

    batches = _build_relation_batches(sections, relation_config("chapter"))

    assert [batch["section_ids"] for batch in batches] == [["1.1", "1.2", "1.3"], ["2.1"]]
    assert batches[0]["split_mode"] == "chapter"
    assert batches[0]["parent_title"] == "一、战区背景与态势"


def test_relation_fixed_sections_batch_size_two_is_preserved() -> None:
    sections = _build_relation_sections(sample_numbered_document())

    batches = _build_relation_batches(sections, relation_config("fixed_sections", batch_size=2))

    assert [batch["section_ids"] for batch in batches] == [["1.1", "1.2"], ["1.3", "2.1"]]
    assert all(batch["split_mode"] == "fixed_sections" for batch in batches)


def test_relation_batches_auto_split_when_token_limit_is_exceeded() -> None:
    markdown = (
        "一、超长章节\n\n"
        "1.1 超长小节\n\n"
        f"{'甲' * 80}\n\n"
        "1.2 短小节\n\n"
        "乙方抵达。"
    )
    sections = _build_relation_sections(markdown)

    batches = _build_relation_batches(sections, relation_config("chapter", max_batch_tokens=10))

    assert all(batch["estimated_tokens"] <= 10 for batch in batches)
    assert any(batch["split_mode"] == "hard_split" for batch in batches)
    assert any(batch["section_ids"] == ["1.2"] for batch in batches)
