from __future__ import annotations

from pathlib import Path

from app.config import AppConfig
from app.paddle_ocr import PaddleOcrClient


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


def test_extract_pages_reads_markdown_variants(tmp_path: Path) -> None:
    client = PaddleOcrClient(build_config(tmp_path))
    result = {
        "layoutParsingResults": [
            {
                "prunedResult": {"blocks": []},
                "markdown": {"markdownText": "第一页内容", "images": {"img": "x"}},
            },
            {
                "prunedResult": {"blocks": []},
                "markdown": {"text": "第二页内容"},
            },
        ]
    }

    pages = client.extract_pages(result)

    assert len(pages) == 2
    assert pages[0].markdown_text == "第一页内容"
    assert pages[0].markdown_images == {"img": "x"}
    assert pages[1].markdown_text == "第二页内容"


def test_normalize_restructured_document_falls_back_to_page_text(tmp_path: Path) -> None:
    client = PaddleOcrClient(build_config(tmp_path))
    pages = client.extract_pages(
        {
            "layoutParsingResults": [
                {"prunedResult": {}, "markdown": {"markdownText": "A"}},
                {"prunedResult": {}, "markdown": {"markdownText": "B"}},
            ]
        }
    )

    restructured = client.normalize_restructured_document({}, pages)

    assert restructured.markdown_text == "A\n\nB"

