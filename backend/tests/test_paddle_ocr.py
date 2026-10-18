from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.config import AppConfig
from app.paddle_ocr import PaddleOcrClient


def build_config(tmp_path: Path) -> AppConfig:
    return replace(AppConfig.from_env(tmp_path / "absent.env"), storage_root=tmp_path)


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


def test_extract_pages_reads_restructured_blocks_as_paragraphs(tmp_path: Path) -> None:
    client = PaddleOcrClient(build_config(tmp_path))
    result = {
        "page_count": 2,
        "parsing_res_list": [
            {
                "block_label": "doc_title",
                "block_content": "投资协议书",
                "block_bbox": [10, 10, 100, 40],
                "block_id": 0,
                "global_block_id": 0,
                "group_id": 0,
            },
            {
                "block_label": "number",
                "block_content": "1",
                "block_bbox": [10, 100, 30, 120],
                "block_id": 1,
                "global_block_id": 1,
                "group_id": 1,
            },
            {
                "block_label": "text",
                "block_content": "甲方：宜宾市叙州区人民政府",
                "block_bbox": [10, 60, 300, 90],
                "block_id": 2,
                "global_block_id": 2,
                "group_id": 2,
            },
            {
                "block_label": "text",
                "block_content": "乙方：江苏沃宏装备有限公司",
                "block_bbox": [10, 10, 300, 40],
                "block_id": 0,
                "global_block_id": 3,
                "group_id": 0,
            },
        ],
    }

    pages = client.extract_pages(result)
    restructured = client.normalize_restructured_document(result, pages)

    assert len(pages) == 2
    assert pages[0].markdown_text == "投资协议书\n\n甲方：宜宾市叙州区人民政府"
    assert pages[1].markdown_text == "乙方：江苏沃宏装备有限公司"
    assert restructured.markdown_text == (
        "投资协议书\n\n甲方：宜宾市叙州区人民政府\n\n乙方：江苏沃宏装备有限公司"
    )
    assert restructured.layout_parsing_results[0]["page"] == 1
    assert restructured.layout_parsing_results[-1]["page"] == 2
