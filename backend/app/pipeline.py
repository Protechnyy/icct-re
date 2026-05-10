from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import (
    DEFAULT_RELATION_BATCH_SIZE,
    DEFAULT_RELATION_INCLUDE_PARENT_TITLE,
    DEFAULT_RELATION_MAX_BATCH_TOKENS,
    DEFAULT_RELATION_SPLIT_MODE,
    RELATION_SPLIT_MODES,
    AppConfig,
)
from .paddle_ocr import PaddleOcrClient
from .skill4re_client import RelationExtractor, Skill4ReClient
from .types import Chunk
from .utils import chunk_text_with_page_map, utcnow_iso

if TYPE_CHECKING:
    from .task_store import RedisTaskStore


RELATION_FIELD_ORDER = ("head", "relation", "tail", "evidence", "skill")
NUMBERED_SECTION_RE = re.compile(r"^(?P<section_id>\d+(?:[.．]\d+)+)\s*(?P<title>.*)$")
CHINESE_SECTION_RE = re.compile(r"^(?P<section_id>[一二三四五六七八九十百千万]+)、")


class DocumentPipeline:
    def __init__(
        self,
        config: AppConfig,
        task_store: RedisTaskStore,
        ocr_client: PaddleOcrClient,
        relation_extractor: RelationExtractor,
    ) -> None:
        self.config = config
        self.task_store = task_store
        self.ocr_client = ocr_client
        self.relation_extractor = relation_extractor

    def process_task(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        file_path = Path(payload["file_path"])
        filename = payload["filename"]
        file_type = int(payload["file_type"])

        self.task_store.update_task(task_id, status="ocr_running", stage="layout_parsing", progress=10, error=None)
        layout_result = self.ocr_client.layout_parse(file_path=file_path, file_type=file_type)
        ocr_pages = self.ocr_client.extract_pages(layout_result)
        if not ocr_pages:
            raise RuntimeError(f"OCR parsed 0 pages from {filename}; check the uploaded file type and PaddleOCR logs.")

        self.task_store.update_task(task_id, stage="restructure_pages", progress=30)
        restructure_fallback = False
        try:
            restructure_payload = self.ocr_client.build_restructure_payload(ocr_pages, layout_result=layout_result)
            restructure_result = self.ocr_client.restructure_pages(restructure_payload)
            restructured = self.ocr_client.normalize_restructured_document(restructure_result, ocr_pages)
        except Exception:
            restructure_fallback = True
            restructured = self.ocr_client.normalize_restructured_document({}, ocr_pages)

        page_texts = [(page.page_index, page.markdown_text) for page in ocr_pages]
        document_text = restructured.markdown_text.strip() or "\n\n".join(text for _, text in page_texts if text.strip())
        chunks = [Chunk(**chunk) for chunk in chunk_text_with_page_map(page_texts, self.config.max_chunk_chars)]
        relation_split_config = _relation_split_config(self.config, payload)
        relation_sections = _build_relation_sections(
            document_text,
            include_parent_title=bool(relation_split_config["include_parent_title"]),
        )

        self.task_store.update_task(task_id, status="extracting", stage="relation_extraction", progress=55)
        skill4re_result, stage_outputs, final_relations, relation_batches = self._extract_relations(
            document_text,
            relation_sections,
            relation_split_config,
        )

        self.task_store.update_task(task_id, status="merging", stage="document_merge", progress=85)
        safe_layout_result = {k: v for k, v in layout_result.items() if k != "_pages_res"}
        result = {
            "document_meta": {
                "task_id": task_id,
                "filename": filename,
                "page_count": len(ocr_pages),
                "processed_at": utcnow_iso(),
                "model": self.config.vllm_model,
                "extractor": "skill4re",
                "skill4re_backend": self.config.skill4re_backend,
                "skill4re_model": self.config.skill4re_model,
            },
            "ocr_summary": {
                "ocr_restructure_fallback": restructure_fallback,
                "document_text_length": len(document_text),
                "chunk_count": len(chunks),
                "relation_section_count": len(relation_sections),
                "relation_batch_count": len(relation_batches),
                "relation_count": len(final_relations),
                "selected_skills": skill4re_result.get("routing", {}).get("selected_skills", []),
            },
            "ocr_raw": safe_layout_result,
            "ocr_pages": [page.to_dict() for page in ocr_pages],
            "ocr_restructured": restructured.to_dict(),
            "document_text": document_text,
            "chunks": [chunk.to_dict() for chunk in chunks],
            "relation_split_config": relation_split_config,
            "relation_sections": relation_sections,
            "relation_batches": [_public_relation_batch(batch) for batch in relation_batches],
            "stage_outputs": stage_outputs,
            "skill4re_result": skill4re_result,
            "final_relations": final_relations,
            "final_relation_list": {"relation_list": final_relations},
        }
        result_dir = self.config.storage_root / "results" / task_id
        saved_paths = _result_file_paths(result_dir)
        result["document_meta"]["result_dir"] = str(result_dir)
        result["saved_paths"] = saved_paths
        _save_result_files(result_dir, result)
        self.task_store.set_result(task_id, result)
        self.task_store.update_task(task_id, status="succeeded", stage="completed", progress=100, error=None)
        return result

    def _extract_relations(
        self,
        document_text: str,
        relation_sections: list[dict[str, Any]],
        relation_split_config: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        relation_batches = _build_relation_batches(relation_sections, relation_split_config)
        if not relation_batches:
            relation_batches = _build_document_relation_batches(document_text, relation_split_config)

        batch_results: list[dict[str, Any]] = []
        all_relations: list[dict[str, Any]] = []
        for batch in relation_batches:
            batch_result = _order_relation_payload(self.relation_extractor.extract_document(batch["text"]))
            prediction = batch_result.get("prediction", {})
            relation_list = prediction.get("relation_list", []) if isinstance(prediction, dict) else []
            batch_relations = [
                _attach_relation_source(_order_relation_item(item), batch)
                for item in relation_list
                if isinstance(item, dict)
            ]
            all_relations.extend(batch_relations)
            batch_results.append(
                {
                    "batch_index": batch["batch_index"],
                    "split_mode": batch["split_mode"],
                    "section_ids": batch["section_ids"],
                    "parent_title": batch["parent_title"],
                    "page_start": batch["page_start"],
                    "page_end": batch["page_end"],
                    "block_ids": batch["block_ids"],
                    "estimated_tokens": batch["estimated_tokens"],
                    "result": batch_result,
                    "relations": batch_relations,
                }
            )

        final_relations = _dedupe_relations(all_relations)
        skill4re_result = _combine_batch_skill4re_results(
            batch_results,
            final_relations,
            relation_sections,
            relation_split_config,
        )
        prediction = skill4re_result.get("prediction", {})
        stage_outputs = {
            "routing": skill4re_result.get("routing", {}),
            "preprocess": skill4re_result.get("preprocess", {}),
            "timing": skill4re_result.get("timing", {}),
            "proofreading": skill4re_result.get("proofreading", {}),
            "chunk_predictions": skill4re_result.get("chunk_predictions", []),
            "batch_results": batch_results,
            "prediction": prediction,
        }
        return skill4re_result, stage_outputs, final_relations, relation_batches


def bootstrap_pipeline(config: AppConfig) -> DocumentPipeline:
    from .task_store import RedisTaskStore

    task_store = RedisTaskStore(config.redis_url)
    ocr_client = PaddleOcrClient(config)
    relation_extractor = Skill4ReClient(config)
    return DocumentPipeline(config, task_store, ocr_client, relation_extractor)


def _order_relation_payload(value: Any) -> Any:
    if isinstance(value, list):
        return [_order_relation_payload(item) for item in value]
    if isinstance(value, dict):
        if {"head", "relation", "tail"}.issubset(value):
            return _order_relation_item(value)
        return {key: _order_relation_payload(item) for key, item in value.items()}
    return value


def _order_relation_item(item: dict[str, Any]) -> dict[str, Any]:
    ordered = {field: item[field] for field in RELATION_FIELD_ORDER if field in item}
    for key, value in item.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _relation_split_config(config: AppConfig, payload: dict[str, Any]) -> dict[str, Any]:
    payload_config = payload.get("relation_split_config") if isinstance(payload.get("relation_split_config"), dict) else {}
    split_mode = _clean_relation_split_mode(
        _first_config_value(payload_config, payload, ("split_mode", "relation_split_mode"), config.relation_split_mode),
        config.relation_split_mode,
    )
    batch_size = _positive_int_config_value(
        _first_config_value(payload_config, payload, ("batch_size", "relation_batch_size"), config.relation_batch_size),
        config.relation_batch_size,
    )
    if split_mode != "fixed_sections":
        batch_size = 1
    max_batch_tokens = _positive_int_config_value(
        _first_config_value(
            payload_config,
            payload,
            ("max_batch_tokens", "relation_max_batch_tokens"),
            config.relation_max_batch_tokens,
        ),
        config.relation_max_batch_tokens,
    )
    include_parent_title = _bool_config_value(
        _first_config_value(
            payload_config,
            payload,
            ("include_parent_title", "relation_include_parent_title"),
            config.relation_include_parent_title,
        ),
        config.relation_include_parent_title,
    )
    return {
        "split_mode": split_mode,
        "batch_size": batch_size,
        "max_batch_tokens": max_batch_tokens,
        "include_parent_title": include_parent_title,
    }


def _first_config_value(
    payload_config: dict[str, Any],
    payload: dict[str, Any],
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    for source in (payload_config, payload):
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return value
    return default


def _clean_relation_split_mode(value: Any, default: str = DEFAULT_RELATION_SPLIT_MODE) -> str:
    mode = str(value or default).strip()
    fallback = default if default in RELATION_SPLIT_MODES else DEFAULT_RELATION_SPLIT_MODE
    return mode if mode in RELATION_SPLIT_MODES else fallback


def _positive_int_config_value(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def _bool_config_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_relation_sections(markdown_text: str, include_parent_title: bool = True) -> list[dict[str, Any]]:
    paragraphs = _markdown_paragraphs(markdown_text)
    if not paragraphs:
        return []
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    parent_title: str | None = None
    for index, paragraph in enumerate(paragraphs):
        if _is_markdown_title(paragraph):
            if _is_chinese_markdown_title(paragraph) and _next_markdown_title_is_numbered(paragraphs, index):
                if current is not None:
                    sections.append(_finalize_relation_section(current))
                    current = None
                parent_title = paragraph
                continue

            if current is not None:
                sections.append(_finalize_relation_section(current))
            section_parent_title = parent_title if parent_title and _is_numbered_markdown_title(paragraph) else ""
            current = {
                "section_id": _section_id_from_title(paragraph, len(sections) + 1),
                "title": paragraph,
                "parent_title": section_parent_title,
                "paragraphs": [_markdown_paragraph(section_parent_title)]
                if section_parent_title and include_parent_title
                else [],
            }
            if not _is_numbered_markdown_title(paragraph):
                parent_title = None
        elif current is None:
            prefix_paragraphs = [_markdown_paragraph(parent_title)] if parent_title and include_parent_title else []
            current = {
                "section_id": _section_id_from_title(parent_title, len(sections) + 1)
                if parent_title
                else f"section-{len(sections) + 1:03d}",
                "title": parent_title if parent_title else "",
                "parent_title": "",
                "paragraphs": prefix_paragraphs,
            }
            parent_title = None
        current["paragraphs"].append(_markdown_paragraph(paragraph))

    if current is not None:
        sections.append(_finalize_relation_section(current))
    return [section for section in sections if section["text"]]


def _markdown_paragraphs(markdown_text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n+", str(markdown_text or "")) if part.strip()]


def _markdown_paragraph(content: str | None) -> dict[str, Any]:
    return {
        "page": None,
        "label": "markdown",
        "content": str(content or "").strip(),
        "bbox": None,
        "global_block_id": None,
    }


def _is_markdown_title(paragraph: str) -> bool:
    content = paragraph.strip()
    return bool(_is_doc_markdown_title(content) or _is_numbered_markdown_title(content) or _is_chinese_markdown_title(content))


def _is_doc_markdown_title(paragraph: str) -> bool:
    content = paragraph.strip()
    return content.startswith("《") and "》" in content and len(content) <= 80


def _is_numbered_markdown_title(paragraph: str) -> bool:
    return bool(NUMBERED_SECTION_RE.match(paragraph.strip()))


def _is_chinese_markdown_title(paragraph: str) -> bool:
    content = paragraph.strip()
    return bool(CHINESE_SECTION_RE.match(content)) and not _is_numbered_markdown_title(content)


def _next_markdown_title_is_numbered(paragraphs: list[str], index: int) -> bool:
    for paragraph in paragraphs[index + 1 :]:
        if not paragraph.strip():
            continue
        return _is_numbered_markdown_title(paragraph)
    return False


def _section_id_from_title(title: str | None, fallback_index: int) -> str:
    stripped = str(title or "").strip()
    number_match = NUMBERED_SECTION_RE.match(stripped)
    if number_match:
        return number_match.group("section_id").replace("．", ".")
    chinese_match = CHINESE_SECTION_RE.match(stripped)
    if chinese_match:
        return chinese_match.group("section_id")
    if stripped.startswith("《") and "》" in stripped:
        return "doc-title"
    return f"section-{fallback_index:03d}"


def _finalize_relation_section(section: dict[str, Any]) -> dict[str, Any]:
    paragraphs = [item for item in section.get("paragraphs", []) if item.get("content")]
    pages = [_safe_int(item.get("page"), 1) for item in paragraphs]
    block_ids = [
        _safe_int(item.get("global_block_id"), None)
        for item in paragraphs
        if item.get("global_block_id") is not None
    ]
    return {
        "section_id": section["section_id"],
        "title": section.get("title") or "",
        "parent_title": section.get("parent_title") or "",
        "page_start": min(pages) if pages else 1,
        "page_end": max(pages) if pages else 1,
        "block_ids": block_ids,
        "text": "\n\n".join(str(item["content"]).strip() for item in paragraphs if item.get("content")).strip(),
        "paragraphs": paragraphs,
    }


def _build_relation_batches(sections: list[dict[str, Any]], split_config: dict[str, Any] | int) -> list[dict[str, Any]]:
    if isinstance(split_config, int):
        split_config = {
            "split_mode": "fixed_sections",
            "batch_size": split_config,
            "max_batch_tokens": DEFAULT_RELATION_MAX_BATCH_TOKENS,
            "include_parent_title": DEFAULT_RELATION_INCLUDE_PARENT_TITLE,
        }
    split_mode = _clean_relation_split_mode(split_config.get("split_mode"), DEFAULT_RELATION_SPLIT_MODE)
    batch_size = _positive_int_config_value(split_config.get("batch_size"), DEFAULT_RELATION_BATCH_SIZE)
    max_batch_tokens = _positive_int_config_value(
        split_config.get("max_batch_tokens"),
        DEFAULT_RELATION_MAX_BATCH_TOKENS,
    )

    if split_mode == "chapter":
        batches = _build_chapter_relation_batches(sections)
    elif split_mode == "paragraph":
        batches = _build_paragraph_relation_batches(sections)
    elif split_mode == "fixed_sections":
        batches = _build_section_relation_batches(sections, batch_size, "fixed_sections")
    else:
        batches = _build_section_relation_batches(sections, 1, "small_section")

    guarded_batches: list[dict[str, Any]] = []
    for batch in batches:
        guarded_batches.extend(_guard_relation_batch_tokens(batch, max_batch_tokens))
    return _index_relation_batches(guarded_batches)


def _build_document_relation_batches(document_text: str, split_config: dict[str, Any]) -> list[dict[str, Any]]:
    batch = {
        "batch_index": 0,
        "split_mode": _clean_relation_split_mode(split_config.get("split_mode"), DEFAULT_RELATION_SPLIT_MODE),
        "section_ids": ["document"],
        "parent_title": "",
        "page_start": 1,
        "page_end": 1,
        "block_ids": [],
        "text": str(document_text or "").strip(),
        "sections": [],
        "estimated_tokens": _estimate_relation_tokens(document_text),
    }
    max_batch_tokens = _positive_int_config_value(
        split_config.get("max_batch_tokens"),
        DEFAULT_RELATION_MAX_BATCH_TOKENS,
    )
    return _index_relation_batches(_guard_relation_batch_tokens(batch, max_batch_tokens))


def _build_section_relation_batches(
    sections: list[dict[str, Any]],
    batch_size: int,
    split_mode: str,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for start in range(0, len(sections), batch_size):
        batch_sections = sections[start : start + batch_size]
        batches.append(_make_relation_batch(batch_sections, split_mode))
    return [batch for batch in batches if batch.get("text")]


def _build_chapter_relation_batches(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for chapter_title, chapter_sections in _chapter_section_groups(sections):
        batches.append(_make_relation_batch(chapter_sections, "chapter", parent_title=chapter_title))
    return [batch for batch in batches if batch.get("text")]


def _chapter_section_groups(sections: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    current_key: tuple[str, str] | None = None
    current_title = ""
    current_sections: list[dict[str, Any]] = []

    for section in sections:
        parent_title = str(section.get("parent_title") or "").strip()
        if parent_title:
            key = ("parent", parent_title)
            chapter_title = parent_title
        else:
            section_id = str(section.get("section_id") or "")
            key = ("section", section_id)
            chapter_title = str(section.get("title") or "").strip()

        if current_key is not None and key != current_key:
            groups.append((current_title, current_sections))
            current_sections = []
        current_key = key
        current_title = chapter_title
        current_sections.append(section)

    if current_sections:
        groups.append((current_title, current_sections))
    return groups


def _build_paragraph_relation_batches(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for section in sections:
        batches.extend(_paragraph_batches_for_section(section))
    return batches


def _paragraph_batches_for_section(section: dict[str, Any]) -> list[dict[str, Any]]:
    context_paragraphs, body_paragraphs = _section_context_and_body_paragraphs(section)
    if not body_paragraphs:
        body_paragraphs = context_paragraphs or list(section.get("paragraphs", []))
        context_paragraphs = []

    batches: list[dict[str, Any]] = []
    for paragraph in body_paragraphs:
        paragraphs = _unique_paragraphs([*context_paragraphs, paragraph])
        text = "\n\n".join(str(item.get("content") or "").strip() for item in paragraphs if item.get("content")).strip()
        if not text:
            continue
        batch_section = _clone_section_with_paragraphs(section, paragraphs, text)
        batches.append(
            _make_relation_batch(
                [batch_section],
                "paragraph",
                text=text,
                parent_title=str(section.get("parent_title") or ""),
            )
        )
    return batches


def _section_context_and_body_paragraphs(
    section: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    title = str(section.get("title") or "").strip()
    parent_title = str(section.get("parent_title") or "").strip()
    context: list[dict[str, Any]] = []
    body: list[dict[str, Any]] = []
    for paragraph in section.get("paragraphs", []):
        content = str(paragraph.get("content") or "").strip()
        if not content:
            continue
        if content in {title, parent_title}:
            context.append(paragraph)
        else:
            body.append(paragraph)
    return context, body


def _guard_relation_batch_tokens(batch: dict[str, Any], max_batch_tokens: int) -> list[dict[str, Any]]:
    batch = _with_estimated_tokens(batch)
    if batch["estimated_tokens"] <= max_batch_tokens:
        return [batch]

    split_mode = str(batch.get("split_mode") or "")
    sections = batch.get("sections", []) if isinstance(batch.get("sections"), list) else []
    if split_mode == "chapter":
        guarded: list[dict[str, Any]] = []
        for next_batch in _build_section_relation_batches(sections, 1, "small_section"):
            guarded.extend(_guard_relation_batch_tokens(next_batch, max_batch_tokens))
        return guarded

    if split_mode == "fixed_sections" and len(sections) > 1:
        guarded = []
        for next_batch in _build_section_relation_batches(sections, 1, "small_section"):
            guarded.extend(_guard_relation_batch_tokens(next_batch, max_batch_tokens))
        return guarded

    if split_mode in {"small_section", "fixed_sections"} and sections:
        guarded = []
        for section in sections:
            for next_batch in _paragraph_batches_for_section(section):
                guarded.extend(_guard_relation_batch_tokens(next_batch, max_batch_tokens))
        return guarded

    if split_mode == "paragraph":
        return _hard_split_relation_batch(batch, max_batch_tokens)

    return _hard_split_relation_batch(batch, max_batch_tokens)


def _hard_split_relation_batch(batch: dict[str, Any], max_batch_tokens: int) -> list[dict[str, Any]]:
    chunks = _hard_split_text(str(batch.get("text") or ""), max_batch_tokens)
    sections = batch.get("sections", []) if isinstance(batch.get("sections"), list) else []
    split_batches: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        if len(sections) == 1:
            chunk_sections = [_clone_section_with_text(sections[0], chunk)]
            chunk_batch = _make_relation_batch(
                chunk_sections,
                "hard_split",
                text=chunk,
                parent_title=str(batch.get("parent_title") or ""),
            )
        else:
            chunk_batch = dict(batch)
            chunk_batch["split_mode"] = "hard_split"
            chunk_batch["text"] = chunk
        chunk_batch["hard_split_index"] = chunk_index
        split_batches.append(_with_estimated_tokens(chunk_batch))
    return split_batches


def _make_relation_batch(
    sections: list[dict[str, Any]],
    split_mode: str,
    text: str | None = None,
    parent_title: str | None = None,
) -> dict[str, Any]:
    batch_text = (
        str(text).strip()
        if text is not None
        else "\n\n".join(str(section.get("text") or "").strip() for section in sections if section.get("text")).strip()
    )
    pages = [
        page
        for section in sections
        for page in (_safe_int(section.get("page_start"), 1), _safe_int(section.get("page_end"), 1))
        if page is not None
    ]
    block_ids = [
        block_id
        for section in sections
        for block_id in section.get("block_ids", [])
        if block_id is not None
    ]
    batch = {
        "batch_index": 0,
        "split_mode": split_mode,
        "section_ids": [str(section["section_id"]) for section in sections],
        "parent_title": parent_title if parent_title is not None else _common_parent_title(sections),
        "page_start": min(pages) if pages else 1,
        "page_end": max(pages) if pages else 1,
        "block_ids": block_ids,
        "text": batch_text,
        "sections": sections,
    }
    return _with_estimated_tokens(batch)


def _common_parent_title(sections: list[dict[str, Any]]) -> str:
    parent_titles = _unique_values(str(section.get("parent_title") or "").strip() for section in sections)
    parent_titles = [title for title in parent_titles if title]
    return parent_titles[0] if len(parent_titles) == 1 else ""


def _clone_section_with_paragraphs(
    section: dict[str, Any],
    paragraphs: list[dict[str, Any]],
    text: str,
) -> dict[str, Any]:
    cloned_paragraphs = [dict(paragraph) for paragraph in paragraphs if paragraph.get("content")]
    pages = [_safe_int(item.get("page"), None) for item in cloned_paragraphs]
    pages = [page for page in pages if page is not None]
    block_ids = [
        _safe_int(item.get("global_block_id"), None)
        for item in cloned_paragraphs
        if item.get("global_block_id") is not None
    ]
    return {
        **section,
        "page_start": min(pages) if pages else _safe_int(section.get("page_start"), 1),
        "page_end": max(pages) if pages else _safe_int(section.get("page_end"), 1),
        "block_ids": [block_id for block_id in block_ids if block_id is not None],
        "text": text,
        "paragraphs": cloned_paragraphs,
    }


def _clone_section_with_text(section: dict[str, Any], text: str) -> dict[str, Any]:
    source_paragraphs = section.get("paragraphs", [])
    paragraph = dict(source_paragraphs[0]) if source_paragraphs else _markdown_paragraph(text)
    paragraph["content"] = text
    return _clone_section_with_paragraphs(section, [paragraph], text)


def _unique_paragraphs(paragraphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen = set()
    for paragraph in paragraphs:
        content = str(paragraph.get("content") or "").strip()
        if not content:
            continue
        marker = (
            content,
            paragraph.get("page"),
            paragraph.get("global_block_id"),
        )
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(paragraph)
    return unique


def _hard_split_text(text: str, max_batch_tokens: int) -> list[str]:
    remaining = str(text or "").strip()
    if not remaining:
        return []
    max_chars = max(1, int(max_batch_tokens) * 2)
    chunks: list[str] = []
    while len(remaining) > max_chars:
        split_at = _best_hard_split_position(remaining, max_chars)
        chunk = remaining[:split_at].strip()
        if not chunk:
            chunk = remaining[:max_chars].strip()
            split_at = max_chars
        chunks.append(chunk)
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _best_hard_split_position(text: str, max_chars: int) -> int:
    minimum = max(1, max_chars // 2)
    for separator in ("\n\n", "\n", "。", "；", "，", " "):
        position = text.rfind(separator, 0, max_chars)
        if position >= minimum:
            return position + len(separator)
    return max_chars


def _with_estimated_tokens(batch: dict[str, Any]) -> dict[str, Any]:
    next_batch = dict(batch)
    next_batch["estimated_tokens"] = _estimate_relation_tokens(next_batch.get("text", ""))
    return next_batch


def _estimate_relation_tokens(text: Any) -> int:
    stripped = str(text or "").strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 1) // 2)


def _index_relation_batches(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: list[dict[str, Any]] = []
    for batch in batches:
        if not str(batch.get("text") or "").strip():
            continue
        next_batch = _with_estimated_tokens(batch)
        next_batch["batch_index"] = len(indexed)
        indexed.append(next_batch)
    return indexed


def _public_relation_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_index": batch["batch_index"],
        "split_mode": batch["split_mode"],
        "section_ids": batch["section_ids"],
        "parent_title": batch["parent_title"],
        "page_start": batch["page_start"],
        "page_end": batch["page_end"],
        "block_ids": batch["block_ids"],
        "estimated_tokens": batch["estimated_tokens"],
        "text": batch["text"],
    }


def _attach_relation_source(relation: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    matches = _match_relation_paragraphs(relation, batch)
    if matches:
        relation["source_sections"] = _unique_values(match["section_id"] for match in matches)
        relation["source_pages"] = _unique_values(match["page"] for match in matches)
        relation["source_blocks"] = _unique_values(match["global_block_id"] for match in matches)
        relation["source_paragraphs"] = matches
    else:
        relation["source_sections"] = batch["section_ids"]
        relation["source_pages"] = _unique_values(
            page
            for section in batch.get("sections", [])
            for page in (section.get("page_start"), section.get("page_end"))
        )
        relation["source_blocks"] = batch["block_ids"]
        relation["source_paragraphs"] = []
    relation["source_batch_index"] = batch["batch_index"]
    return _order_relation_item(relation)


def _match_relation_paragraphs(relation: dict[str, Any], batch: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = str(relation.get("evidence") or "").strip()
    head = str(relation.get("head") or relation.get("subject") or "").strip()
    tail = str(relation.get("tail") or relation.get("object") or "").strip()
    matches: list[dict[str, Any]] = []
    for section in batch.get("sections", []):
        section_text_compact = _compact_text(section.get("text", ""))
        section_matches_evidence = bool(evidence and _compact_text(evidence) in section_text_compact)
        for paragraph in section.get("paragraphs", []):
            content = str(paragraph.get("content") or "")
            compact_content = _compact_text(content)
            matched = bool(evidence and _compact_text(evidence) in compact_content)
            matched = matched or bool(head and tail and head in content and tail in content)
            if not matched and section_matches_evidence and (head in content or tail in content):
                matched = True
            if matched:
                matches.append(
                    {
                        "section_id": section["section_id"],
                        "page": paragraph.get("page"),
                        "global_block_id": paragraph.get("global_block_id"),
                        "bbox": paragraph.get("bbox"),
                        "content": content,
                    }
                )
    return matches


def _dedupe_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for relation in relations:
        key = (
            _normalize_key_part(relation.get("head") or relation.get("subject")),
            _normalize_key_part(relation.get("relation")),
            _normalize_key_part(relation.get("tail") or relation.get("object")),
            _normalize_key_part(relation.get("evidence")),
        )
        if key not in merged:
            merged[key] = _order_relation_item(dict(relation))
            continue
        existing = merged[key]
        for field in ("source_sections", "source_pages", "source_blocks", "source_paragraphs"):
            existing[field] = _merge_list_values(existing.get(field, []), relation.get(field, []))
    return [_order_relation_item(item) for item in merged.values()]


def _combine_batch_skill4re_results(
    batch_results: list[dict[str, Any]],
    final_relations: list[dict[str, Any]],
    relation_sections: list[dict[str, Any]],
    relation_split_config: dict[str, Any],
) -> dict[str, Any]:
    selected_skills = _unique_values(
        skill
        for batch in batch_results
        for skill in batch.get("result", {}).get("routing", {}).get("selected_skills", [])
    )
    document_selected_skills = _unique_values(
        skill
        for batch in batch_results
        for skill in batch.get("result", {}).get("routing", {}).get("document_selected_skills", [])
    )
    return {
        "preprocess": {
            "used_section_batching": True,
            "section_count": len(relation_sections),
            "batch_count": len(batch_results),
            "split_config": relation_split_config,
            "split_mode": relation_split_config["split_mode"],
            "batch_size": relation_split_config["batch_size"],
            "max_batch_tokens": relation_split_config["max_batch_tokens"],
            "batches": [
                {
                    "batch_index": batch["batch_index"],
                    "split_mode": batch["split_mode"],
                    "section_ids": batch["section_ids"],
                    "parent_title": batch["parent_title"],
                    "page_start": batch["page_start"],
                    "page_end": batch["page_end"],
                    "estimated_tokens": batch["estimated_tokens"],
                    "relation_count": len(batch["relations"]),
                }
                for batch in batch_results
            ],
        },
        "routing": {
            "selected_skills": selected_skills,
            "document_selected_skills": document_selected_skills,
            "scores": {
                f"batch-{batch['batch_index']}": batch.get("result", {}).get("routing", {}).get("scores", {})
                for batch in batch_results
            },
            "router_reason": "OCR paragraphs were split into numbered/title sections and extracted in small batches.",
            "router_mode": "section_batched",
            "cache_hit": all(bool(batch.get("result", {}).get("routing", {}).get("cache_hit")) for batch in batch_results),
            "chunk_routes": [
                {
                    "batch_index": batch["batch_index"],
                    "split_mode": batch["split_mode"],
                    "section_ids": batch["section_ids"],
                    "chunk_routes": batch.get("result", {}).get("routing", {}).get("chunk_routes", []),
                }
                for batch in batch_results
            ],
        },
        "timing": _sum_batch_timing(batch_results),
        "proofreading": _sum_batch_proofreading(batch_results),
        "domain_reflection": _sum_batch_proofreading(batch_results),
        "chunk_predictions": [
            {
                "batch_index": batch["batch_index"],
                "split_mode": batch["split_mode"],
                "section_ids": batch["section_ids"],
                "chunk_predictions": batch.get("result", {}).get("chunk_predictions", []),
            }
            for batch in batch_results
        ],
        "batch_results": batch_results,
        "prediction": {"relation_list": final_relations},
    }


def _sum_batch_timing(batch_results: list[dict[str, Any]]) -> dict[str, float]:
    fields = (
        "routing_seconds",
        "coref_seconds",
        "chunk_routing_seconds",
        "extraction_seconds",
        "summarize_seconds",
        "proofreading_seconds",
        "refinement_seconds",
        "domain_reflection_seconds",
        "total_seconds",
    )
    return {
        field: round(
            sum(float(batch.get("result", {}).get("timing", {}).get(field, 0.0) or 0.0) for batch in batch_results),
            4,
        )
        for field in fields
    }


def _sum_batch_proofreading(batch_results: list[dict[str, Any]]) -> dict[str, int]:
    fields = ("input_relation_count", "high_confidence_count", "low_confidence_count", "output_relation_count")
    return {
        field: sum(int(batch.get("result", {}).get("proofreading", {}).get(field, 0) or 0) for batch in batch_results)
        for field in fields
    }


def _safe_int(value: Any, default: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _normalize_key_part(value: Any) -> str:
    return _compact_text(value).lower()


def _unique_values(values: Any) -> list[Any]:
    unique: list[Any] = []
    seen = set()
    for value in values:
        if value is None:
            continue
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict) else str(value)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(value)
    return unique


def _merge_list_values(left: Any, right: Any) -> list[Any]:
    left_values = left if isinstance(left, list) else []
    right_values = right if isinstance(right, list) else []
    return _unique_values([*left_values, *right_values])


def _result_file_paths(result_dir: Path) -> dict[str, str]:
    return {
        "result_json": str(result_dir / "result.json"),
        "document_text_md": str(result_dir / "document_text.md"),
        "ocr_paragraphs_json": str(result_dir / "ocr_paragraphs.json"),
    }


def _save_result_files(result_dir: Path, result: dict[str, Any]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (result_dir / "document_text.md").write_text(str(result.get("document_text") or "").strip() + "\n", encoding="utf-8")
    ocr_restructured = result.get("ocr_restructured") if isinstance(result.get("ocr_restructured"), dict) else {}
    paragraphs = ocr_restructured.get("layout_parsing_results") if isinstance(ocr_restructured, dict) else []
    (result_dir / "ocr_paragraphs.json").write_text(
        json.dumps(paragraphs if isinstance(paragraphs, list) else [], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
