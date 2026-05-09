from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import AppConfig
from .paddle_ocr import PaddleOcrClient
from .skill4re_client import RelationExtractor, Skill4ReClient
from .types import Chunk
from .utils import chunk_text_with_page_map, utcnow_iso

if TYPE_CHECKING:
    from .task_store import RedisTaskStore


RELATION_FIELD_ORDER = ("head", "relation", "tail", "evidence", "skill")


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

        self.task_store.update_task(task_id, status="extracting", stage="relation_extraction", progress=55)
        skill4re_result, stage_outputs, final_relations = self._extract_relations(document_text)

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
                "relation_count": len(final_relations),
                "selected_skills": skill4re_result.get("routing", {}).get("selected_skills", []),
            },
            "ocr_raw": safe_layout_result,
            "ocr_pages": [page.to_dict() for page in ocr_pages],
            "ocr_restructured": restructured.to_dict(),
            "document_text": document_text,
            "chunks": [chunk.to_dict() for chunk in chunks],
            "stage_outputs": stage_outputs,
            "skill4re_result": skill4re_result,
            "final_relations": final_relations,
            "final_relation_list": {"relation_list": final_relations},
        }
        self.task_store.set_result(task_id, result)
        self.task_store.update_task(task_id, status="succeeded", stage="completed", progress=100, error=None)
        return result

    def _extract_relations(
        self, document_text: str
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        skill4re_result = _order_relation_payload(self.relation_extractor.extract_document(document_text))
        prediction = skill4re_result.get("prediction", {})
        relation_list = prediction.get("relation_list", []) if isinstance(prediction, dict) else []
        final_relations = [_order_relation_item(item) for item in relation_list if isinstance(item, dict)]
        stage_outputs = {
            "routing": skill4re_result.get("routing", {}),
            "preprocess": skill4re_result.get("preprocess", {}),
            "timing": skill4re_result.get("timing", {}),
            "proofreading": skill4re_result.get("proofreading", {}),
            "chunk_predictions": skill4re_result.get("chunk_predictions", []),
            "prediction": prediction,
        }
        return skill4re_result, stage_outputs, final_relations


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
