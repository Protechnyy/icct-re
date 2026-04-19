from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import AppConfig
from .paddle_ocr import PaddleOcrClient
from .types import Chunk, FinalRelation, OcrPage, RelationTriple
from .utils import chunk_text_with_page_map, sentence_segments, utcnow_iso
from .vllm_client import VllmClient

if TYPE_CHECKING:
    from .task_store import RedisTaskStore


class DocumentPipeline:
    def __init__(
        self,
        config: AppConfig,
        task_store: RedisTaskStore,
        ocr_client: PaddleOcrClient,
        vllm_client: VllmClient,
    ) -> None:
        self.config = config
        self.task_store = task_store
        self.ocr_client = ocr_client
        self.vllm_client = vllm_client

    def process_task(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        file_path = Path(payload["file_path"])
        filename = payload["filename"]
        file_type = int(payload["file_type"])

        self.task_store.update_task(task_id, status="ocr_running", stage="layout_parsing", progress=10, error=None)
        layout_result = self.ocr_client.layout_parse(file_path=file_path, file_type=file_type)
        ocr_pages = self.ocr_client.extract_pages(layout_result)

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
        stage_outputs, final_relations, llm_errors = self._extract_relations(ocr_pages, chunks, document_text)

        self.task_store.update_task(task_id, status="merging", stage="document_merge", progress=85)
        safe_layout_result = {k: v for k, v in layout_result.items() if k != "_pages_res"}
        result = {
            "document_meta": {
                "task_id": task_id,
                "filename": filename,
                "page_count": len(ocr_pages),
                "processed_at": utcnow_iso(),
                "model": self.config.vllm_model,
            },
            "ocr_summary": {
                "ocr_restructure_fallback": restructure_fallback,
                "document_text_length": len(document_text),
                "chunk_count": len(chunks),
            },
            "ocr_raw": safe_layout_result,
            "ocr_pages": [page.to_dict() for page in ocr_pages],
            "ocr_restructured": restructured.to_dict(),
            "document_text": document_text,
            "chunks": [chunk.to_dict() for chunk in chunks],
            "stage_outputs": stage_outputs,
            "final_relations": [relation.to_dict() for relation in final_relations],
        }
        if llm_errors:
            result["llm_errors"] = llm_errors
        self.task_store.set_result(task_id, result)
        self.task_store.update_task(task_id, status="succeeded", stage="completed", progress=100, error=None)
        return result

    def _extract_relations(
        self, ocr_pages: list[OcrPage], chunks: list[Chunk], document_text: str
    ) -> tuple[dict[str, list[dict[str, Any]]], list[FinalRelation], list[dict[str, Any]]]:
        sentence_inputs = [
            {"chunk_id": f"sentence-{idx + 1}", "text": text, "page_start": 1, "page_end": 1}
            for idx, text in enumerate(sentence_segments(document_text, self.config.sentence_stage_limit))
        ]
        page_inputs = [
            {"chunk_id": f"page-{page.page_index}", "text": page.markdown_text, "page_start": page.page_index, "page_end": page.page_index}
            for page in ocr_pages[: self.config.page_stage_limit]
            if page.markdown_text.strip()
        ]
        multipage_inputs = [chunk.to_dict() for chunk in chunks]

        sentence_stage, sentence_errors = self._run_stage(sentence_inputs, "句子级抽取")
        page_stage, page_errors = self._run_stage(page_inputs, "页面级抽取")
        multipage_stage, multipage_errors = self._run_stage(multipage_inputs, "多页级抽取")

        final_relations = self._merge_final_relations(multipage_stage + page_stage + sentence_stage)
        errors = sentence_errors + page_errors + multipage_errors
        return (
            {
                "sentence": [self._as_stage_triple(triple).to_dict() for triple in sentence_stage],
                "page": [self._as_stage_triple(triple).to_dict() for triple in page_stage],
                "multipage": [self._as_stage_triple(triple).to_dict() for triple in multipage_stage],
            },
            final_relations,
            errors,
        )

    def _run_stage(
        self, units: list[dict[str, Any]], context: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not units:
            return [], []

        triples: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.config.llm_concurrency) as executor:
            future_map = {
                executor.submit(self.vllm_client.extract_relations, unit["text"], f"{context}::{unit['chunk_id']}"): unit
                for unit in units
            }
            for future in as_completed(future_map):
                unit = future_map[future]
                try:
                    results = future.result()
                except Exception as exc:  # noqa: BLE001
                    errors.append({"chunk_id": unit["chunk_id"], "error": str(exc)})
                    continue
                for item in results:
                    subject = str(item.get("subject", "")).strip()
                    relation = str(item.get("relation", "")).strip()
                    obj = str(item.get("object", "")).strip()
                    if not (subject and relation and obj):
                        continue
                    triples.append(
                        {
                            "subject": subject,
                            "relation": relation,
                            "object": obj,
                            "evidence": unit["text"][:300],
                            "page": unit["page_start"],
                            "chunk_id": unit["chunk_id"],
                            "source_text": unit["text"],
                        }
                    )
        return triples, errors

    def _merge_final_relations(self, items: list[dict[str, Any]]) -> list[FinalRelation]:
        deduped: dict[tuple[str, str, str], FinalRelation] = {}
        for item in items:
            key = (
                item["subject"].strip().lower(),
                item["relation"].strip().lower(),
                item["object"].strip().lower(),
            )
            if key in deduped:
                continue
            deduped[key] = FinalRelation(
                subject=item["subject"],
                relation=item["relation"],
                object=item["object"],
                evidence=item["evidence"],
                page=int(item["page"]),
                chunk_id=item["chunk_id"],
                source_text=item["source_text"],
            )
        return list(deduped.values())

    def _as_stage_triple(self, item: dict[str, Any]) -> RelationTriple:
        return RelationTriple(
            subject=item["subject"],
            relation=item["relation"],
            object=item["object"],
        )


def bootstrap_pipeline(config: AppConfig) -> DocumentPipeline:
    from .task_store import RedisTaskStore

    task_store = RedisTaskStore(config.redis_url)
    ocr_client = PaddleOcrClient(config)
    vllm_client = VllmClient(config)
    return DocumentPipeline(config, task_store, ocr_client, vllm_client)
