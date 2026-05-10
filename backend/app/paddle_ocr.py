from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

import requests

from .config import AppConfig
from .types import OcrPage, RestructuredDocument
from .utils import file_to_base64

LOGGER = logging.getLogger(__name__)

IGNORED_TEXT_BLOCK_LABELS = {
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
    "seal",
}


class PaddleOcrError(RuntimeError):
    pass


class PaddleOcrClient:
    def __init__(self, config: AppConfig, session: requests.Session | None = None) -> None:
        self.config = config
        self.session = session or requests.Session()
        self._pipeline = None

    def healthcheck(self) -> bool:
        if self.config.paddle_ocr_mode == "python_api":
            try:
                response = self.session.get(
                    f"{self.config.paddle_ocr_server_url}/models",
                    timeout=5,
                    headers=self._headers(),
                )
                return response.ok
            except requests.RequestException:
                return False
        try:
            response = self.session.get(f"{self.config.paddle_ocr_base_url}/health", timeout=5)
            return response.ok
        except requests.RequestException:
            return False

    def layout_parse(self, file_path: Path, file_type: int) -> dict[str, Any]:
        if self.config.paddle_ocr_mode == "python_api":
            return self._layout_parse_python(file_path)

        payload: dict[str, Any] = {
            "file": file_to_base64(file_path),
            "fileType": file_type,
            "visualize": False,
            "restructurePages": False,
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useLayoutDetection": True,
        }
        return self._post("/layout-parsing", payload)

    def restructure_pages(self, pages: list[dict[str, Any]] | list[Any] | dict[str, Any]) -> dict[str, Any]:
        if self.config.paddle_ocr_mode == "python_api":
            if isinstance(pages, dict) and isinstance(pages.get("parsing_res_list"), list):
                return pages
            return self._restructure_pages_python(pages)

        payload = {
            "pages": pages,
            "mergeTables": True,
            "relevelTitles": True,
            "concatenatePages": True,
            "prettifyMarkdown": True,
        }
        return self._post("/restructure-pages", payload)

    def extract_pages(self, layout_result: dict[str, Any]) -> list[OcrPage]:
        if isinstance(layout_result.get("parsing_res_list"), list):
            return self._extract_pages_from_blocks(layout_result)

        raw_pages = layout_result.get("layoutParsingResults") or []
        pages: list[OcrPage] = []
        for idx, page in enumerate(raw_pages, start=1):
            markdown = page.get("markdown") or {}
            markdown_text = (
                markdown.get("markdownText")
                or markdown.get("text")
                or markdown.get("content")
                or page.get("markdownText")
                or ""
            )
            pages.append(
                OcrPage(
                    page_index=idx,
                    pruned_result=page.get("prunedResult") or {},
                    markdown_text=str(markdown_text or "").strip(),
                    markdown_images=markdown.get("images"),
                )
            )
        return pages

    def build_restructure_payload(
        self, ocr_pages: list[OcrPage], layout_result: dict[str, Any] | None = None
    ) -> list[dict[str, Any]] | list[Any] | dict[str, Any]:
        if self.config.paddle_ocr_mode == "python_api":
            return layout_result or {}

        return [
            {
                "prunedResult": page.pruned_result,
                "markdownImages": page.markdown_images,
            }
            for page in ocr_pages
        ]

    def normalize_restructured_document(
        self, response: dict[str, Any], fallback_pages: list[OcrPage]
    ) -> RestructuredDocument:
        if self.config.paddle_ocr_mode == "python_api":
            return self._normalize_python_restructured_document(response, fallback_pages)

        layout_results = response.get("layoutParsingResults") or response.get("layoutParsingResult") or []
        markdown_text = response.get("markdownText") or response.get("markdown") or response.get("text") or ""
        if not markdown_text and layout_results:
            collected: list[str] = []
            for item in layout_results:
                markdown = item.get("markdown") if isinstance(item, dict) else None
                if isinstance(markdown, dict):
                    maybe_text = markdown.get("markdownText") or markdown.get("text") or markdown.get("content")
                    if maybe_text:
                        collected.append(str(maybe_text))
                elif isinstance(item, dict) and item.get("text"):
                    collected.append(str(item["text"]))
            markdown_text = "\n\n".join(part.strip() for part in collected if part.strip())
        if not markdown_text:
            markdown_text = "\n\n".join(page.markdown_text for page in fallback_pages if page.markdown_text)
        normalized_layout = layout_results if isinstance(layout_results, list) else []
        return RestructuredDocument(markdown_text=markdown_text.strip(), layout_parsing_results=normalized_layout)

    def _layout_parse_python(self, file_path: Path) -> dict[str, Any]:
        pipeline = self._get_pipeline()
        try:
            pages_res = list(pipeline.predict(input=str(file_path)))
            restructured = list(
                pipeline.restructure_pages(
                    pages_res,
                    merge_tables=True,
                    relevel_titles=True,
                    concatenate_pages=True,
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise PaddleOcrError(f"PaddleOCRVL predict/restructure failed: {exc}") from exc

        document = self._result_to_document(restructured[0]) if restructured else {}
        if not isinstance(document.get("parsing_res_list"), list):
            fallback_pages = [self._result_to_layout_page(res) for res in pages_res]
            document["layoutParsingResults"] = fallback_pages
            document["markdownText"] = "\n\n".join(
                page.get("markdown", {}).get("text", "").strip()
                for page in fallback_pages
                if page.get("markdown", {}).get("text", "").strip()
            )
        else:
            document["paragraphs"] = self._paragraphs_from_blocks(document)
            document["markdownText"] = "\n\n".join(item["content"] for item in document["paragraphs"])
            document["layoutParsingResults"] = self._layout_pages_from_blocks(document)
        document["_pages_res"] = pages_res
        LOGGER.info(
            "PaddleOCRVL parsed %s page(s), %s paragraph(s) for %s",
            document.get("page_count") or len(document.get("layoutParsingResults") or []),
            len(document.get("paragraphs") or []),
            file_path.name,
        )
        return document

    def _restructure_pages_python(self, pages: list[Any] | dict[str, Any]) -> dict[str, Any]:
        if isinstance(pages, dict):
            return pages
        pipeline = self._get_pipeline()
        try:
            restructured = list(
                pipeline.restructure_pages(
                    pages,
                    merge_tables=True,
                    relevel_titles=True,
                    concatenate_pages=True,
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise PaddleOcrError(f"PaddleOCR Python API restructure_pages failed: {exc}") from exc

        document = self._result_to_document(restructured[0]) if restructured else {}
        if isinstance(document.get("parsing_res_list"), list):
            document["paragraphs"] = self._paragraphs_from_blocks(document)
            document["markdownText"] = "\n\n".join(item["content"] for item in document["paragraphs"])
            document["layoutParsingResults"] = self._layout_pages_from_blocks(document)
        return document

    def _normalize_python_restructured_document(
        self, response: dict[str, Any], fallback_pages: list[OcrPage]
    ) -> RestructuredDocument:
        if isinstance(response.get("parsing_res_list"), list):
            paragraphs = response.get("paragraphs") or self._paragraphs_from_blocks(response)
            markdown_text = "\n\n".join(item["content"] for item in paragraphs if item.get("content")).strip()
            if not markdown_text:
                markdown_text = "\n\n".join(page.markdown_text for page in fallback_pages if page.markdown_text).strip()
            return RestructuredDocument(markdown_text=markdown_text, layout_parsing_results=paragraphs)

        raw_results = response.get("_restructured_results") or []
        layout_results = [self._result_to_layout_page(res) for res in raw_results]
        markdown_parts = []
        for item in layout_results:
            markdown = item.get("markdown") or {}
            text = markdown.get("text")
            if text:
                markdown_parts.append(str(text).strip())
        markdown_text = "\n\n".join(part for part in markdown_parts if part).strip()
        if not markdown_text:
            markdown_text = "\n\n".join(page.markdown_text for page in fallback_pages if page.markdown_text).strip()
        return RestructuredDocument(markdown_text=markdown_text, layout_parsing_results=layout_results)

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:  # pragma: no cover
            raise PaddleOcrError(
                "PaddleOCR Python package is not installed. Reinstall backend dependencies to enable the "
                "official PaddleOCR-VL Python API client."
            ) from exc

        kwargs = {
            "vl_rec_backend": "vllm-server",
            "vl_rec_server_url": self.config.paddle_ocr_server_url,
            "vl_rec_max_concurrency": self.config.paddle_ocr_max_concurrency,
        }
        if self.config.paddle_ocr_api_model_name:
            kwargs["vl_rec_api_model_name"] = self.config.paddle_ocr_api_model_name
        if self.config.paddle_ocr_api_key:
            kwargs["vl_rec_api_key"] = self.config.paddle_ocr_api_key
        try:
            self._pipeline = PaddleOCRVL(**kwargs)
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            if exc.name == "paddle":
                raise PaddleOcrError(
                    "PaddlePaddle runtime is not installed. Install it in the backend environment, for example:\n"
                    "python -m pip install paddlepaddle==3.3.1"
                ) from exc
            raise
        LOGGER.info("Initialized PaddleOCRVL Python client with server %s", self.config.paddle_ocr_server_url)
        return self._pipeline

    def _headers(self) -> dict[str, str]:
        if not self.config.paddle_ocr_api_key:
            return {}
        return {"Authorization": f"Bearer {self.config.paddle_ocr_api_key}"}

    def _result_to_document(self, result: Any) -> dict[str, Any]:
        json_payload = getattr(result, "json", {}) or {}
        if isinstance(json_payload, dict) and isinstance(json_payload.get("res"), dict):
            return dict(json_payload["res"])
        if isinstance(json_payload, dict):
            return dict(json_payload)
        return {}

    def _extract_pages_from_blocks(self, document: dict[str, Any]) -> list[OcrPage]:
        page_map = self._infer_block_pages(document.get("parsing_res_list") or [])
        blocks_by_page: dict[int, list[dict[str, Any]]] = {}
        text_by_page: dict[int, list[str]] = {}
        for block in self._text_blocks(document):
            page = page_map.get(int(block.get("global_block_id", block.get("block_id", 0))), 1)
            blocks_by_page.setdefault(page, []).append(block)
            text_by_page.setdefault(page, []).append(str(block.get("block_content") or "").strip())

        page_count = int(document.get("page_count") or max(text_by_page, default=1))
        pages: list[OcrPage] = []
        for page_index in range(1, page_count + 1):
            page_blocks = blocks_by_page.get(page_index, [])
            page_text = "\n\n".join(text_by_page.get(page_index, [])).strip()
            if not page_text and not page_blocks:
                continue
            pages.append(
                OcrPage(
                    page_index=page_index,
                    pruned_result={"blocks": page_blocks},
                    markdown_text=page_text,
                )
            )
        return pages

    def _layout_pages_from_blocks(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "prunedResult": page.pruned_result,
                "markdown": {"text": page.markdown_text, "images": None},
            }
            for page in self._extract_pages_from_blocks(document)
        ]

    def _paragraphs_from_blocks(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        page_map = self._infer_block_pages(document.get("parsing_res_list") or [])
        paragraphs: list[dict[str, Any]] = []
        for block in self._text_blocks(document):
            block_id = int(block.get("global_block_id", block.get("block_id", len(paragraphs))))
            content = str(block.get("block_content") or "").strip()
            paragraphs.append(
                {
                    "page": page_map.get(block_id, 1),
                    "label": block.get("block_label") or "",
                    "content": content,
                    "bbox": block.get("block_bbox"),
                    "global_block_id": block_id,
                    "group_id": block.get("group_id"),
                    "global_group_id": block.get("global_group_id"),
                }
            )
        return paragraphs

    def _text_blocks(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        blocks = document.get("parsing_res_list") or []
        if not isinstance(blocks, list):
            return []
        text_blocks: list[dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            label = str(block.get("block_label") or "")
            content = str(block.get("block_content") or "").strip()
            if not content or label in IGNORED_TEXT_BLOCK_LABELS:
                continue
            text_blocks.append(block)
        return text_blocks

    def _infer_block_pages(self, blocks: list[dict[str, Any]]) -> dict[int, int]:
        page_map: dict[int, int] = {}
        page = 0
        prev_group_id = -1
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_id = int(block.get("global_block_id", block.get("block_id", 0)))
            if block.get("page") is not None:
                page_map[block_id] = int(block["page"])
                continue

            label = str(block.get("block_label") or "")
            group_id = int(block.get("group_id") if block.get("group_id") is not None else 0)
            if label in {"aside_text", "seal"}:
                page_map[block_id] = page or 1
                continue
            if group_id <= prev_group_id and group_id == 0:
                page += 1
            if page == 0:
                page = 1
            page_map[block_id] = page
            prev_group_id = group_id
        return page_map

    def _result_to_layout_page(self, result: Any) -> dict[str, Any]:
        json_payload = getattr(result, "json", {}) or {}
        markdown_payload = getattr(result, "markdown", {}) or {}
        res_payload = json_payload.get("res") if isinstance(json_payload, dict) else {}
        if not isinstance(res_payload, dict):
            res_payload = {}
        pruned_result = {key: value for key, value in res_payload.items() if key not in {"input_path", "page_index"}}
        return {
            "prunedResult": pruned_result,
            "markdown": {
                "text": self._extract_markdown_text(markdown_payload),
                "images": self._extract_markdown_images(markdown_payload),
            },
        }

    def _extract_markdown_text(self, markdown_payload: dict[str, Any]) -> str:
        if not isinstance(markdown_payload, dict):
            return ""
        text = markdown_payload.get("markdown_texts")
        if isinstance(text, (list, tuple)):
            return "\n\n".join(str(item).strip() for item in text if str(item).strip()).strip()
        if text is None:
            return ""
        return str(text).strip()

    def _extract_markdown_images(self, markdown_payload: dict[str, Any]) -> dict[str, str] | None:
        if not isinstance(markdown_payload, dict):
            return None
        images = markdown_payload.get("markdown_images")
        if not isinstance(images, dict):
            return None
        encoded: dict[str, str] = {}
        for key, image in images.items():
            if isinstance(image, str):
                encoded[key] = image
            elif hasattr(image, "save"):
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                encoded[key] = base64.b64encode(buffer.getvalue()).decode("ascii")
        return encoded or None

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.paddle_ocr_base_url}{path}"
        try:
            response = self.session.post(url, json=payload, timeout=self.config.paddle_ocr_timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise PaddleOcrError(f"PaddleOCR request failed: {exc}") from exc
        except ValueError as exc:
            raise PaddleOcrError("PaddleOCR returned invalid JSON") from exc

        if isinstance(data, dict) and data.get("errorCode") not in (None, 0):
            raise PaddleOcrError(data.get("errorMsg") or f"PaddleOCR errorCode={data['errorCode']}")
        if isinstance(data, dict) and "result" in data and isinstance(data["result"], dict):
            return data["result"]
        if not isinstance(data, dict):
            raise PaddleOcrError("PaddleOCR returned an unexpected payload")
        return data
