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


class PaddleOcrError(RuntimeError):
    pass


class PaddleOcrClient:
    def __init__(self, config: AppConfig, session: requests.Session | None = None) -> None:
        self.config = config
        self.session = session or requests.Session()
        self._pipeline = None

    def healthcheck(self) -> bool:
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

    def restructure_pages(self, pages: list[dict[str, Any]] | list[Any]) -> dict[str, Any]:
        if self.config.paddle_ocr_mode == "python_api":
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
    ) -> list[dict[str, Any]] | list[Any]:
        if self.config.paddle_ocr_mode == "python_api":
            return list((layout_result or {}).get("_pages_res") or [])

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
        except Exception as exc:  # noqa: BLE001
            raise PaddleOcrError(f"PaddleOCR Python API predict failed: {exc}") from exc

        layout_results = [self._result_to_layout_page(res) for res in pages_res]
        LOGGER.info("PaddleOCR Python API parsed %s page(s) for %s", len(layout_results), file_path.name)
        return {
            "layoutParsingResults": layout_results,
            "_pages_res": pages_res,
        }

    def _restructure_pages_python(self, pages: list[Any]) -> dict[str, Any]:
        pipeline = self._get_pipeline()
        try:
            restructured = list(
                pipeline.restructure_pages(
                    pages,
                    merge_tables=True,
                    relevel_titles=True,
                    concatenate_pages=True,
                    prettify_markdown=True,
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise PaddleOcrError(f"PaddleOCR Python API restructure_pages failed: {exc}") from exc

        return {"_restructured_results": restructured}

    def _normalize_python_restructured_document(
        self, response: dict[str, Any], fallback_pages: list[OcrPage]
    ) -> RestructuredDocument:
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
                    "PaddlePaddle runtime is not installed. Install it in backend/.venv, for example:\n"
                    "python -m pip install paddlepaddle-gpu==3.3.1 "
                    "-i https://www.paddlepaddle.org.cn/packages/stable/cu130/"
                ) from exc
            raise
        LOGGER.info("Initialized PaddleOCRVL Python client with server %s", self.config.paddle_ocr_server_url)
        return self._pipeline

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
