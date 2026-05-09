from __future__ import annotations

import base64
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PDF_FILE_TYPE = 0
IMAGE_FILE_TYPE = 1
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_UPLOAD_SUFFIXES = {".pdf", *SUPPORTED_IMAGE_SUFFIXES}


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def generate_task_id() -> str:
    return uuid.uuid4().hex


def detect_file_type(path: Path) -> int:
    suffix = path.suffix.lower() if path.suffix.lower() in SUPPORTED_UPLOAD_SUFFIXES else sniff_supported_suffix(path)
    if suffix == ".pdf":
        return PDF_FILE_TYPE
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return IMAGE_FILE_TYPE
    raise ValueError(f"Unsupported file type: {path.name}")


def ensure_supported_file_suffix(path: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_UPLOAD_SUFFIXES:
        return path

    sniffed_suffix = sniff_supported_suffix(path)
    if sniffed_suffix is None:
        raise ValueError(f"Unsupported file type: {path.name}")

    renamed = path.with_name(f"{path.name}{sniffed_suffix}")
    path.rename(renamed)
    return renamed


def sniff_supported_suffix(path: Path) -> str | None:
    header = path.read_bytes()[:16]
    if header.startswith(b"%PDF-"):
        return ".pdf"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if header.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if header.startswith(b"BM"):
        return ".bmp"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return ".webp"
    return None


def file_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def safe_filename(name: str) -> str:
    filename = Path(name.replace("\\", "/")).name
    filename = re.sub(r"[\x00-\x1f\x7f]+", "_", filename)
    filename = re.sub(r"[/:]+", "_", filename).strip(" .")
    return filename or "upload"


def chunk_text_with_page_map(page_texts: list[tuple[int, str]], max_chunk_chars: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    buffer: list[str] = []
    page_start: int | None = None
    page_end: int | None = None

    def flush() -> None:
        nonlocal buffer, page_start, page_end
        text = "\n\n".join(part.strip() for part in buffer if part.strip()).strip()
        if text:
            chunk_id = f"chunk-{len(chunks) + 1}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "page_start": page_start if page_start is not None else 1,
                    "page_end": page_end if page_end is not None else 1,
                }
            )
        buffer = []
        page_start = None
        page_end = None

    for page_index, text in page_texts:
        normalized = text.strip()
        if not normalized:
            continue
        if page_start is None:
            page_start = page_index
        projected = "\n\n".join(buffer + [normalized]).strip()
        if buffer and len(projected) > max_chunk_chars:
            flush()
            page_start = page_index
        buffer.append(normalized)
        page_end = page_index

        while buffer and len("\n\n".join(buffer)) > max_chunk_chars:
            long_text = buffer.pop()
            segments = split_long_text(long_text, max_chunk_chars)
            if buffer:
                flush()
            for idx, segment in enumerate(segments):
                chunks.append(
                    {
                        "chunk_id": f"chunk-{len(chunks) + 1}",
                        "text": segment,
                        "page_start": page_index,
                        "page_end": page_index,
                    }
                )
            page_start = None
            page_end = None

    flush()
    return chunks


def split_long_text(text: str, max_chunk_chars: int) -> list[str]:
    pieces: list[str] = []
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    current = ""
    for paragraph in paragraphs or [text]:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if current and len(candidate) > max_chunk_chars:
            pieces.append(current)
            current = paragraph
        else:
            current = candidate
        while len(current) > max_chunk_chars:
            pieces.append(current[:max_chunk_chars].strip())
            current = current[max_chunk_chars:].strip()
    if current:
        pieces.append(current)
    return pieces


def sentence_segments(text: str, limit: int) -> list[str]:
    segments = [seg.strip() for seg in re.split(r"(?<=[。！？.!?])\s+", text) if seg.strip()]
    return segments[:limit]


def extract_json_array(raw: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            if isinstance(parsed.get("relations"), list):
                return [item for item in parsed["relations"] if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\[[\s\S]*\])", raw)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)]
