from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OcrPage:
    page_index: int
    pruned_result: dict[str, Any]
    markdown_text: str
    markdown_images: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RestructuredDocument:
    markdown_text: str
    layout_parsing_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_start: int
    page_end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RelationTriple:
    subject: str
    relation: str
    object: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FinalRelation(RelationTriple):
    evidence: str
    page: int
    chunk_id: str
    source_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskStatus:
    task_id: str
    filename: str
    status: str
    progress: int
    stage: str
    error: str | None = None
    result_ready: bool = False
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

