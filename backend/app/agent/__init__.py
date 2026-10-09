from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from ..config import AppConfig
from .runtime import run_verification
from .trace import Trace


def verify_document(
    relations: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    document_text: str,
    config: AppConfig,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    *, model: Any = None, trace: Trace | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not config.agent_enabled:
        return deepcopy(relations), {"status": "disabled"}
    if not relations or not document_text.strip():
        return deepcopy(relations), {"status": "skipped", "reason": "没有可核查的内容"}
    try:
        return run_verification(relations, sections, document_text, config, progress_callback, model, trace=trace)
    except Exception as exc:
        return deepcopy(relations), {"status": "failed", "reason": "核查阶段失败：" + type(exc).__name__,
                                     "summary": {}, "tasks": [], "trace": [], "removed_relations": [],
                                     "entity_aliases": {}, "changes": []}
