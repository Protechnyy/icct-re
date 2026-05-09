"""证据验证与选择模块。

验证关系的证据是否在原文中存在，并选择最合适的证据。
"""

from typing import List, Optional


def normalize_relation_text(value: str) -> str:
    """规范化关系文本，移除空白字符。"""
    import re
    return re.sub(r"\s+", "", value.strip())


def text_contains(source_text: str, value: str) -> bool:
    """检查 source_text 是否包含 value。"""
    if not source_text or not value:
        return False
    return normalize_relation_text(value) in normalize_relation_text(source_text)


def split_evidence_units(text: str) -> List[str]:
    """将文本按句子分割为证据单元。"""
    import re
    return [
        item.strip(" ，。；：,.;")
        for item in re.split(r"[。！？!?；;\n]+", text)
        if item.strip(" ，。；：,.;")
    ]


def find_supporting_evidence(source_text: str, head: str, tail: str) -> str:
    """在原文中查找支持 head-tail 关系的证据。"""
    if not source_text:
        return ""
    units = split_evidence_units(source_text)
    candidates = [
        unit
        for unit in units
        if text_contains(unit, head) and text_contains(unit, tail)
    ]
    if candidates:
        return min(candidates, key=len)
    # 如果 head 是通用文档标题，只查找 tail
    from skill4re.normalization.dedup import GENERIC_DOCUMENT_HEADS
    if head in GENERIC_DOCUMENT_HEADS:
        tail_candidates = [unit for unit in units if text_contains(unit, tail)]
        if tail_candidates:
            return min(tail_candidates, key=len)
    return ""


def relation_has_source_support(
    source_text: Optional[str],
    head: str,
    tail: str,
    evidence: str,
) -> bool:
    """检查关系是否有原文支持。"""
    if source_text is None:
        return True
    if text_contains(source_text, evidence):
        return True
    return bool(find_supporting_evidence(source_text, head, tail))


def select_evidence(
    head: str,
    relation: str,
    tail: str,
    evidence: str,
    source_text: Optional[str] = None,
) -> str:
    """选择最合适的证据。"""
    candidates = []
    if evidence.strip() and (source_text is None or text_contains(source_text, evidence)):
        candidates.append(evidence.strip())
    if source_text is not None:
        supporting_evidence = find_supporting_evidence(source_text, head, tail)
        if supporting_evidence:
            candidates.append(supporting_evidence)
    synthesized = f"{head}{relation}{tail}".strip()
    if source_text is None or text_contains(source_text, synthesized):
        candidates.append(synthesized)
    candidates = [item for item in candidates if item]
    if not candidates:
        return ""
    return min(candidates, key=len)
