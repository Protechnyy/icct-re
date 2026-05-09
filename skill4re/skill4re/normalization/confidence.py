"""置信度评分模块。

评估单条关系的置信度，并将关系按置信度分组。
"""

from typing import Dict, List, Optional, Tuple

from skill4re.normalization.evidence import text_contains


HIGH_CONFIDENCE_RELATIONS = {
    "部署于", "集结于", "位于", "覆盖", "压制", "干扰", "封锁", "切断",
    "保障", "依托", "连接", "受控于", "节制", "指挥", "任务", "开始于",
    "结束于", "失败于", "受限于", "摧毁", "阻断", "遮断", "侦察",
}


def relation_confidence(
    rel: Dict,
    source_text: Optional[str] = None,
) -> float:
    """评估单条关系的置信度（0.0 ~ 1.0）。

    判断依据：
    1. evidence 是否合理（长度适中、非空）
    2. head 和 tail 是否都在 evidence 中出现
    3. relation 是否为已知高频关系词
    4. head/tail 长度是否合理（非整句）
    """
    score = 0.0
    head = rel.get("head", "")
    tail = rel.get("tail", "")
    relation = rel.get("relation", "")
    evidence = rel.get("evidence", "")

    # evidence 非空且长度合理
    if evidence and 4 <= len(evidence) <= 120:
        score += 0.25
    elif evidence:
        score += 0.1

    # head 和 tail 都在 evidence 中出现
    if evidence and head in evidence and tail in evidence:
        score += 0.3
    elif evidence and (head in evidence or tail in evidence):
        score += 0.1

    # relation 是已知高频关系词
    if relation in HIGH_CONFIDENCE_RELATIONS:
        score += 0.2

    # head/tail 长度合理（不是整句）
    if 1 <= len(head) <= 16 and 1 <= len(tail) <= 20:
        score += 0.15
    elif len(head) <= 20 and len(tail) <= 24:
        score += 0.05

    # source_text 支持
    if source_text and evidence and text_contains(source_text, evidence):
        score += 0.1

    return min(score, 1.0)


def split_by_confidence(
    relations: List[Dict],
    source_text: Optional[str] = None,
    threshold: float = 0.5,
) -> Tuple[List[Dict], List[Dict]]:
    """将关系列表按置信度分为高置信度和低置信度两组。"""
    high_conf = []
    low_conf = []
    for rel in relations:
        if relation_confidence(rel, source_text) >= threshold:
            high_conf.append(rel)
        else:
            low_conf.append(rel)
    return high_conf, low_conf
