"""去重与合并模块。

关系清洗、规范化、模糊去重、跨 chunk 合并。
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from skill4re.normalization.entity import fix_entity_name
from skill4re.normalization.evidence import (
    find_supporting_evidence,
    normalize_relation_text,
    relation_has_source_support,
    select_evidence,
    text_contains,
)


LONG_TAIL_THRESHOLD = 24
GENERIC_DOCUMENT_HEADS = {"本次行动", "本阶段行动"}
TAIL_COMPRESSION_RELATIONS = {
    "任务",
    "失败于",
    "受限于",
    "不得",
    "不得攻击",
    "不攻击",
    "不得脱离",
    "优先转运",
    "准备抗击",
}

# 过泛的关系词，应该被过滤或替换
GENERIC_RELATIONS = {
    "必须", "关键", "威胁", "涉及", "包含", "属于", "有关", "关于",
    "实施", "进行", "执行", "开展", "推进", "推动", "促进", "加强",
    "提高", "提升", "增强", "完善", "优化", "改进", "改善",
}

# 过泛关系词的推荐替换
RELATION_REPLACEMENTS = {
    "必须": "要求",
    "关键": "依赖",
    "威胁": "威胁到",
    "涉及": "相关",
    "包含": "组成",
}


def looks_like_clause(text: str) -> bool:
    if len(text) <= 20:
        return False
    markers = ["并", "且", "以便", "用于", "确保", "随后", "同时", "若", "则", "必须", "任务", "行动"]
    return any(marker in text for marker in markers)


def looks_like_bad_tail(text: str) -> bool:
    if len(text) <= 28:
        return False
    markers = ["；", "。", "，且", "，并", "否则", "以防", "优先", "原则", "行动", "任务通报"]
    return any(marker in text for marker in markers)


def _strip_prefix_clause(text: str) -> str:
    candidate = text.strip(" ，。；：,.;")
    if not candidate:
        return candidate
    prefix_patterns = [
        r"^.+?(?:后|前|时|情况下)，",
        r"^.+?(?:后|前|时|情况下)",
    ]
    for pattern in prefix_patterns:
        match = re.match(pattern, candidate)
        if match:
            remainder = candidate[match.end() :].strip(" ，。；：,.;")
            if remainder:
                candidate = remainder
                break
    return candidate


def _strip_leading_task_verb(text: str) -> str:
    candidate = text.strip(" ，。；：,.;")
    verb_patterns = [
        r"^(?:建立)(.+)$",
        r"^(?:优先转运)(.+)$",
        r"^(?:准备抗击)(.+)$",
        r"^(?:压制)(.+)$",
        r"^(?:封锁)(.+)$",
        r"^(?:切断)(.+)$",
        r"^(?:截获)(.+)$",
        r"^(?:摧毁)(.+)$",
        r"^(?:设置)(.+)$",
        r"^(?:构设)(.+)$",
        r"^(?:插入)(.+)$",
        r"^(?:遮断射击)(.+)$",
        r"^(?:重点压制)(.+)$",
        r"^(?:不得)(.+)$",
    ]
    for pattern in verb_patterns:
        match = re.match(pattern, candidate)
        if match:
            remainder = match.group(1).strip(" ，。；：,.;")
            if remainder:
                return remainder
    return candidate


def compress_long_tail(text: str) -> str:
    candidate = text.strip(" ，。；：,.;")
    candidate = _strip_prefix_clause(candidate)
    if len(candidate) <= LONG_TAIL_THRESHOLD:
        return _strip_leading_task_verb(candidate)
    if len(candidate) <= LONG_TAIL_THRESHOLD:
        return candidate
    clauses = [
        item.strip(" ，。；：,.;")
        for item in re.split(r"[；。]", candidate)
        if item.strip(" ，。；：,.;")
    ]
    if clauses:
        shortest = min(clauses, key=len)
        if len(shortest) < len(candidate):
            candidate = shortest
    return _strip_leading_task_verb(candidate)


def split_compound_tail(text: str) -> List[str]:
    if not text:
        return []
    candidate = text.strip(" ，。；：,.;")
    if not candidate:
        return []
    parts = [
        item.strip(" ，。；：,.;")
        for item in re.split(r"[、]|以及|及|与|和", candidate)
    ]
    parts = [item for item in parts if item]
    if len(parts) < 2:
        return []
    if any(len(item) > 24 for item in parts):
        return []
    return parts


def expand_relation_item(rel: Dict[str, str]) -> List[Dict[str, str]]:
    relation = rel.get("relation", "")
    tail = rel.get("tail", "")
    evidence = rel.get("evidence", "")
    skill = rel.get("skill", "")
    splittable_relations = {
        "位于",
        "部署于",
        "集结于",
        "压制",
        "覆盖",
        "封锁",
        "切断",
        "依托",
        "连接",
        "保障",
    }
    if relation not in splittable_relations or skill not in {"force-organization", "engagement-effects"}:
        return [rel]
    parts = split_compound_tail(tail)
    if not parts:
        return [rel]
    matched_parts = [part for part in parts if part in evidence]
    if len(matched_parts) < 2:
        return [rel]
    expanded = []
    for part in matched_parts:
        item = rel.copy()
        item["tail"] = part
        expanded.append(item)
    return expanded if len(expanded) >= 2 else [rel]


def normalize_relation_phrase(head: str, relation: str, tail: str) -> str:
    relation = relation.strip()
    simple_map = {
        "受控制": "受控于",
        "接受": "接收",
        "位于于": "位于",
    }
    relation = simple_map.get(relation, relation)
    if relation == "部署" and any(token in tail for token in ["区", "桥", "岸", "高地", "坡", "站", "线", "口", "带", "地域"]):
        return "部署于"
    if relation == "负责" and len(tail) <= 10:
        return "任务"
    return relation


def sanitize_relation_item(
    rel: Dict,
    valid_skill_names: Optional[Set[str]] = None,
    source_text: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    head = rel.get("head", "").strip(" ，。；：,.;")
    raw_tail = rel.get("tail", "").strip(" ，。；：,.;")
    # 修复实体名中的重复错误
    head = fix_entity_name(head, source_text or "")
    raw_tail = fix_entity_name(raw_tail, source_text or "")
    relation = normalize_relation_phrase(head, rel.get("relation", "").strip(" ，。；：,.;"), raw_tail)
    tail = compress_long_tail(raw_tail) if relation in TAIL_COMPRESSION_RELATIONS else raw_tail
    relation = normalize_relation_phrase(head, relation, tail)
    evidence = rel.get("evidence", "").strip()
    skill = rel.get("skill", "").strip()
    if not head or not relation or not tail or not skill:
        return None
    if len(relation) > 12:
        return None
    if head in {"主要目标", "次要目标", "失败条件", "时间约束", "优先原则"}:
        return None
    if relation in {"为", "由", "是"}:
        return None
    if looks_like_clause(head) and head not in {"本次行动", "本阶段行动"}:
        return None
    if looks_like_bad_tail(tail):
        return None
    if head in {"我方", "敌方"} and relation in {"实施", "拟实施", "目标为", "任务", "计划"}:
        return None
    if relation in {"主要目标", "次要目标", "失败条件"}:
        return None
    # 过滤过泛的关系词
    if relation in GENERIC_RELATIONS:
        # 尝试替换
        if relation in RELATION_REPLACEMENTS:
            relation = RELATION_REPLACEMENTS[relation]
        else:
            return None
    if valid_skill_names is not None and skill not in valid_skill_names:
        return None
    if not relation_has_source_support(source_text, head, tail, evidence):
        return None
    return {
        "head": head,
        "relation": relation,
        "tail": tail,
        "evidence": select_evidence(head, relation, tail, evidence, source_text=source_text),
        "skill": skill,
    }


def _edit_distance(a: str, b: str) -> int:
    """计算两个字符串的编辑距离。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # 优化：只计算距离，不需要完整的矩阵
    if abs(len(a) - len(b)) > 3:
        return abs(len(a) - len(b))  # 快速剪枝
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        curr = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[len(b)]


def fuzzy_entity_match(a: str, b: str) -> bool:
    """判断两个实体名是否指向同一实体（模糊匹配）。

    规则：
    1. 精确匹配
    2. 一个是另一个的子串（如"第3营" vs "第3机械化步兵营"）
    3. 编辑距离 <= 2 且长度差 <= 3（如"远程火箭营" vs "远程火箭部队"）
    """
    if a == b:
        return True
    if not a or not b:
        return False
    # 子串匹配
    if a in b or b in a:
        return True
    # 编辑距离匹配（仅对长度相近的字符串）
    if abs(len(a) - len(b)) <= 3:
        return _edit_distance(a, b) <= 2
    return False


def _find_fuzzy_match(
    head: str,
    relation: str,
    tail: str,
    skill: str,
    seen: Dict[Tuple[str, str, str, str], Dict],
) -> Optional[Tuple[str, str, str, str]]:
    """在已有的 seen 字典中查找模糊匹配的 key。"""
    for existing_key in seen:
        if existing_key[1] == relation and existing_key[3] == skill:
            if fuzzy_entity_match(existing_key[0], head) and fuzzy_entity_match(existing_key[2], tail):
                return existing_key
    return None


def merge_chunk_relations(
    chunk_results: List[Dict],
    valid_skill_names: Optional[Set[str]] = None,
    source_text: Optional[str] = None,
) -> Dict:
    merged = []
    seen: Dict[Tuple[str, str, str, str], Dict] = {}
    for item in chunk_results:
        for rel in item.get("relation_list", []):
            sanitized = sanitize_relation_item(
                rel,
                valid_skill_names=valid_skill_names,
                source_text=source_text,
            )
            if not sanitized:
                continue
            for expanded in expand_relation_item(sanitized):
                head = normalize_relation_text(expanded.get("head", ""))
                relation = normalize_relation_text(expanded.get("relation", ""))
                tail = normalize_relation_text(expanded.get("tail", ""))
                skill = normalize_relation_text(expanded.get("skill", ""))
                if not head or not relation or not tail:
                    continue
                evidence = expanded.get("evidence", "").strip()

                # 先尝试精确匹配
                key = (head, relation, tail, skill)
                if key in seen:
                    existing = seen[key]
                    if evidence and (not existing["evidence"] or len(evidence) < len(existing["evidence"])):
                        existing["evidence"] = evidence
                    continue

                # 再尝试模糊匹配
                matched_key = _find_fuzzy_match(head, relation, tail, skill, seen)
                if matched_key is not None:
                    existing = seen[matched_key]
                    # 保留更完整的实体名
                    if len(head) > len(matched_key[0]):
                        existing["head"] = head
                    if len(tail) > len(matched_key[2]):
                        existing["tail"] = tail
                    # 更新 evidence
                    if evidence and (not existing["evidence"] or len(evidence) < len(existing["evidence"])):
                        existing["evidence"] = evidence
                    # 更新 seen 字典的 key（指向同一个对象）
                    new_key = (existing["head"], relation, existing["tail"], skill)
                    if new_key != matched_key:
                        del seen[matched_key]
                        seen[new_key] = existing
                    continue

                # 新增
                stored = expanded
                seen[key] = stored
                merged.append(stored)
    return {"relation_list": merged}


def normalize_prediction(
    prediction: Dict,
    valid_skill_names: Optional[Set[str]] = None,
    source_text: Optional[str] = None,
) -> Dict:
    return merge_chunk_relations(
        [prediction],
        valid_skill_names=valid_skill_names,
        source_text=source_text,
    )
