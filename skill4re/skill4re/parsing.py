import json
import re
from typing import Dict, Optional


def recover_relation_list(text: str) -> Optional[Dict]:
    key_pos = text.find('"relation_list"')
    if key_pos < 0:
        return None
    array_start = text.find("[", key_pos)
    if array_start < 0:
        return None
    items = []
    depth = 0
    in_string = False
    escape = False
    obj_start = None
    for idx in range(array_start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                obj_start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start is not None:
                    chunk = text[obj_start : idx + 1]
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict):
                        items.append(obj)
                    obj_start = None
        elif ch == "]" and depth == 0:
            break
    if items:
        return {"relation_list": items}
    return None


def parse_json(content: str) -> Dict:
    text = content.strip()
    if text.startswith("```json"):
        text = text[len("```json") :].strip()
    if text.startswith("```"):
        text = text[len("```") :].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                recovered = recover_relation_list(candidate)
                if recovered is not None:
                    return recovered
        recovered = recover_relation_list(text)
        if recovered is not None:
            return recovered
        raise


def coerce_router_payload(value) -> Dict:
    """路由输出预期为 JSON 对象；模型输出数组时按可验证的等价形态恢复，无法恢复就报错。"""
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and "primary_skill" in item:
                return item
        str_items = [item for item in value if isinstance(item, str)]
        if str_items and len(str_items) == len(value):
            return {"primary_skill": str_items[0], "aux_skills": str_items[1:], "reason": ""}
    raise ValueError(f"router output is not a JSON object: {json.dumps(value, ensure_ascii=False)[:200]}")


def _is_relation_entry(item) -> bool:
    return isinstance(item, dict) and bool({"subject", "head", "object", "tail", "relation"} & set(item))


def coerce_relation_payload(value) -> Dict:
    """抽取与校验输出预期为含 relation_list 的对象；模型输出数组时恢复为该结构，无法恢复就报错。"""
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value:
        if all(isinstance(item, dict) and isinstance(item.get("relation_list"), list) for item in value):
            merged: list = []
            for item in value:
                merged.extend(item["relation_list"])
            return {"relation_list": merged}
        if all(_is_relation_entry(item) for item in value):
            return {"relation_list": value}
    raise ValueError(f"relation output is not a JSON object: {json.dumps(value, ensure_ascii=False)[:200]}")
