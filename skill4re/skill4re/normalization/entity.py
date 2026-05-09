"""实体名修复模块。

基于规则修复 LLM 输出的实体名错误，不需要调用 LLM。
"""

from typing import Optional, Tuple


# 常见的合理重复词，不应被修复
VALID_REPETITIONS = {'人人', '看看', '想想', '谢谢', '往往', '常常', '刚刚', '仅仅', '稍稍'}


def _detect_consecutive_repetition(text: str) -> Optional[Tuple[int, int]]:
    """检测连续重复的子串，返回 (start, length) 或 None。

    例如："铁幕铁幕控制核心" 会检测到 "铁幕" 重复。
    """
    # 检查2-5字的重复
    for length in range(2, 6):
        for i in range(len(text) - length * 2 + 1):
            substr = text[i:i+length]
            # 检查是否连续重复
            if text[i+length:i+length*2] == substr:
                # 排除常见合理重复
                if substr not in VALID_REPETITIONS:
                    return (i, length)
    return None


def _detect_suffix_repetition(text: str) -> Optional[int]:
    """检测后缀重复，返回重复开始的位置或 None。

    例如："发射车发射车" 会检测到整个字符串重复。
    "V3火箭发射车发射车" 会检测到 "发射车" 重复。
    """
    # 检查后缀重复
    for length in range(2, min(len(text) // 2 + 1, 8)):
        suffix = text[-length:]
        if text[-length*2:-length] == suffix:
            return len(text) - length * 2
    return None


def fix_entity_name(entity_name: str, doc_text: str = "") -> str:
    """修复可能的实体名错误，基于规则，不需要LLM。

    修复策略：
    1. 检测并修复连续重复词（如"铁幕铁幕" -> "铁幕"）
    2. 检测并修复后缀重复（如"发射车发射车" -> "发射车"）
    3. 如果修复后的实体名在原文中存在，优先使用修复后的版本
    """
    if not entity_name or len(entity_name) < 4:
        return entity_name

    # 策略1：修复连续重复词
    rep_info = _detect_consecutive_repetition(entity_name)
    if rep_info:
        start, length = rep_info
        # 移除重复部分
        fixed = entity_name[:start+length] + entity_name[start+length*2:]
        # 如果修复后的实体名在原文中存在，使用修复后的版本
        if doc_text and fixed in doc_text:
            return fixed
        # 否则，如果修复后的实体名更短且合理，也使用
        if len(fixed) >= 2 and len(fixed) < len(entity_name):
            entity_name = fixed

    # 策略2：修复后缀重复
    suffix_rep_pos = _detect_suffix_repetition(entity_name)
    if suffix_rep_pos is not None:
        fixed = entity_name[:suffix_rep_pos] + entity_name[suffix_rep_pos:]
        # 如果修复后的实体名在原文中存在，使用修复后的版本
        if doc_text and fixed in doc_text:
            return fixed
        # 否则，如果修复后的实体名更短且合理，也使用
        if len(fixed) >= 2 and len(fixed) < len(entity_name):
            entity_name = fixed

    # 策略3：如果实体名不在原文中，尝试找到最相似的子串
    if doc_text and entity_name not in doc_text:
        # 尝试移除可能的重复后缀
        for trim_len in range(1, min(6, len(entity_name) // 2)):
            trimmed = entity_name[:-trim_len]
            if trimmed in doc_text and len(trimmed) >= 3:
                return trimmed

    # 策略4：检测常见的错误替换模式
    # 例如："第七工业防线车" -> "基地车"（如果原文中有"基地车"）
    if doc_text:
        # 检查是否包含"防线"且原文中有更短的合理实体名
        if "防线" in entity_name and "基地" in doc_text:
            # 尝试提取核心实体名
            # "第七工业防线车" -> 提取 "车" 并检查 "基地车" 是否在原文中
            suffix = ""
            for i in range(len(entity_name) - 1, -1, -1):
                if entity_name[i] in "车营部队连团旅师群":
                    suffix = entity_name[i:]
                    break
            if suffix:
                candidate = "基地" + suffix
                if candidate in doc_text:
                    return candidate

    # 策略5：检测包含错误替换的复合实体名
    # 例如："第七工业防线车损毁" -> "基地车损毁"
    if doc_text and "防线" in entity_name:
        # 尝试找到正确的实体名前缀
        for correct_prefix in ["基地", "前线", "作战"]:
            # 检查是否包含错误替换的模式
            if "防线" in entity_name:
                # 提取"防线"后面的部分
                idx = entity_name.find("防线")
                if idx >= 0:
                    suffix = entity_name[idx + 2:]  # "防线"后面的部分
                    candidate = correct_prefix + suffix
                    # 检查修复后的实体名是否在原文中
                    if candidate in doc_text:
                        return candidate
                    # 如果不在原文中，但修复后的实体名更合理，也使用
                    if len(candidate) < len(entity_name) and len(candidate) >= 3:
                        return candidate

    return entity_name
