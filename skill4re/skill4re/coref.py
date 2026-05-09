import json
import logging
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from skill4re.backends import LocalQwenGenerator, generate_text, generate_text_with_requests
from skill4re.parsing import parse_json

logger = logging.getLogger(__name__)

COREF_PROMPT_TEMPLATE = """请识别以下军事文档中的共指关系。
将所有指向同一实体的不同表述归一化为最完整的名称。

要求：
1. 只收录文档中实际出现的别名，不要推测
2. 优先使用最完整的实体名称作为主键（如"第3机械化步兵营"优于"第3营"）
3. 代词（它、其、该部、该群）如果指向明确实体也要收录
4. 地点的简称和全称都要收录（如"A高地"和"该高地"）
5. 不要收录过于泛化的词（如"敌方"、"我方"、"部队"）

只输出 JSON 映射表，格式：
{{"第3机械化步兵营": ["该营", "第3营", "其"], "A高地": ["该高地"]}}

文档：
{doc_text}"""


def build_coref_prompt(doc_text: str) -> str:
    return COREF_PROMPT_TEMPLATE.format(doc_text=doc_text)


def resolve_coreferences(
    doc_text: str,
    backend: str,
    client: Optional[OpenAI],
    local_generator: Optional[LocalQwenGenerator],
    model: str,
    max_tokens: int = 1500,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    request_timeout: int = 120,
    enable_thinking: bool = False,
) -> Tuple[str, Dict[str, List[str]]]:
    """对文档做共指消解，返回 (替换后的文档, 映射表)。

    仅对长文档调用，短文档跳过以节省 LLM 调用。
    """
    prompt = build_coref_prompt(doc_text)
    try:
        if client is not None or local_generator is not None:
            content = generate_text(
                prompt=prompt,
                backend=backend,
                api_client=client,
                local_generator=local_generator,
                model=model,
                max_tokens=max_tokens,
            )
        else:
            content = generate_text_with_requests(
                prompt=prompt,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                backend=backend,
                base_url=base_url,
                timeout=request_timeout,
                enable_thinking=enable_thinking,
            )
        mapping = parse_json(content)
        if not isinstance(mapping, dict):
            logger.warning("Coref returned non-dict, skipping: %s", type(mapping).__name__)
            return doc_text, {}

        # 过滤无效映射
        valid_mapping: Dict[str, List[str]] = {}
        for canonical, aliases in mapping.items():
            if not isinstance(aliases, list):
                continue
            canonical = canonical.strip()
            if not canonical or len(canonical) < 2:
                continue
            clean_aliases = []
            for alias in aliases:
                alias = str(alias).strip()
                if alias and len(alias) >= 2 and alias != canonical:
                    clean_aliases.append(alias)
            if clean_aliases:
                valid_mapping[canonical] = clean_aliases

        if not valid_mapping:
            return doc_text, {}

        # 替换文本中的别名（先替换长的，避免短别名误匹配）
        resolved = doc_text
        for canonical, aliases in valid_mapping.items():
            for alias in sorted(aliases, key=len, reverse=True):
                resolved = resolved.replace(alias, canonical)

        logger.info("Coref resolved %d entity groups", len(valid_mapping))
        return resolved, valid_mapping

    except Exception as exc:
        logger.warning("Coref resolution failed, using original text: %s: %s", type(exc).__name__, exc)
        return doc_text, {}
