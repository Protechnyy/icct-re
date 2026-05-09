import json
import re
from pathlib import Path
from typing import Dict, List


SECTION_MARKER_RE = re.compile(
    r"(^|(?<=[\n。！？!?；;])\s*)"
    r"(?P<marker>"
    r"第[一二三四五六七八九十百千万\d]+阶段"
    r"|[（(][一二三四五六七八九十百千万\d]+[）)]"
    r"|\d+[.．、](?=\s|[\u4e00-\u9fffA-Za-z])"
    r"|[一二三四五六七八九十]+[、.．]"
    r")"
)


def load_samples(path: Path, limit: int) -> List[Dict]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if len(samples) >= limit:
                break
            obj = json.loads(line)
            samples.append(
                {
                    "sample_index": idx,
                    "sample_id": obj.get("sample_id"),
                    "input_meta": obj.get("input", {}),
                    "doc_text": obj.get("output", ""),
                }
            )
    return samples


def approx_token_count(text: str) -> int:
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_tokens = re.findall(r"[A-Za-z0-9_+\-]+", text)
    punctuation = re.findall(r"[，。！？；：,.!?;:（）()【】\[\]“”\"'、】【\-]", text)
    return cjk_chars + len(latin_tokens) + max(1, len(punctuation) // 4)


def split_sentences(text: str) -> List[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？!?])\s+|(?<=[。！？!?])|(?<=\.)\s+\n*|\n+", text)
        if sentence.strip()
    ]
    if not sentences:
        return [text]
    return sentences


def split_by_sections(text: str) -> List[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    starts = [0]
    for match in SECTION_MARKER_RE.finditer(normalized):
        marker_start = match.start("marker")
        if marker_start > 0 and marker_start not in starts:
            starts.append(marker_start)

    if len(starts) == 1:
        section = normalized.strip()
        return [section] if section else []

    starts = sorted(starts)
    sections: List[str] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(normalized)
        section = normalized[start:end].strip()
        if section:
            sections.append(section)
    return sections


def split_section_to_units(section: str, token_budget: int) -> List[str]:
    if approx_token_count(section) <= token_budget:
        return [section]
    return split_sentences(section)


def chunk_document(text: str, token_budget: int) -> List[str]:
    sections = split_by_sections(text)
    units: List[str] = []
    for section in sections:
        units.extend(split_section_to_units(section, token_budget))

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = approx_token_count(unit)
        if current and current_tokens + unit_tokens > token_budget:
            chunks.append(" ".join(current).strip())
            current = [unit]
            current_tokens = unit_tokens
        else:
            current.append(unit)
            current_tokens += unit_tokens
    if current:
        chunks.append(" ".join(current).strip())
    return chunks
