import json as _json
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SkillExample:
    """一条 few-shot 示例。

    文档级格式：
        text: 多句段落（3-8 句）
        json: {"relation_list": [{"head":..., "relation":..., "tail":..., "evidence":..., "skill":...}, ...]}
    """
    text: str
    json: str

    @classmethod
    def from_dict(cls, item: Dict) -> "SkillExample":
        # 兼容旧格式（单条关系）和新格式（relation_list）
        raw_json = item["json"]
        if isinstance(raw_json, dict):
            raw_json = _json.dumps(raw_json, ensure_ascii=False)
        return cls(text=item["text"], json=raw_json)

    def parse_relations(self) -> List[Dict]:
        """解析 json 字段为关系列表，兼容单条和文档级格式。"""
        parsed = _json.loads(self.json)
        if isinstance(parsed, dict) and "relation_list" in parsed:
            return parsed["relation_list"]
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        return []

    @property
    def is_document_level(self) -> bool:
        """判断是否为文档级示例（多条关系）。"""
        return len(self.parse_relations()) > 1


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    focus: str
    head_prior: str
    tail_prior: str
    relation_style: str
    negative_scope: str
    keywords: List[str]
    fewshot: List[SkillExample]
    extraction_rules: List[str]

    @classmethod
    def from_dict(cls, payload: Dict) -> "Skill":
        return cls(
            name=payload["name"],
            description=payload["description"],
            focus=payload["focus"],
            head_prior=payload["head_prior"],
            tail_prior=payload["tail_prior"],
            relation_style=payload["relation_style"],
            negative_scope=payload["negative_scope"],
            keywords=list(payload.get("keywords", [])),
            fewshot=[
                SkillExample.from_dict(item)
                for item in payload.get("fewshot", [])
            ],
            extraction_rules=list(payload.get("extraction_rules", [])),
        )
