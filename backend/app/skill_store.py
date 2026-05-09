from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import AppConfig


SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
REQUIRED_STRING_FIELDS = (
    "name",
    "description",
    "focus",
    "head_prior",
    "tail_prior",
    "relation_style",
    "negative_scope",
)
REQUIRED_LIST_FIELDS = ("extraction_rules", "keywords", "fewshot")


class SkillStoreError(ValueError):
    pass


class SkillStore:
    def __init__(self, config: AppConfig) -> None:
        self.skills_dir = config.skill4re_skills_dir

    def list_skills(self) -> list[dict[str, Any]]:
        skills = []
        for path in sorted(self.skills_dir.glob("*.json")):
            payload = self._read_skill(path)
            payload["_meta"] = {
                "filename": path.name,
                "path": str(path),
                "updated_at": path.stat().st_mtime,
            }
            skills.append(payload)
        return skills

    def get_skill(self, name: str) -> dict[str, Any]:
        path = self._skill_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Skill not found: {name}")
        payload = self._read_skill(path)
        payload["_meta"] = {
            "filename": path.name,
            "path": str(path),
            "updated_at": path.stat().st_mtime,
        }
        return payload

    def create_skill(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = self._validate_skill(payload)
        path = self._skill_path(normalized["name"])
        if path.exists():
            raise FileExistsError(f"Skill already exists: {normalized['name']}")
        self._write_skill(path, normalized)
        return self.get_skill(normalized["name"])

    def update_skill(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        current_path = self._skill_path(name)
        if not current_path.exists():
            raise FileNotFoundError(f"Skill not found: {name}")

        normalized = self._validate_skill(payload)
        next_path = self._skill_path(normalized["name"])
        if next_path != current_path and next_path.exists():
            raise FileExistsError(f"Skill already exists: {normalized['name']}")

        self._write_skill(next_path, normalized)
        if next_path != current_path:
            current_path.unlink()
        return self.get_skill(normalized["name"])

    def _read_skill(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SkillStoreError(f"Invalid JSON in {path.name}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SkillStoreError(f"Skill file must contain a JSON object: {path.name}")
        return payload

    def _write_skill(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _skill_path(self, name: str) -> Path:
        if not SKILL_NAME_RE.fullmatch(name):
            raise SkillStoreError("Skill name can only contain letters, numbers, underscore, and hyphen.")
        path = (self.skills_dir / f"{name}.json").resolve()
        skills_dir = self.skills_dir.resolve()
        if skills_dir not in path.parents:
            raise SkillStoreError("Invalid skill path.")
        return path

    def _validate_skill(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise SkillStoreError("Skill payload must be a JSON object.")

        normalized: dict[str, Any] = {}
        for field in REQUIRED_STRING_FIELDS:
            value = str(payload.get(field, "")).strip()
            if not value:
                raise SkillStoreError(f"Missing required field: {field}")
            normalized[field] = value

        if not SKILL_NAME_RE.fullmatch(normalized["name"]):
            raise SkillStoreError("Skill name can only contain letters, numbers, underscore, and hyphen.")

        for field in REQUIRED_LIST_FIELDS:
            value = payload.get(field)
            if not isinstance(value, list):
                raise SkillStoreError(f"Field must be a list: {field}")
            normalized[field] = value

        normalized["extraction_rules"] = self._normalize_string_list(
            normalized["extraction_rules"], "extraction_rules"
        )
        normalized["keywords"] = self._normalize_string_list(normalized["keywords"], "keywords")
        normalized["fewshot"] = self._normalize_fewshot(normalized["fewshot"])
        return normalized

    def _normalize_string_list(self, items: list[Any], field: str) -> list[str]:
        normalized = [str(item).strip() for item in items if str(item).strip()]
        if not normalized:
            raise SkillStoreError(f"Field must contain at least one item: {field}")
        return normalized

    def _normalize_fewshot(self, items: list[Any]) -> list[dict[str, Any]]:
        normalized = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                raise SkillStoreError(f"fewshot[{index}] must be an object.")
            text = str(item.get("text", "")).strip()
            raw_json = item.get("json")
            if not text:
                raise SkillStoreError(f"fewshot[{index}].text is required.")
            if isinstance(raw_json, dict):
                json_text = json.dumps(raw_json, ensure_ascii=False)
            else:
                json_text = str(raw_json or "").strip()
            if not json_text:
                raise SkillStoreError(f"fewshot[{index}].json is required.")
            try:
                parsed = json.loads(json_text)
            except json.JSONDecodeError as exc:
                raise SkillStoreError(f"fewshot[{index}].json must be valid JSON: {exc}") from exc
            relation_list = parsed.get("relation_list") if isinstance(parsed, dict) else None
            if not isinstance(relation_list, list):
                raise SkillStoreError(f"fewshot[{index}].json must contain relation_list.")
            normalized.append(
                {
                    "text": text,
                    "json": json.dumps(parsed, ensure_ascii=False),
                    "is_document_level": bool(item.get("is_document_level", len(relation_list) > 1)),
                }
            )
        if not normalized:
            raise SkillStoreError("Field must contain at least one item: fewshot")
        return normalized
