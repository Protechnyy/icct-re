"""Validated edits and document memory; callers never mutate the input snapshot."""
from collections import Counter
from copy import deepcopy
import json

from .document import compact, query_entity
from .relations import assign_relation_ids


class RelationWorkspace:
    def __init__(self, relations, index, max_added_per_task=5):
        self.relations = assign_relation_ids(relations)
        for relation in self.relations:
            relation["verification"] = {"status": "unchecked", "task_ids": []}
        self.index = index
        self.max_added_per_task = max_added_per_task
        self.added_by_task = Counter()
        self.removed_relations = []
        self.entity_aliases = {}
        self.conclusions = []
        self.changes = []

    def find(self, relation_id):
        return next((item for item in self.relations if item["relation_id"] == relation_id), None)

    def mark(self, relation, task_id, status):
        verification = relation["verification"]
        verification["status"] = status
        if task_id not in verification["task_ids"]:
            verification["task_ids"].append(task_id)

    def source_fields(self, evidence):
        located = self.index.locate_evidence(evidence)
        if not located["matched"]:
            raise ValueError("证据无法在原文中定位")
        locations = located["locations"]
        return {"source_sections": list(dict.fromkeys(p["section_id"] for p in locations if p["section_id"])),
                "source_pages": list(dict.fromkeys(p["page"] for p in locations if p.get("page") is not None)),
                "source_blocks": list(dict.fromkeys(p["global_block_id"] for p in locations if p.get("global_block_id") is not None)),
                "source_paragraphs": deepcopy(locations)}

    def record(self, kind, task_id, before, after, **extra):
        ids = list(dict.fromkeys(item["relation_id"] for item in [*before, *after]))
        change = {"type": kind, "task_id": task_id, "relation_ids": ids,
                  "before": deepcopy(before), "after": deepcopy(after), **deepcopy(extra)}
        self.changes.append(change)
        return {"ok": True, "change": change}

    def correct_relation(self, relation_id, replacement_json, task_id):
        relation = self.find(relation_id)
        if relation is None:
            return {"error": "关系编号不存在"}
        try:
            replacement = json.loads(replacement_json)
            if not isinstance(replacement, dict) or not str(replacement.get("evidence", "")).strip():
                raise ValueError("更正必须提供证据引文")
            update = {key: replacement[key] for key in ("head", "relation", "tail", "evidence") if key in replacement}
            if any(not isinstance(value, str) or not value.strip() for value in update.values()):
                raise ValueError("关系字段必须是非空字符串")
            sources = self.source_fields(update["evidence"])
        except (ValueError, TypeError) as exc:
            return {"error": str(exc)}
        before = deepcopy(relation)
        relation.update(update)
        relation.update(sources)
        self.mark(relation, task_id, "added" if relation["verification"]["status"] == "added" else "corrected")
        return self.record("correction", task_id, [before], [relation], evidence=update["evidence"],
                           pages=sources["source_pages"])

    def add_relation(self, relation_json, task_id):
        if self.added_by_task[task_id] >= self.max_added_per_task:
            return {"error": "单任务新增已达上限"}
        try:
            proposal = json.loads(relation_json)
            if not isinstance(proposal, dict):
                raise ValueError("新增关系必须是 JSON 对象")
            relation = {key: proposal.get(key, "") for key in ("head", "relation", "tail", "evidence", "skill")}
            if any(not isinstance(relation[key], str) or not relation[key].strip()
                   for key in ("head", "relation", "tail", "evidence")):
                raise ValueError("新增必须提供头实体、关系、尾实体和证据")
            relation.update(self.source_fields(relation["evidence"]))
        except (ValueError, TypeError) as exc:
            return {"error": str(exc)}
        key = tuple(compact(relation[field]).lower() for field in ("head", "relation", "tail", "evidence"))
        if any(tuple(compact(item.get(field)).lower() for field in ("head", "relation", "tail", "evidence")) == key
               for item in self.relations):
            return {"error": "关系已存在"}
        identity = assign_relation_ids([relation])[0]["relation_id"]
        all_ids = {item["relation_id"] for item in self.relations + self.removed_relations}
        serial = 1
        while identity in all_ids:
            serial += 1
            identity = assign_relation_ids([relation])[0]["relation_id"] + f"-{serial}"
        relation["relation_id"] = identity
        relation["verification"] = {"status": "added", "task_ids": [task_id]}
        self.relations.append(relation)
        self.added_by_task[task_id] += 1
        return self.record("addition", task_id, [], [relation], evidence=relation["evidence"],
                           pages=relation["source_pages"])

    def delete_relation(self, relation_id, reason, task_id):
        if not str(reason).strip():
            return {"error": "删除必须说明理由"}
        relation = self.find(relation_id)
        if relation is None:
            return {"error": "关系编号不存在"}
        before = deepcopy(relation)
        self.removed_relations.append({**before, "reason": reason, "task_id": task_id})
        self.relations.remove(relation)
        return self.record("deletion", task_id, [before], [], reason=reason,
                           evidence=before.get("evidence", ""), pages=before.get("source_pages", []))

    def merge_entities(self, canonical_name, aliases_json, task_id):
        if not compact(canonical_name) or compact(canonical_name) not in compact(self.index.document_text):
            return {"error": "规范名称不在原文中"}
        try:
            aliases = json.loads(aliases_json)
            if not isinstance(aliases, list) or not aliases or any(not isinstance(name, str) or not name.strip() for name in aliases):
                raise ValueError("别名必须是非空字符串列表")
        except (ValueError, TypeError) as exc:
            return {"error": str(exc)}
        aliases = list(dict.fromkeys(name for name in aliases if name != canonical_name))
        known = {item[field] for item in self.relations for field in ("head", "tail")}
        if not aliases or not any(name in known or name in self.entity_aliases for name in aliases):
            return {"error": "别名在关系图中不存在"}
        before = deepcopy([item for item in self.relations if any(item[field] in aliases for field in ("head", "tail"))])
        memory_aliases = list(self.entity_aliases.get(canonical_name, []))
        for alias in aliases:
            memory_aliases.extend([alias, *self.entity_aliases.pop(alias, [])])
        self.entity_aliases[canonical_name] = list(dict.fromkeys(name for name in memory_aliases if name != canonical_name))
        for item in self.relations:
            changed = False
            for field in ("head", "tail"):
                if item[field] in aliases:
                    item[field] = canonical_name
                    changed = True
            if changed:
                self.mark(item, task_id, "added" if item["verification"]["status"] == "added" else "corrected")
        merged = {}
        for item in self.relations:
            key = tuple(compact(item.get(field)).lower() for field in ("head", "relation", "tail", "evidence"))
            if key not in merged:
                merged[key] = item
            else:
                kept = merged[key]
                for field in ("source_sections", "source_pages", "source_blocks", "source_paragraphs"):
                    for value in item.get(field, []):
                        if value not in kept.setdefault(field, []):
                            kept[field].append(value)
                for identity in item["verification"]["task_ids"]:
                    self.mark(kept, identity, "added" if kept["verification"]["status"] == "added" else "corrected")
                self.removed_relations.append({**deepcopy(item), "reason": "实体合并后重复", "task_id": task_id})
        self.relations = list(merged.values())
        affected_ids = {item["relation_id"] for item in before}
        after = [item for item in self.relations if item["relation_id"] in affected_ids
                 or task_id in item["verification"]["task_ids"]]
        return self.record("entity_merge", task_id, before, after, canonical_name=canonical_name,
                           aliases=aliases, evidence=canonical_name,
                           pages=list(dict.fromkeys(p["page"] for p in self.index.locate_evidence(canonical_name)["locations"]
                                                    if p.get("page") is not None)))

    def query_entity(self, name):
        return query_entity(self.relations, self.entity_aliases, name)

    def memory_for(self, task, limit=800):
        target = str(task["target"])
        names = {target}
        sections = {target}
        relation = self.find(target)
        if relation:
            names.update((relation.get("head", ""), relation.get("tail", "")))
            sections.update(str(value) for value in relation.get("source_sections", []))
        if task["type"] == "check_section":
            names.update(item[field] for item in self.relations for field in ("head", "tail")
                         if target in item.get("source_sections", []))
        lines = []
        for canonical, aliases in self.entity_aliases.items():
            if names & {canonical, *aliases}:
                lines.append(f"别名：{canonical} = {', '.join(aliases)}")
        for entry in self.conclusions:
            if names & set(entry.get("entities", [])) or sections & set(entry.get("sections", [])):
                lines.append("已确认结论：" + entry["conclusion"])
        return "\n".join(lines)[:limit]
