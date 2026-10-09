import json

import pytest

from app.agent.planning import plan_tasks, triage, validate_plan
from app.agent.relations import assign_relation_ids


def base():
    return assign_relation_ids([{"head": "第三营", "relation": "部署于", "tail": "河谷",
        "evidence": "第三营部署于河谷。", "source_sections": ["s1"]}])


@pytest.mark.parametrize("kind,updates,text", [
    ("missing_evidence", {"evidence": "第三营部署于城区。"}, "第三营部署于河谷。"),
    ("entity_not_in_evidence", {"head": "旅部"}, "第三营部署于河谷。"),
    ("low_confidence", {"evidence": ""}, "第三营部署于河谷。"),
    ("pronoun_entity", {"head": "该营"}, "第三营部署于河谷。"),
])
def test_relation_clues_hit_and_miss(kind, updates, text):
    original = base()
    assert kind not in {clue["kind"] for clue in triage(original, [], text)}
    original[0].update(updates)
    assert kind in {clue["kind"] for clue in triage(original, [], text)}


def test_similar_entities_in_different_sections_only():
    relations = base() + assign_relation_ids([{"head": "第三机械化营", "relation": "部署于", "tail": "城区",
        "evidence": "第三机械化营部署于城区。", "source_sections": ["s2"]}])
    # Existing fuzzy_entity_match treats substring names as candidates.
    relations[1]["head"] = "第三营部"
    assert "similar_entity" in {c["kind"] for c in triage(relations, [], "")}
    relations[1]["source_sections"] = ["s1"]
    assert "similar_entity" not in {c["kind"] for c in triage(relations, [], "")}


def test_sparse_long_section_hit_and_miss():
    relations = base()
    relations += [{**relations[0], "relation_id": f"r{i}", "source_sections": ["s2"]} for i in range(3)]
    sections = [{"section_id": f"s{i}", "text": "原文" * 100} for i in range(1, 4)]
    assert "sparse_section" in {c["kind"] for c in triage(relations, sections, "")}
    relations.append({**relations[0], "relation_id": "r4", "source_sections": ["s3"]})
    assert "sparse_section" not in {c["kind"] for c in triage(relations, sections, "")}


def test_model_plan_validation_and_task_limit():
    relations = base()
    sections = [{"section_id": "s1"}]
    planned = [{"type": "verify_evidence", "target": relations[0]["relation_id"], "reason": "核查"},
               {"type": "normalize_entity", "target": "第三营"},
               {"type": "check_section", "target": "missing"}]
    tasks, meta = plan_tasks(lambda prompt: json.dumps({"tasks": planned}), relations, sections,
                             "第三营部署于河谷。", 1)
    assert tasks[0]["target"] == relations[0]["relation_id"] and len(tasks) == 1
    assert meta["truncated_count"] == 1 and meta["rejected"][0]["reason"] == "目标不存在"
    assert not meta["degraded"]


@pytest.mark.parametrize("response", ["乱码", RuntimeError("offline")])
def test_planning_failure_uses_severity_sorted_rule_tasks(response):
    relations = base()
    relations[0]["evidence"] = "不存在"
    relations[0]["head"] = "该营"
    def call(prompt):
        if isinstance(response, Exception):
            raise response
        return response
    tasks, meta = plan_tasks(call, relations, [], "第三营部署于河谷。", 20)
    assert meta["degraded"] and tasks[0]["type"] == "verify_evidence"
    assert tasks[0]["reason"] == "证据在全文中定位不到"
    assert tasks[1]["type"] == "normalize_entity"
