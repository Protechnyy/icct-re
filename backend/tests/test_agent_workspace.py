from copy import deepcopy
import json

from app.agent.document import DocumentIndex
from app.agent.workspace import RelationWorkspace


def make_workspace(max_added=5):
    text = "第三营部署于河谷。该营由旅部指挥。旅部驻扎于城区。"
    index = DocumentIndex([{"section_id": "s1", "paragraphs": [{"content": text, "page": 3}]}], text)
    relations = [{"head": "该营", "relation": "部署于", "tail": "河谷", "evidence": "第三营部署于河谷。"},
                 {"head": "第三营", "relation": "部署于", "tail": "河谷", "evidence": "第三营部署于河谷。"}]
    return RelationWorkspace(relations, index, max_added), relations


def test_workspace_does_not_mutate_original():
    workspace, relations = make_workspace()
    original = deepcopy(relations)
    workspace.merge_entities("第三营", '["该营"]', "t1")
    assert relations == original
    assert all("relation_id" not in item for item in relations)


def test_correction_requires_evidence_and_updates_sources():
    workspace, _ = make_workspace()
    identity = workspace.relations[0]["relation_id"]
    before = deepcopy(workspace.relations)
    assert "error" in workspace.correct_relation(identity, '{"evidence":"虚构句子"}', "t1")
    assert workspace.relations == before
    result = workspace.correct_relation(identity, '{"head":"第三营","evidence":"第三营 部署于河谷。"}', "t1")
    assert result["ok"]
    relation = workspace.find(identity)
    assert relation["source_sections"] == ["s1"] and relation["source_pages"] == [3]
    assert relation["verification"] == {"status": "corrected", "task_ids": ["t1"]}


def test_addition_evidence_and_per_task_limit():
    workspace, _ = make_workspace(max_added=1)
    invalid = json.dumps({"head": "旅部", "relation": "驻扎于", "tail": "城区", "evidence": "凭空生成"})
    assert "error" in workspace.add_relation(invalid, "t1")
    valid = json.dumps({"head": "旅部", "relation": "驻扎于", "tail": "城区", "evidence": "旅部驻扎于城区。"})
    assert workspace.add_relation(valid, "t1")["ok"]
    assert workspace.relations[-1]["source_pages"] == [3]
    assert "上限" in workspace.add_relation(valid, "t1")["error"]


def test_deletion_requires_reason_and_archives_content():
    workspace, _ = make_workspace()
    before = deepcopy(workspace.relations[0])
    identity = before["relation_id"]
    assert "error" in workspace.delete_relation(identity, " ", "t1")
    assert workspace.delete_relation(identity, "原文不支持该关系", "t1")["ok"]
    assert workspace.find(identity) is None
    assert workspace.removed_relations[0] == {**before, "reason": "原文不支持该关系", "task_id": "t1"}


def test_merge_validates_original_name_and_deduplicates():
    workspace, _ = make_workspace()
    before = deepcopy(workspace.relations)
    assert "error" in workspace.merge_entities("第四营", '["该营"]', "t1")
    assert workspace.relations == before
    result = workspace.merge_entities("第三营", '["该营"]', "t1")
    assert result["ok"] and len(workspace.relations) == 1
    assert workspace.entity_aliases == {"第三营": ["该营"]}
    assert workspace.removed_relations[0]["reason"] == "实体合并后重复"


def test_memory_includes_confirmed_aliases_and_is_bounded():
    workspace, _ = make_workspace()
    workspace.merge_entities("第三营", '["该营"]', "t1")
    workspace.conclusions.append({"entities": ["第三营"], "sections": ["s1"], "conclusion": "该营是第三营" * 1000})
    task = {"type": "verify_evidence", "target": workspace.relations[0]["relation_id"]}
    memory = workspace.memory_for(task)
    assert "该营" in memory and "第三营" in memory and len(memory) <= 800
    other, _ = make_workspace()
    assert not other.entity_aliases and not other.conclusions
