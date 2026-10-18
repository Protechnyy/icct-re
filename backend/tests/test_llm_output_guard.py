import json

from app.llm_output_guard import OutputRepetitionGuard


def test_distinct_relations_can_share_heads_and_sources():
    payload = {"relations": [{"relation_id": f"r{index}", "relation_type_id": "command", "head_id": "a",
                              "tail_id": f"b{index}", "evidence": [{"source_id": "s1"}]} for index in range(15)]}
    guard = OutputRepetitionGuard()
    text = json.dumps(payload)
    assert all(guard.feed(text[index:index + 41]) is None for index in range(0, len(text), 41))


def test_repeated_relations_are_detected_before_json_is_complete():
    relations = [{"relation_id": f"r{index}", "relation_type_id": "command", "head_id": "a",
                  "tail_id": "b", "evidence": [{"source_id": "s1"}]} for index in range(20)]
    text = json.dumps({"relations": relations})[:-10]
    assert "重复同一关系" in OutputRepetitionGuard().feed(text)


def test_repeated_source_references_are_detected_before_json_is_complete():
    text = json.dumps({"entities": [{"mentions": [{"source_id": "s1"}] * 80}]})[:-10]
    assert "重复来源编号" in OutputRepetitionGuard().feed(text)


def test_different_evidence_for_same_relation_is_preserved():
    payload = {"relations": [{"relation_type_id": "command", "head_id": "a", "tail_id": "b",
                              "evidence": [{"source_id": f"s{index}"}]} for index in range(20)]}
    assert OutputRepetitionGuard().feed(json.dumps(payload)) is None


def test_excessive_whitespace_is_detected_across_chunks():
    guard = OutputRepetitionGuard()
    assert guard.feed('{"relations": [') is None
    for _ in range(10):
        assert guard.feed(" " * 100) is None
    assert "空白字符" in guard.feed(" " * 100)
