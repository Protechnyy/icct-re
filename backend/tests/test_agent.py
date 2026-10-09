from copy import deepcopy
from dataclasses import replace

from app.agent import verify_document
from test_pipeline import build_config


def test_disabled_preserves_relations_and_does_not_call_model(tmp_path):
    relations = [{"head": "甲", "relation": "位于", "tail": "乙", "evidence": "甲位于乙。"}]
    original = deepcopy(relations)
    class FailModel:
        def invoke(self, *args, **kwargs):
            raise AssertionError("disabled verification must not invoke a model")
    output, result = verify_document(relations, [], "甲位于乙。", build_config(tmp_path), model=FailModel())
    assert output == original == relations
    assert output is not relations
    assert result == {"status": "disabled"}


def test_empty_relations_are_skipped(tmp_path):
    output, result = verify_document([], [], "原文", replace(build_config(tmp_path), agent_enabled=True))
    assert output == [] and result["status"] == "skipped"


def test_relation_ids_are_unique_stable_and_do_not_mutate_input():
    from app.agent.relations import assign_relation_ids
    relations = [{"head": "甲", "relation": "位于", "tail": "乙"}] * 2
    first = assign_relation_ids(relations)
    assert first == assign_relation_ids(relations)
    assert len({item["relation_id"] for item in first}) == 2
    assert "relation_id" not in relations[0]
