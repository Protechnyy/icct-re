import csv
import json

import pytest

from app.agent.evaluation import aggregate_metrics, annotation_metrics, comparison, export_changes, replay_file, score_judgments
from app.agent.relations import assign_relation_ids
from test_agent_runtime import ScriptedModel, call_tool, configuration, plan


def test_replay_preserves_input_and_writes_separate_result(tmp_path):
    source = tmp_path / "input" / "result.json"
    source.parent.mkdir()
    relations = [{"head": "甲", "relation": "位于", "tail": "乙", "evidence": "甲位于乙。"}]
    source.write_text(json.dumps({"document_text": "甲位于乙。", "final_relations": relations}))
    stat = source.stat(); content = source.read_bytes()
    identity = assign_relation_ids(relations)[0]["relation_id"]
    model = ScriptedModel(replies=[plan(("verify_evidence", identity)), "原文支持。"])
    metrics = replay_file(source, tmp_path / "output", configuration(tmp_path), model=model)
    assert source.read_bytes() == content and source.stat().st_mtime_ns == stat.st_mtime_ns
    result = json.loads((tmp_path / "output/input/result.json").read_text())
    assert result["agent_result"]["status"] == "completed" and metrics["after"]["locatable_evidence_ratio"] == 1


def test_replay_rejects_missing_content_and_input_output_overlap(tmp_path):
    source = tmp_path / "input/result.json"; source.parent.mkdir()
    source.write_text('{}')
    with pytest.raises(ValueError, match="原文"):
        replay_file(source, tmp_path / "out", configuration(tmp_path))
    source.write_text('{"document_text":"原文"}')
    with pytest.raises(ValueError, match="关系列表"):
        replay_file(source, tmp_path / "out", configuration(tmp_path))
    with pytest.raises(ValueError, match="独立"):
        replay_file(source, source.parent, configuration(tmp_path))


def test_automatic_metrics_and_multi_document_aggregate():
    before = [{"head": "甲", "tail": "乙", "evidence": "甲位于乙。"}]
    after = before + [{"head": "乙", "tail": "丙", "evidence": "乙位于丙。"}]
    result = {"status": "completed", "summary": {"llm_calls": 3, "tokens": {"total_tokens": 8}, "elapsed_seconds": 1},
              "changes": [{"type": "addition"}]}
    metrics = comparison(before, after, "甲位于乙。乙位于丙。", result)
    assert metrics["before"]["relation_count"] == 1 and metrics["after"]["entity_count"] == 3
    assert metrics["changes"] == {"correction": 0, "deletion": 0, "addition": 1, "entity_merge": 0}
    summary = aggregate_metrics([metrics, metrics])
    assert summary["document_count"] == 2 and summary["llm_calls"] == 6
    assert summary["after"]["relation_count"] == 4 and summary["after"]["locatable_evidence_ratio"] == 1


def test_change_csv_and_judgment_accuracy(tmp_path):
    changes = [{"type": kind, "task_id": "t1", "before": [], "after": [], "evidence": "原文", "pages": [1]}
               for kind in ("correction", "correction", "deletion", "addition", "entity_merge")]
    path = tmp_path / "changes.csv"; export_changes(path, changes, "sample")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5 and all(row["judgment"] == "" for row in rows)
    for row, label in zip(rows, ("正确", "错误", "true", "", "1")):
        row["judgment"] = label
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    scores = score_judgments([path])
    assert scores["overall"]["accuracy"] == 0.75 and scores["overall"]["unreviewed"] == 1
    assert scores["correction"]["accuracy"] == 0.5 and scores["deletion"]["accuracy"] == 1
    assert scores["addition"]["accuracy"] is None


def test_annotation_precision_recall_f1_and_optional_skip():
    gold = [{"head": "甲", "relation": "位于", "tail": "乙"}, {"head": "乙", "relation": "位于", "tail": "丙"}]
    predicted = [{"head": " 甲 ", "relation": "位 于", "tail": "乙"},
                 {"head": "甲", "relation": "位于", "tail": "丁"}, {"head": "甲", "relation": "位于", "tail": "戊"}]
    metrics = annotation_metrics(predicted, gold)
    assert metrics["precision"] == 1 / 3 and metrics["recall"] == 0.5 and metrics["f1"] == 0.4
    assert comparison(predicted, gold, "", {"status": "completed"})["annotations"]["skipped"]
    assert not comparison(predicted, gold, "", {"status": "completed"}, gold)["annotations"].get("skipped")
