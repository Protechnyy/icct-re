from __future__ import annotations

import csv
import io

from app.api import CSV_COLUMNS, _relations_to_csv
from app.api import create_app
from test_pipeline import build_config


def test_relations_to_csv_exports_final_relations_with_sources() -> None:
    payload = {
        "final_relation_list": {
            "relation_list": [
                {
                    "head": "铁幕回声行动",
                    "relation": "文件等级",
                    "tail": "机密",
                    "evidence": "文件等级：机密。",
                    "skill": "military",
                    "source_sections": ["六"],
                    "source_pages": [2],
                    "source_blocks": [12],
                    "source_paragraphs": [{"content": "文件等级：机密。"}],
                    "source_batch_index": 3,
                }
            ]
        }
    }

    csv_text = _relations_to_csv(payload)
    rows = list(csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff"))))

    assert csv_text.startswith("\ufeff")
    assert tuple(rows[0].keys()) == CSV_COLUMNS
    assert rows[0]["主体"] == "铁幕回声行动"
    assert rows[0]["来源章节"] == "六"
    assert rows[0]["来源页码"] == "2"
    assert rows[0]["来源段落"] == "文件等级：机密。"


def test_relations_to_csv_writes_headers_for_empty_result() -> None:
    csv_text = _relations_to_csv({"final_relations": []})
    rows = list(csv.reader(io.StringIO(csv_text.lstrip("\ufeff"))))

    assert rows == [list(CSV_COLUMNS)]


def test_relations_to_csv_prevents_spreadsheet_formula_injection() -> None:
    payload = {
        "final_relations": [
            {"head": "=1+1", "relation": "+SUM(A1:A2)", "tail": "@command"}
        ]
    }

    row = next(csv.DictReader(io.StringIO(_relations_to_csv(payload).lstrip("\ufeff"))))

    assert row["主体"] == "'=1+1"
    assert row["关系"] == "'+SUM(A1:A2)"
    assert row["客体"] == "'@command"


def test_result_and_status_interfaces_return_agent_fields(tmp_path, monkeypatch):
    config = build_config(tmp_path)
    agent_result = {"status": "completed", "summary": {"llm_calls": 2}, "trace": [{"kind": "plan"}]}
    progress = {"total_tasks": 1, "completed_tasks": 0, "current_task": {"id": "t1"}, "recent_events": []}
    class Store:
        def get_task(self, task_id):
            return {"task_id": task_id, "filename": "sample.pdf", "stage": "agent_verification",
                    "agent_progress": progress, "payload": {"file_path": "private"}}
        def get_result(self, task_id):
            return {"agent_result": agent_result, "pre_agent_relations": [], "final_relations": []}
    monkeypatch.setattr("app.api.AppConfig.from_env", lambda: config)
    monkeypatch.setattr("app.api.RedisTaskStore", lambda url: Store())
    client = create_app().test_client()
    response = client.get("/api/result/task-1")
    assert response.status_code == 200 and response.json["agent_result"] == agent_result
    state = client.get("/api/status/task-1")
    assert state.json["agent_progress"] == progress and "payload" not in state.json
    exported = client.get("/api/result/task-1/csv")
    assert exported.status_code == 200
    assert list(csv.reader(io.StringIO(exported.data.decode("utf-8-sig")))) == [list(CSV_COLUMNS)]
