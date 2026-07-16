from __future__ import annotations

import csv
import io

from app.api import CSV_COLUMNS, _relations_to_csv


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
