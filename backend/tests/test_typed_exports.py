import copy
import csv
import io
import json
from pathlib import Path

import pytest

from app.extraction_export import entities_to_csv, typed_relations_to_csv, write_csv
from app.pipeline import _typed_sources_and_batches
from app.types import OcrPage
from skill4re.extraction_schema import ExtractionValidationError, public_result, validate_result


def test_documented_result_and_csv():
    path = Path(__file__).resolve().parents[2] / "docs/examples/typed-result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    validate_result(result)
    rows = list(csv.DictReader(io.StringIO(entities_to_csv(result).lstrip("\ufeff"))))
    assert len(rows) == 1
    assert rows[0]["实体ID"] == result["entities"][0]["entity_id"]
    assert set(rows[0]) == {"实体ID", "实体名称", "实体类型", "别名"}


def test_csv_formula_protection_preserves_numbers():
    rows = list(csv.DictReader(io.StringIO(write_csv(["value", "number"], [{"value": "=1+1", "number": -31.5}, {"value": " \t=1+1", "number": None}]).lstrip("\ufeff"))))
    assert rows[0] == {"value": "'=1+1", "number": "-31.5"}
    assert rows[1]["value"] == "' \t=1+1"


def test_ocr_block_origin_is_only_assigned_when_unique():
    text = "甲单位位于乙港。"
    pages = [OcrPage(page_index=2, markdown_text=text, pruned_result={"blocks": [{"block_content": text, "global_block_id": 7}]})]
    sources, _ = _typed_sources_and_batches(text, [], [(2, text)], 2500, ocr_pages=pages)
    assert sources[0]["page"] == 2 and sources[0]["block_id"] == 7
    sources, _ = _typed_sources_and_batches(text, [], [(2, text), (3, text)], 2500, ocr_pages=pages + [OcrPage(page_index=3, markdown_text=text, pruned_result={"blocks": [{"block_content": text, "global_block_id": 8}]})])
    assert sources[0]["page"] is None and sources[0]["block_id"] is None


@pytest.mark.parametrize("change", ["unknown_type", "disabled_relation", "self_relation"])
def test_result_validation_reports_invalid_fields(change):
    path = Path(__file__).resolve().parents[2] / "docs/examples/typed-result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    entity = result["entities"][0]
    if change == "unknown_type":
        entity["type_ids"] = ["unknown"]
    else:
        if change == "disabled_relation":
            result["extraction_schema"]["relation_types"][0]["enabled"] = False
        result["relations"] = [{"relation_id": "r1", "relation_type_id": "command", "head_id": entity["entity_id"], "tail_id": entity["entity_id"]}]
    with pytest.raises(ExtractionValidationError):
        validate_result(result)


def test_historical_projection_preserves_entities_without_result_sources():
    result = json.loads((Path(__file__).parent / "fixtures/typed-result-v2.json").read_text(encoding="utf-8"))
    result["stage_outputs"] = {"timing": {"total_elapsed_seconds": 1}, "requests": [], "extraction": {"evidence": "处理上下文"}}
    original = copy.deepcopy(result)
    projected = public_result(result)
    assert result == original
    assert projected["result_schema_version"] == 4
    assert projected == public_result(projected)
    assert projected["stage_outputs"] == {"timing": {"total_elapsed_seconds": 1}, "requests": []}
    entity = projected["entities"][0]
    assert set(entity) == {"entity_id", "name", "type_ids", "aliases"}
    assert entities_to_csv(original) == entities_to_csv(projected)


def test_public_results_and_exports_accept_relations_without_sources():
    result = json.loads((Path(__file__).resolve().parents[2] / "docs/examples/typed-result.json").read_text(encoding="utf-8"))
    other = copy.deepcopy(result["entities"][0])
    other.update(entity_id="entity_000002", name="联合指挥所乙")
    result["entities"].append(other)
    result["relations"] = [{"relation_id": "r1", "relation_type_id": "command", "head_id": "entity_000001", "tail_id": "entity_000002"}]
    validate_result(result)
    rows = list(csv.DictReader(io.StringIO(typed_relations_to_csv(result).lstrip("\ufeff"))))
    assert rows[0]["头实体"] == "联合指挥所甲" and rows[0]["尾实体"] == "联合指挥所乙"
    assert not any("来源" in key or "证据" in key for key in rows[0])
    entities = list(csv.DictReader(io.StringIO(entities_to_csv(result).lstrip("\ufeff"))))
    assert len(entities) == 2
    assert not any("来源" in key or "证据" in key for key in entities[0])


def test_legacy_result_projection_keeps_business_fields():
    original = {"document_text": "甲机构指挥乙机构。", "final_relations": [
        {"head": "甲机构", "relation": "指挥", "tail": "乙机构", "evidence": "甲机构指挥乙机构。", "source_paragraphs": [{"content": "甲机构指挥乙机构。"}]},
    ]}
    assert public_result(original) == {"document_text": original["document_text"], "final_relations": [{"head": "甲机构", "relation": "指挥", "tail": "乙机构"}]}


@pytest.mark.parametrize("container", ["final_relation_list", "wrapped_list", "final_relations", "skill4re_result", "empty_primary"])
@pytest.mark.parametrize("head_key,tail_key", [("head", "tail"), ("subject", "object"), ("head_entity", "tail_entity")])
def test_legacy_entity_export_reads_every_supported_relation_container(container, head_key, tail_key):
    relations = [{head_key: "指挥所甲", "relation": "指挥", tail_key: "前出分队乙", "head_type": "指挥机构", "tail_type": "前出分队"}]
    if container == "wrapped_list":
        result = {"final_relation_list": {"relation_list": relations}}
    elif container == "skill4re_result":
        result = {"skill4re_result": {"relation_list": relations}}
    elif container == "empty_primary":
        result = {"final_relation_list": [], "final_relations": relations}
    else:
        result = {container: relations}
    original = copy.deepcopy(result)
    rows = list(csv.DictReader(io.StringIO(entities_to_csv(result).lstrip("\ufeff"))))
    assert [(row["实体名称"], row["实体类型"]) for row in rows] == [("指挥所甲", "指挥机构"), ("前出分队乙", "前出分队")]
    assert result == original
