import pytest

from app.pipeline import _build_relation_batches, _build_relation_sections, _typed_sources_and_batches
from app.skill4re_client import provider_schema
from skill4re.extraction_schema import ExtractionSchemaSnapshot, ExtractionValidationError, default_config, validate_structure
from skill4re.typed_extraction import (
    ConfiguredExtractor, compact_json, identity_candidates, merge_entities, relation_context_groups,
    resolve_entities, resolve_evidence, resolve_relations, resolve_source_references,
)


def source_record(text, identifier, document):
    start = document.index(text)
    return {"source_id": identifier, "text": text, "start": start, "end": start + len(text),
            "page": None, "section_id": None, "block_id": None, "batch_index": 9}


def test_unknown_source_and_modified_quote_fail():
    document = "联合指挥所甲指挥前出分队乙。"
    source = source_record(document, "s1", document)
    with pytest.raises(ExtractionValidationError, match="source_id"):
        resolve_source_references([{"source_id": "unknown"}], {"s1": source})
    with pytest.raises(ExtractionValidationError, match="没有原样"):
        resolve_evidence([{"source_id": "s1", "text": "联合指挥所甲指挥前出分队丙。"}], {"s1": source})


def test_normalized_parallel_name_preserves_original_identity_evidence():
    document = "第10空军集团军：辖第1歼击航空兵师，第2、第34歼击轰炸航空兵师，第25轰炸航空兵团，"
    evidence = {"source_id": "s1"}
    entity = {"entity_id": "e1", "name": "第2歼击轰炸航空兵师",
              "type_ids": ["air_strike_force"], "type_assertions": [{"type_id": "air_strike_force", "evidence": [evidence]}],
              "aliases": [], "mentions": [evidence]}
    snapshot = ExtractionSchemaSnapshot.from_payload(default_config())
    sources = {"s1": source_record(document, "s1", document)}
    resolved = resolve_entities({"entities": [entity]}, sources, "batch_7", snapshot, document)
    assert resolved[0]["name"] == "第2歼击轰炸航空兵师"
    assert resolved[0]["mentions"][0]["text"] == document


def test_source_reference_schema_is_request_scoped():
    snapshot = ExtractionSchemaSnapshot.from_payload(default_config())
    first = snapshot.response_schema("entity_recognition", ["s1"])
    second = snapshot.response_schema("entity_recognition", ["s2"])
    first_evidence = first["properties"]["entities"]["items"]["properties"]["mentions"]["items"]
    second_evidence = second["properties"]["entities"]["items"]["properties"]["mentions"]["items"]
    assert first_evidence["properties"]["source_id"]["enum"] == ["s1"]
    assert second_evidence["properties"]["source_id"]["enum"] == ["s2"]
    document = "前出分队乙保持待命。"
    source = source_record(document, "s1", document)
    resolved = resolve_source_references([{"source_id": "s1"}], {"s1": source})
    assert resolved[0]["text"] == document
    with pytest.raises(ExtractionValidationError, match="source_id"):
        resolve_source_references([{"source_id": "s2"}], {"s1": source})


def test_source_unit_keeps_wrapped_continuation_and_exact_range():
    document = "第230、第733轰炸航空\n\n兵团和第48侦察航空兵团。"
    config = {"split_mode": "small_section", "batch_size": 1, "max_batch_tokens": 2500, "include_parent_title": True}
    batches = _build_relation_batches(_build_relation_sections(document), config)
    sources, _ = _typed_sources_and_batches(document, batches, [], 2500)
    assert sources[0]["text"] == document
    assert all(document[item["start"]:item["end"]] == item["text"] for item in sources)
    reference = {"source_id": sources[0]["source_id"]}
    entity = {"entity_id": "e1", "name": "第48侦察航空兵团", "type_ids": ["reconnaissance_platform"],
              "type_assertions": [{"type_id": "reconnaissance_platform", "evidence": [reference]}],
              "aliases": [], "mentions": [reference]}
    snapshot = ExtractionSchemaSnapshot.from_payload(default_config())
    resolved = resolve_entities({"entities": [entity]}, {source["source_id"]: source for source in sources}, "batch_7", snapshot, document)
    assert resolved[0]["mentions"][0]["text"] == document


def test_relation_reference_preserves_exact_pair_and_original_evidence():
    document = "联合指挥所甲指挥前出分队乙。"
    sources = {"s1": source_record(document, "s1", document)}
    payload = {"relations": [{"relation_id": "r1", "relation_type_id": "command", "head_id": "e1", "tail_id": "e2", "evidence": [{"source_id": "s1"}]}]}
    relation = resolve_relations(payload, sources, "batch_1")[0]
    assert relation["relation_type_id"] == "command"
    assert (relation["head_id"], relation["tail_id"]) == ("e1", "e2")
    assert relation["evidence"][0]["text"] == document
    payload["relations"][0]["evidence"][0]["source_id"] = "missing"
    with pytest.raises(ExtractionValidationError, match="source_id"):
        resolve_relations(payload, sources, "batch_1")


def test_identity_entity_enum_is_request_scoped():
    snapshot = ExtractionSchemaSnapshot.from_payload(default_config())
    first = snapshot.response_schema("entity_merge", [], entities=[{"entity_id": "e1"}])
    second = snapshot.response_schema("entity_merge", [], entities=[{"entity_id": "e2"}])
    assert first["properties"]["groups"]["items"]["properties"]["entity_ids"]["items"]["enum"] == ["e1"]
    assert second["properties"]["groups"]["items"]["properties"]["entity_ids"]["items"]["enum"] == ["e2"]


def test_same_name_entities_in_one_source_require_identity_confirmation():
    document = "东部指挥所的前出分队甲和西部指挥所的前出分队甲是不同单位。"
    snapshot = ExtractionSchemaSnapshot.from_payload(default_config())
    reference = {"source_id": "s1"}
    records = [{"entity_id": identifier, "name": "前出分队甲", "type_ids": ["forward_detachment"],
                "type_assertions": [{"type_id": "forward_detachment", "evidence": [reference]}],
                "aliases": [], "mentions": [reference]} for identifier in ("east", "west")]
    resolved = resolve_entities({"entities": records}, {"s1": source_record(document, "s1", document)}, "batch_1", snapshot, document)
    entities, mapping = merge_entities(resolved, [], snapshot, document)
    assert len(entities) == 2
    assert len(set(mapping.values())) == 2
    candidates = identity_candidates(entities)
    assert [(item["left_id"], item["right_id"]) for item in candidates] == [(entities[0]["entity_id"], entities[1]["entity_id"])]


def test_relation_context_distinguishes_homonyms_and_limits_relation_sources():
    lines = ["东部指挥所的前出分队甲负责甲区域。", "西部指挥所的前出分队甲负责乙区域。", "东部指挥所命令本方前出分队甲支援。", "地图比例尺1:500000。"]
    document = "\n".join(lines)
    sources = {f"s{index}": source_record(text, f"s{index}", document) for index, text in enumerate(lines, 1)}
    snapshot = ExtractionSchemaSnapshot.from_payload(default_config())
    entities = []
    for identifier, name, type_id, source_id in (
        ("east", "前出分队甲", "forward_detachment", "s1"),
        ("west", "前出分队甲", "forward_detachment", "s2"),
        ("command", "东部指挥所", "command_organization", "s1"),
    ):
        reference = {"source_id": source_id}
        record = {"entity_id": identifier, "name": name, "type_ids": [type_id],
                  "type_assertions": [{"type_id": type_id, "evidence": [reference]}], "aliases": [], "mentions": [reference]}
        entities.extend(resolve_entities({"entities": [record]}, sources, "batch_1", snapshot, document))
    extractor = ConfiguredExtractor(snapshot, None)
    content = next(relation_context_groups(entities, [sources["s3"]], sources, snapshot, compact_json, 10000))
    descriptors = {item["entity_id"]: item for item in content["entities"]}
    assert descriptors["batch_1_east"]["definitions"] == [{"source_id": "s1"}]
    assert descriptors["batch_1_west"]["definitions"] == [{"source_id": "s2"}]
    assert content["definitions"] == [{"source_id": key, "text": sources[key]["text"]} for key in ("s1", "s2")]
    assert content["sources"] == [{"source_id": "s3", "text": lines[2]}]
    schema = extractor.response_schema("extraction", content)
    payload = {"relations": [{"relation_id": "r1", "relation_type_id": "command", "head_id": "batch_1_command",
                              "tail_id": "batch_1_east", "evidence": [{"source_id": "s3"}]}]}
    validate_structure(payload, schema)
    payload["relations"][0]["evidence"] = [{"source_id": "s1"}]
    with pytest.raises(ExtractionValidationError):
        validate_structure(payload, schema)
    content = next(relation_context_groups(entities, list(sources.values()), sources, snapshot, compact_json, 10000))
    assert content["definitions"] == []
    unrelated = next(relation_context_groups(entities, [sources["s4"]], sources, snapshot, compact_json, 10000))
    assert unrelated["definitions"] == []
    assert all("definitions" not in item for item in unrelated["entities"])


def test_unambiguous_registry_does_not_copy_document_sources():
    document = "联合指挥所甲辖前出分队乙。\n地图比例尺1:500000。"
    snapshot = ExtractionSchemaSnapshot.from_payload(default_config())
    sources = {"s1": source_record(document.splitlines()[0], "s1", document),
               "s2": source_record(document.splitlines()[1], "s2", document)}
    records = [{"entity_id": identifier, "name": name, "type_ids": [type_id],
                "type_assertions": [{"type_id": type_id, "evidence": [{"source_id": "s1"}]}],
                "aliases": [], "mentions": [{"source_id": "s1"}]}
               for identifier, name, type_id in (("a", "联合指挥所甲", "command_organization"), ("b", "前出分队乙", "forward_detachment"))]
    entities = resolve_entities({"entities": records}, sources, "batch_1", snapshot, document)
    content = next(relation_context_groups(entities, [sources["s2"]], sources, snapshot, compact_json, 10000))
    assert len(content["entities"]) == 2
    assert all("definitions" not in entity for entity in content["entities"])
    assert content["definitions"] == []
    assert content["sources"] == [{"source_id": "s2", "text": sources["s2"]["text"]}]


def test_source_reference_lists_reject_duplicates_and_excess_items():
    snapshot = ExtractionSchemaSnapshot.from_payload(default_config())
    schema = snapshot.response_schema("extraction", ["s1", "s2"])
    evidence = schema["properties"]["relations"]["items"]["properties"]["evidence"]
    validate_structure([{"source_id": "s1"}, {"source_id": "s2"}], evidence)
    with pytest.raises(ExtractionValidationError, match="non-unique"):
        validate_structure([{"source_id": "s1"}, {"source_id": "s1"}], evidence)
    with pytest.raises(ExtractionValidationError):
        validate_structure([{"source_id": "s1"}] * 3, evidence)
    recognition = snapshot.response_schema("entity_recognition", ["s1", "s2"])
    mentions = recognition["properties"]["entities"]["items"]["properties"]["mentions"]
    assert mentions["maxItems"] == evidence["maxItems"] == 2
    assert mentions["uniqueItems"] is True
    transported = provider_schema(evidence)
    assert transported["maxItems"] == 2
    with pytest.raises(ExtractionValidationError):
        validate_structure([{"source_id": "s1"}] * 3, transported)


def test_shared_alias_context_survives_entity_partitioning():
    document = "指挥所甲辖东部前出分队甲。\n指挥所乙辖西部前出分队甲。\n指挥所甲命令本方前出分队甲支援。"
    snapshot = ExtractionSchemaSnapshot.from_payload(default_config())
    sources = {f"s{index}": source_record(text, f"s{index}", document) for index, text in enumerate(document.splitlines(), 1)}
    entities = []
    for identifier, name, type_id, source_id in (
        ("command", "指挥所甲", "command_organization", "s1"),
        ("east", "东部前出分队甲", "forward_detachment", "s1"),
        ("west", "西部前出分队甲", "forward_detachment", "s2"),
    ):
        reference = {"source_id": source_id}
        aliases = [] if identifier == "command" else [{"name": "前出分队甲", "evidence": [reference]}]
        record = {"entity_id": identifier, "name": name, "type_ids": [type_id], "aliases": aliases,
                  "type_assertions": [{"type_id": type_id, "evidence": [reference]}], "mentions": [reference]}
        entities.extend(resolve_entities({"entities": [record]}, sources, "batch_1", snapshot, document))
    content = next(relation_context_groups(entities, [sources["s3"]], sources, snapshot, compact_json, 10000,
                                          ["batch_1_command"], ["batch_1_east"]))
    descriptors = {item["entity_id"]: item for item in content["entities"]}
    assert "definitions" not in descriptors["batch_1_command"]
    assert descriptors["batch_1_east"]["definitions"] == [{"source_id": "s1"}]
    assert "batch_1_west" not in descriptors
    assert content["definitions"] == [{"source_id": "s1", "text": sources["s1"]["text"]}]
