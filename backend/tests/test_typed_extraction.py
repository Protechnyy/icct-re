import pytest

from app.pipeline import _build_relation_batches, _build_relation_sections, _typed_sources_and_batches
from app.skill4re_client import provider_schema
from skill4re.dataset import approx_token_count
from skill4re.extraction_schema import ExtractionSchemaSnapshot, ExtractionValidationError, default_config, pair_allowed, validate_graph, validate_structure
from skill4re.typed_extraction import compact_json, identity_candidates, identity_request_groups, merge_entities, merge_relations, relation_context_groups, resolve_evidence, resolve_identity_groups


def entity_record(identifier, name, type_id, document):
    start = document.index(name)
    evidence = {"text": name, "source_id": identifier + "_source", "start": start, "end": start + len(name),
                "page": None, "section_id": None, "block_id": None, "batch_index": 0}
    return {"entity_id": identifier, "name": name, "type_ids": [type_id],
            "type_assertions": [{"type_id": type_id, "evidence": [evidence]}],
            "aliases": [], "mentions": [evidence]}


def test_source_ranges_preserve_unicode_and_text_without_images():
    document = "# 标题\n\n联合指挥所甲😀指挥前出分队乙。\n\n![图片](image.png)\n\n前出分队乙位于丙港。"
    split = {"split_mode": "small_section", "batch_size": 1, "max_batch_tokens": 2500, "include_parent_title": True}
    batches = _build_relation_batches(_build_relation_sections(document), split)
    sources, batches = _typed_sources_and_batches(document, batches, [], 2500)
    assert all(document[item["start"]:item["end"]] == item["text"] for item in sources)
    assert all(item["page"] is None and item["block_id"] is None for item in sources)
    assert all("image.png" not in item["text"] and "![" not in item["text"] for item in sources)
    assert any("😀" in item["text"] for item in sources)
    assert {item["source_id"] for item in sources} == {identifier for batch in batches for identifier in batch["source_ids"]}


def test_candidate_partition_preserves_every_legal_pair():
    snapshot = ExtractionSchemaSnapshot.from_payload(default_config())
    document = "甲机构、乙机构、丙平台、丁平台、戊平台、己分队"
    entities = [entity_record("e1", "甲机构", "command_organization", document),
                entity_record("e2", "乙机构", "command_organization", document),
                entity_record("e3", "丙平台", "surface_strike_force", document),
                entity_record("e4", "丁平台", "air_strike_force", document),
                entity_record("e5", "戊平台", "reconnaissance_platform", document),
                entity_record("e6", "己分队", "forward_detachment", document)]
    sources = {item["source_id"]: item for entity in entities for item in entity["mentions"]}
    identifiers = [item["entity_id"] for item in entities]
    minimum = [group for head in identifiers for tail in identifiers if head != tail
               for group in relation_context_groups(entities, [], sources, snapshot, compact_json, 10000, [head], [tail])]
    limit = max(approx_token_count(compact_json(group)) for group in minimum)
    groups = list(relation_context_groups(entities, [], sources, snapshot, compact_json, limit))
    assert len(groups) > 1
    assert all(approx_token_count(compact_json(group)) <= limit for group in groups)
    config = snapshot.payload()
    types = {item["id"]: item for item in config["entity_types"]}
    expected = {(relation["id"], head["entity_id"], tail["entity_id"]) for relation in config["relation_types"]
                for head in entities for tail in entities if head != tail and pair_allowed(relation, head["type_ids"], tail["type_ids"], types)}
    actual = {(branch["properties"]["relation_type_id"]["const"], head, tail) for group in groups
              for branch in snapshot.relation_branches(group["entities"], group["head_ids"], group["tail_ids"])
              for head in branch["properties"]["head_id"]["enum"] for tail in branch["properties"]["tail_id"]["enum"] if head != tail}
    assert actual == expected
    assert all("candidates" not in group for group in groups)


def test_full_local_schema_enforces_uniqueness_after_transport_projection():
    snapshot = ExtractionSchemaSnapshot.from_payload(default_config())
    schema = snapshot.response_schema("entity_recognition")
    projected = provider_schema(schema)
    original_types = schema["properties"]["entities"]["items"]["properties"]["type_ids"]
    projected_types = projected["properties"]["entities"]["items"]["properties"]["type_ids"]
    assert original_types["uniqueItems"] is True
    assert "uniqueItems" not in projected_types
    with pytest.raises(ExtractionValidationError):
        validate_structure(["command_organization", "command_organization"], original_types)


def test_identity_requests_batch_related_records_and_keep_every_candidate():
    document = "法国第1集团军与法国第1集团军。德国第2集团军与德国第2集团军。"
    entities = [entity_record(f"e{index}", name, "command_organization", document)
                for index, name in enumerate(("法国第1集团军", "法国第1集团军", "德国第2集团军", "德国第2集团军"))]
    sources = {item["source_id"]: item for entity in entities for item in entity["mentions"]}
    candidates = identity_candidates(entities)
    groups = list(identity_request_groups(candidates, entities, sources, compact_json, 10000))
    assert len(candidates) == 2 and len(groups) == 1
    assert groups[0]["candidates"] == candidates
    assert {item["entity_id"] for item in groups[0]["entities"]} == {item["entity_id"] for item in entities}


def test_identity_groups_reject_overlaps_and_unrelated_records():
    sources = {"s1": {"source_id": "s1", "text": "甲机构又称乙机构。", "start": 0}}
    candidates = [{"left_id": "a", "right_id": "b"}, {"left_id": "c", "right_id": "d"}]
    valid = {"groups": [{"entity_ids": ["a", "b"], "evidence": [{"source_id": "s1"}]}]}
    assert resolve_identity_groups(valid, candidates, sources, "merge_1")[0]["decision"] == "same"
    overlap = {"groups": valid["groups"] + valid["groups"]}
    with pytest.raises(ExtractionValidationError, match="重复"):
        resolve_identity_groups(overlap, candidates, sources, "merge_1")
    wrong = {"groups": [{"entity_ids": ["a", "c"], "evidence": [{"source_id": "s1"}]}]}
    with pytest.raises(ExtractionValidationError, match="同一候选"):
        resolve_identity_groups(wrong, candidates, sources, "merge_1")


def test_relation_transport_schema_rejects_cross_type_combination():
    snapshot = ExtractionSchemaSnapshot.from_payload(default_config())
    entities = [{"entity_id": "a", "type_ids": ["command_organization"]},
                {"entity_id": "b", "type_ids": ["surface_strike_force"]},
                {"entity_id": "c", "type_ids": ["air_strike_force"]}]
    schema = snapshot.response_schema("extraction", ["s1"], entities=entities)
    payload = {"relations": [{"relation_id": "r1", "relation_type_id": "coordination", "head_id": "b", "tail_id": "c", "evidence": [{"source_id": "s1"}]}]}
    validate_structure(payload, schema)
    payload["relations"][0]["head_id"] = "a"
    with pytest.raises(ExtractionValidationError):
        validate_structure(payload, schema)
    payload["relations"][0]["relation_type_id"] = "command"
    validate_structure(payload, schema)


def test_request_groups_preserve_candidates_sources_and_adjacent_context():
    snapshot = ExtractionSchemaSnapshot.from_payload(default_config())
    document = "甲机构、乙分队"
    entities = [entity_record("e1", "甲机构", "command_organization", document), entity_record("e2", "乙分队", "forward_detachment", document)]
    sources = [{"source_id": f"s{index}", "text": "来源文本" * 100} for index in range(8)]
    source_map = {item["source_id"]: item for entity in entities for item in entity["mentions"]}
    minimum_groups = list(relation_context_groups(entities, sources[:2], source_map, snapshot, compact_json, 10000))
    limit = max(approx_token_count(compact_json(group)) for group in minimum_groups)
    groups = list(relation_context_groups(entities, sources, source_map, snapshot, compact_json, limit))
    assert len(groups) > 1
    assert all(approx_token_count(compact_json(group)) <= limit for group in groups)
    coverage = {(head, tail, source["source_id"]) for group in groups for head in group["head_ids"] for tail in group["tail_ids"] if head != tail for source in group["sources"]}
    assert coverage == {("e1", "e2", source["source_id"]) for source in sources}
    assert all(any(left in group["sources"] and right in group["sources"] for group in groups) for left, right in zip(sources, sources[1:]))


def test_type_combinations_and_invalid_references_fail():
    snapshot = ExtractionSchemaSnapshot.from_payload(default_config())
    document = "甲机构与乙平台。"
    head = entity_record("e1", "甲机构", "command_organization", document)
    tail = entity_record("e2", "乙平台", "weapon_platform", document)
    relation = {"relation_id": "r1", "relation_type_id": "coordination", "head_id": "e1", "tail_id": "e2", "evidence": head["mentions"]}
    with pytest.raises(ExtractionValidationError, match="类型不符合"):
        validate_graph([head, tail], [relation], snapshot, document)
    relation["head_id"] = "unknown"
    with pytest.raises(ExtractionValidationError, match="不存在"):
        validate_graph([head, tail], [relation], snapshot, document)


def test_evidence_and_symmetric_merge_keep_sources():
    document = "甲平台协同乙平台。"
    source = {"source_id": "s1", "text": document, "start": 0}
    evidence = resolve_evidence([{"source_id": "s1", "text": document}], {"s1": source})
    with pytest.raises(ExtractionValidationError, match="没有原样"):
        resolve_evidence([{"source_id": "s1", "text": "未提供的原文"}], {"s1": source})
    snapshot = ExtractionSchemaSnapshot.from_payload(default_config())
    relations = [{"relation_id": "a", "relation_type_id": "coordination", "head_id": "e1", "tail_id": "e2", "evidence": evidence},
                 {"relation_id": "b", "relation_type_id": "coordination", "head_id": "e2", "tail_id": "e1", "evidence": evidence}]
    assert len(merge_relations(relations, snapshot)) == 1
    entities = [entity_record("e1", "甲平台", "weapon_platform", document), entity_record("e2", "乙平台", "weapon_platform", document)]
    merged, mapping = merge_entities(entities, [], snapshot, document)
    assert len(merged) == 2
    assert mapping["e1"] != mapping["e2"]
