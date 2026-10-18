import copy

import pytest

from skill4re.extraction_schema import (
    ExtractionSchemaSnapshot, ExtractionValidationError,
    ancestors, default_config, pair_allowed, validate_config,
)


def test_default_types_and_pairs():
    config = default_config()
    snapshot = ExtractionSchemaSnapshot.from_payload(config)
    types = {item["id"]: item for item in config["entity_types"]}
    relations = {item["id"]: item for item in config["relation_types"]}
    assert len(types) == 8
    assert len(snapshot.active_type_ids()) == 8
    assert [len(item["allowed_pairs"]) for item in relations.values()] == [3, 2, 1]
    assert pair_allowed(relations["coordination"], ["surface_strike_force"], ["air_strike_force"], types)
    assert not pair_allowed(relations["coordination"], ["command_organization"], ["weapon_platform"], types)
    assert pair_allowed(relations["command"], ["command_organization"], ["long_range_missile_strike_force"], types)
    assert not pair_allowed(relations["command"], ["weapon_platform"], ["command_organization"], types)


def test_config_errors_have_field_paths():
    config = default_config()
    config["entity_types"][0]["parent_id"] = "command_organization"
    with pytest.raises(ExtractionValidationError, match="父类型循环"):
        validate_config(config)
    config = default_config()
    config["relation_types"][0]["allowed_pairs"][0]["tail_type_id"] = "missing"
    with pytest.raises(ExtractionValidationError, match="allowed_pairs"):
        validate_config(config)
    config = default_config()
    config["entity_types"].append(copy.deepcopy(config["entity_types"][0]))
    with pytest.raises(ExtractionValidationError, match="重复 ID"):
        validate_config(config)
    config = default_config()
    config["entity_types"][0]["description"] = " "
    with pytest.raises(ExtractionValidationError, match="description"):
        validate_config(config)
    config = default_config()
    config["relation_types"][0]["examples"][0]["head_type_id"] = "weapon_platform"
    with pytest.raises(ExtractionValidationError, match="examples"):
        validate_config(config)


def test_snapshot_is_independent_and_inherits_new_children():
    config = default_config()
    original = ExtractionSchemaSnapshot.from_payload(config)
    config["entity_types"].append({"id": "new_platform", "name": "新增平台", "description": "新增平台类型", "parent_id": "weapon_platform", "examples": []})
    updated = ExtractionSchemaSnapshot.from_payload(config)
    assert "new_platform" in updated.active_type_ids()
    assert "new_platform" not in original.active_type_ids()
    assert "weapon_platform" in ancestors("new_platform", {item["id"]: item for item in config["entity_types"]})
    payload = original.payload()
    payload["entity_types"].clear()
    assert len(original.payload()["entity_types"]) == 8


def test_disabled_relations_are_valid_but_cannot_start():
    config = default_config()
    for relation in config["relation_types"]:
        relation["enabled"] = False
    snapshot = ExtractionSchemaSnapshot.from_payload(config)
    assert snapshot.active_type_ids() == set()
    with pytest.raises(ExtractionValidationError, match="没有启用"):
        snapshot.require_active()
