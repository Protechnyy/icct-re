import copy
import json
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator


class ExtractionValidationError(ValueError):
    def __init__(self, path, message):
        self.path = str(path)
        self.message = str(message)
        super().__init__(f"{self.path}: {self.message}")


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def array_schema(items, minimum=0):
    return {"type": "array", "items": items, "minItems": minimum}


TEXT = {"type": "string", "minLength": 1, "pattern": r"\S"}
IDENTIFIER = {"type": "string", "pattern": r"^[A-Za-z][A-Za-z0-9_]*$"}
NULLABLE_TEXT = {"type": ["string", "null"], "minLength": 1}
RELATION_NAMES = {"指挥关系": "指挥", "协同关系": "协同", "情报保障关系": "情报保障"}
MODEL_EVIDENCE_SCHEMA = object_schema({"source_id": TEXT})
EVIDENCE_SCHEMA = object_schema({
    "text": TEXT, "source_id": TEXT,
    "start": {"type": "integer", "minimum": 0}, "end": {"type": "integer", "minimum": 1},
    "page": {"type": ["integer", "null"]}, "section_id": NULLABLE_TEXT,
    "block_id": {"type": ["integer", "string", "null"]},
    "batch_index": {"type": ["integer", "null"]},
})
TYPE_EXAMPLE_SCHEMA = object_schema({"text": TEXT, "entity_name": TEXT, "type_id": IDENTIFIER, "evidence": TEXT})
RELATION_EXAMPLE_SCHEMA = object_schema({
    "text": TEXT, "head": TEXT, "tail": TEXT,
    "head_type_id": IDENTIFIER, "tail_type_id": IDENTIFIER, "evidence": TEXT,
})
CONFIG_SCHEMA = object_schema({
    "schema_version": {"const": 1}, "revision": {"type": "integer", "minimum": 1},
    "entity_types": array_schema(object_schema({
        "id": IDENTIFIER, "name": TEXT, "description": TEXT, "parent_id": NULLABLE_TEXT,
        "examples": array_schema(TYPE_EXAMPLE_SCHEMA),
    }), 1),
    "relation_types": array_schema(object_schema({
        "id": IDENTIFIER, "name": TEXT, "description": TEXT, "enabled": {"type": "boolean"},
        "symmetric": {"type": "boolean"},
        "allowed_pairs": array_schema(object_schema({"head_type_id": IDENTIFIER, "tail_type_id": IDENTIFIER}), 1),
        "examples": array_schema(RELATION_EXAMPLE_SCHEMA),
    })),
})


def entity_schema(evidence):
    schema = object_schema({
        "entity_id": TEXT, "name": TEXT, "type_ids": {**array_schema(IDENTIFIER, 1), "uniqueItems": True},
        "type_assertions": array_schema(object_schema({"type_id": IDENTIFIER, "evidence": array_schema(evidence, 1)}), 1),
        "aliases": array_schema(object_schema({"name": TEXT, "evidence": array_schema(evidence, 1)})),
        "mentions": array_schema(evidence, 1),
    })
    return schema


def relation_schema(evidence):
    return object_schema({
        "relation_id": TEXT, "relation_type_id": IDENTIFIER,
        "head_id": TEXT, "tail_id": TEXT, "evidence": array_schema(evidence, 1),
    })


ENTITY_SCHEMA = entity_schema(EVIDENCE_SCHEMA)
RELATION_SCHEMA = relation_schema(EVIDENCE_SCHEMA)
MERGE_SCHEMA = object_schema({"groups": array_schema(object_schema({
    "entity_ids": {**array_schema(TEXT, 2), "uniqueItems": True},
    "evidence": array_schema(MODEL_EVIDENCE_SCHEMA, 1),
}))})
PUBLIC_ENTITY_SCHEMA = object_schema({
    "entity_id": TEXT, "name": TEXT, "type_ids": {**array_schema(IDENTIFIER, 1), "uniqueItems": True},
    "aliases": array_schema(object_schema({"name": TEXT})),
})
PUBLIC_RELATION_SCHEMA = object_schema({
    "relation_id": TEXT, "relation_type_id": IDENTIFIER, "head_id": TEXT, "tail_id": TEXT,
})
PUBLIC_RESULT_SCHEMA = {
    "type": "object", "required": ["result_schema_version", "document_text", "extraction_schema", "entities", "relations"],
    "properties": {"result_schema_version": {"const": 4}, "document_text": {"type": "string"},
                   "extraction_schema": CONFIG_SCHEMA, "entities": array_schema(PUBLIC_ENTITY_SCHEMA),
                   "relations": array_schema(PUBLIC_RELATION_SCHEMA)},
}


def validate_structure(value, schema, path="$"):
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda error: str(list(error.path)))
    if errors:
        error = errors[0]
        suffix = "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.path)
        raise ExtractionValidationError(path + suffix, error.message)


def unique_index(items, field, path):
    result = {}
    for index, item in enumerate(items):
        identifier = item[field]
        if identifier in result:
            raise ExtractionValidationError(f"{path}[{index}].{field}", f"重复 ID：{identifier}")
        result[identifier] = item
    return result


def ancestors(type_id, types):
    result = set()
    current = type_id
    while current is not None:
        if current not in types:
            raise ExtractionValidationError("entity_types.parent_id", f"类型不存在：{current}")
        if current in result:
            raise ExtractionValidationError("entity_types.parent_id", f"父类型循环：{current}")
        result.add(current)
        current = types[current]["parent_id"]
    return result


def pair_allowed(relation, head_types, tail_types, types):
    head = set().union(*(ancestors(value, types) for value in head_types))
    tail = set().union(*(ancestors(value, types) for value in tail_types))
    for pair in relation["allowed_pairs"]:
        if pair["head_type_id"] in head and pair["tail_type_id"] in tail:
            return True
        if relation["symmetric"] and pair["tail_type_id"] in head and pair["head_type_id"] in tail:
            return True
    return False


def validate_config(payload):
    validate_structure(payload, CONFIG_SCHEMA)
    types = unique_index(payload["entity_types"], "id", "entity_types")
    unique_index(payload["relation_types"], "id", "relation_types")
    for index, item in enumerate(payload["entity_types"]):
        try:
            ancestors(item["id"], types)
        except ExtractionValidationError as error:
            raise ExtractionValidationError(f"entity_types[{index}].parent_id", error.message) from error
        for example in item["examples"]:
            if example["type_id"] not in types or example["evidence"] not in example["text"] or example["entity_name"] not in example["text"]:
                raise ExtractionValidationError(f"entity_types[{index}].examples", "示例类型或原文依据无效")
    for index, relation in enumerate(payload["relation_types"]):
        for pair in relation["allowed_pairs"]:
            if pair["head_type_id"] not in types or pair["tail_type_id"] not in types:
                raise ExtractionValidationError(f"relation_types[{index}].allowed_pairs", "引用了不存在的实体类型")
        for example in relation["examples"]:
            if example["head_type_id"] not in types or example["tail_type_id"] not in types:
                raise ExtractionValidationError(f"relation_types[{index}].examples", "示例引用了不存在的实体类型")
            if not pair_allowed(relation, [example["head_type_id"]], [example["tail_type_id"]], types):
                raise ExtractionValidationError(f"relation_types[{index}].examples", "示例不符合允许的类型组合")
            if any(example[field] not in example["text"] for field in ("head", "tail", "evidence")):
                raise ExtractionValidationError(f"relation_types[{index}].examples", "示例的实体或证据不在原文中")
    return payload


def default_config():
    payload = json.loads(Path(__file__).with_name("default_extraction_schema.json").read_text(encoding="utf-8"))
    return validate_config(payload)


@dataclass(frozen=True)
class ExtractionSchemaSnapshot:
    serialized: str

    @classmethod
    def from_payload(cls, payload):
        validate_config(payload)
        return cls(json.dumps(payload, ensure_ascii=False, allow_nan=False))

    def payload(self):
        return json.loads(self.serialized)

    @property
    def revision(self):
        return self.payload()["revision"]

    def active_type_ids(self):
        payload = self.payload()
        types = {item["id"]: item for item in payload["entity_types"]}
        referenced = {pair[field] for relation in payload["relation_types"] if relation["enabled"]
                      for pair in relation["allowed_pairs"] for field in ("head_type_id", "tail_type_id")}
        return {identifier for identifier in types if ancestors(identifier, types) & referenced}

    def require_active(self):
        if not any(item["enabled"] for item in self.payload()["relation_types"]):
            raise ExtractionValidationError("relation_types", "没有启用的关系类型")

    def response_schema(self, stage, source_ids=None, candidate_ids=None, entities=None, head_ids=None, tail_ids=None):
        if stage == "entity_merge":
            schema = copy.deepcopy(MERGE_SCHEMA)
            if entities is not None:
                schema["properties"]["groups"]["items"]["properties"]["entity_ids"]["items"] = {"type": "string", "enum": sorted(item["entity_id"] for item in entities)}
        elif stage in {"entity_recognition", "entity_proofreading"}:
            schema = entity_schema(MODEL_EVIDENCE_SCHEMA)
            identifiers = sorted(self.active_type_ids())
            schema["properties"]["type_ids"]["items"] = {"type": "string", "enum": identifiers}
            schema["properties"]["type_assertions"]["items"]["properties"]["type_id"] = {"type": "string", "enum": identifiers}
            schema = object_schema({"entities": array_schema(schema)})
        else:
            items = relation_schema(MODEL_EVIDENCE_SCHEMA)
            branches = self.relation_branches(entities, head_ids, tail_ids) if entities is not None else None
            if branches:
                items = {"anyOf": branches}
            relations = array_schema(items)
            if branches == []:
                relations["maxItems"] = 0
            schema = object_schema({"relations": relations})
        schema = copy.deepcopy(schema)
        if source_ids is not None:
            restrict_source_ids(schema, sorted(set(source_ids)))
        return schema

    def relation_branches(self, entities, head_ids=None, tail_ids=None):
        config = self.payload()
        types = {item["id"]: item for item in config["entity_types"]}
        heads = set(head_ids) if head_ids is not None else {item["entity_id"] for item in entities}
        tails = set(tail_ids) if tail_ids is not None else {item["entity_id"] for item in entities}
        expanded = {item["entity_id"]: set().union(*(ancestors(value, types) for value in item["type_ids"])) for item in entities}
        branches = []
        for relation in config["relation_types"]:
            if not relation["enabled"]:
                continue
            for pair in relation["allowed_pairs"]:
                orientations = [(pair["head_type_id"], pair["tail_type_id"])]
                if relation["symmetric"] and pair["head_type_id"] != pair["tail_type_id"]:
                    orientations.append((pair["tail_type_id"], pair["head_type_id"]))
                for head_type, tail_type in orientations:
                    allowed_heads = sorted(value for value in heads if head_type in expanded[value])
                    allowed_tails = sorted(value for value in tails if tail_type in expanded[value])
                    if not any(head != tail for head in allowed_heads for tail in allowed_tails):
                        continue
                    branch = relation_schema(MODEL_EVIDENCE_SCHEMA)
                    branch["properties"]["relation_type_id"] = {"const": relation["id"], "type": "string"}
                    branch["properties"]["head_id"] = {"type": "string", "enum": allowed_heads}
                    branch["properties"]["tail_id"] = {"type": "string", "enum": allowed_tails}
                    branches.append(branch)
        return branches


def restrict_source_ids(schema, identifiers):
    if isinstance(schema, list):
        for item in schema:
            restrict_source_ids(item, identifiers)
    elif isinstance(schema, dict):
        if schema.get("type") == "array" and set(schema.get("items", {}).get("properties", {})) == {"source_id"}:
            schema["maxItems"] = len(identifiers)
            schema["uniqueItems"] = True
        if set(schema.get("properties", {})) == {"source_id"}:
            schema["properties"]["source_id"] = {"type": "string", "enum": identifiers}
        for item in schema.values():
            restrict_source_ids(item, identifiers)


def validate_evidence(items, document_text, path):
    for index, item in enumerate(items):
        validate_structure(item, EVIDENCE_SCHEMA, f"{path}[{index}]")
        if item["end"] > len(document_text) or item["end"] <= item["start"] or document_text[item["start"]:item["end"]] != item["text"]:
            raise ExtractionValidationError(f"{path}[{index}]", "原文范围与证据不一致")


def validate_graph(entities, relations, snapshot, document_text):
    validate_structure(entities, array_schema(ENTITY_SCHEMA), "entities")
    validate_structure(relations, array_schema(RELATION_SCHEMA), "relations")
    payload = snapshot.payload()
    types = {item["id"]: item for item in payload["entity_types"]}
    relation_types = {item["id"]: item for item in payload["relation_types"] if item["enabled"]}
    entity_map = unique_index(entities, "entity_id", "entities")
    unique_index(relations, "relation_id", "relations")
    active = snapshot.active_type_ids()
    for index, entity in enumerate(entities):
        path = f"entities[{index}]"
        validate_structure(entity, ENTITY_SCHEMA, path)
        if not set(entity["type_ids"]) <= active:
            raise ExtractionValidationError(path + ".type_ids", "类型不在当前抽取范围")
        if set(entity["type_ids"]) != {item["type_id"] for item in entity["type_assertions"]}:
            raise ExtractionValidationError(path + ".type_assertions", "每个实体类型必须具有原文依据")
        validate_evidence(entity["mentions"], document_text, path + ".mentions")
        for item in entity["type_assertions"] + entity["aliases"]:
            validate_evidence(item["evidence"], document_text, path + ".evidence")
    for index, relation in enumerate(relations):
        path = f"relations[{index}]"
        validate_structure(relation, RELATION_SCHEMA, path)
        if relation["relation_type_id"] not in relation_types:
            raise ExtractionValidationError(path + ".relation_type_id", "关系不存在或尚未启用")
        if relation["head_id"] not in entity_map or relation["tail_id"] not in entity_map:
            raise ExtractionValidationError(path, "关系引用了不存在的实体")
        head, tail = entity_map[relation["head_id"]], entity_map[relation["tail_id"]]
        if not pair_allowed(relation_types[relation["relation_type_id"]], head["type_ids"], tail["type_ids"], types):
            raise ExtractionValidationError(path, "头尾实体类型不符合允许的组合")
        validate_evidence(relation["evidence"], document_text, path + ".evidence")


def validate_result(result):
    if result.get("result_schema_version") in (2, 3):
        public_result(result)
        return
    validate_public_result(result)


def validate_public_result(result):
    validate_structure(result, PUBLIC_RESULT_SCHEMA)
    snapshot = ExtractionSchemaSnapshot.from_payload(result["extraction_schema"])
    config = snapshot.payload()
    types = {item["id"]: item for item in config["entity_types"]}
    relations = {item["id"]: item for item in config["relation_types"] if item["enabled"]}
    entities = unique_index(result["entities"], "entity_id", "entities")
    unique_index(result["relations"], "relation_id", "relations")
    active = snapshot.active_type_ids()
    for index, entity in enumerate(result["entities"]):
        path = f"entities[{index}]"
        if not set(entity["type_ids"]) <= active:
            raise ExtractionValidationError(path + ".type_ids", "类型不在当前抽取范围")
    for index, relation in enumerate(result["relations"]):
        path = f"relations[{index}]"
        if relation["relation_type_id"] not in relations:
            raise ExtractionValidationError(path + ".relation_type_id", "关系不存在或尚未启用")
        if relation["head_id"] not in entities or relation["tail_id"] not in entities:
            raise ExtractionValidationError(path, "关系引用了不存在的实体")
        if relation["head_id"] == relation["tail_id"]:
            raise ExtractionValidationError(path, "关系两端必须是不同实体")
        if not pair_allowed(relations[relation["relation_type_id"]], entities[relation["head_id"]]["type_ids"], entities[relation["tail_id"]]["type_ids"], types):
            raise ExtractionValidationError(path, "头尾实体类型不符合允许的组合")


def public_graph(entities, relations):
    output = []
    for entity in entities:
        item = {field: copy.deepcopy(entity[field]) for field in ("entity_id", "name", "type_ids")}
        item["aliases"] = [{"name": alias["name"]} for alias in entity["aliases"]]
        output.append(item)
    return output, [{field: relation[field] for field in PUBLIC_RELATION_SCHEMA["properties"]} for relation in relations]


def result_metadata(result):
    output = {field: copy.deepcopy(result[field]) for field in (
        "document_text", "extraction_schema", "document_meta", "ocr_summary", "saved_paths",
    ) if field in result}
    if "extraction_schema" in output:
        for relation in output["extraction_schema"]["relation_types"]:
            relation["name"] = RELATION_NAMES.get(relation["name"], relation["name"])
    if "stage_outputs" in result:
        output["stage_outputs"] = {field: copy.deepcopy(result["stage_outputs"][field])
                                   for field in ("timing", "requests") if field in result["stage_outputs"]}
    return output


def public_result(result):
    if "result_schema_version" not in result:
        container = (result.get("final_relation_list") or result.get("final_relations")
                     or result.get("skill4re_result", {}).get("relation_list") or [])
        relations = container.get("relation_list", []) if isinstance(container, dict) else container
        if not isinstance(relations, list):
            raise ExtractionValidationError("relations", "历史关系结果必须是列表")
        output = result_metadata(result)
        output["final_relations"] = [{field: item[field] for field in (
            "head", "subject", "head_entity", "tail", "object", "tail_entity", "relation",
            "head_type", "subject_type", "tail_type", "object_type",
        ) if field in item} for item in relations]
        for relation in output["final_relations"]:
            if "relation" in relation:
                relation["relation"] = RELATION_NAMES.get(relation["relation"], relation["relation"])
        return output
    if result["result_schema_version"] not in (2, 3, 4):
        raise ExtractionValidationError("result_schema_version", "不支持的结果格式版本")
    if result["result_schema_version"] == 4:
        validate_public_result(result)
    else:
        historical_schema = copy.deepcopy(PUBLIC_RESULT_SCHEMA)
        properties = historical_schema["properties"]
        properties["result_schema_version"] = {"const": result["result_schema_version"]}
        properties["entities"]["items"] = copy.deepcopy(ENTITY_SCHEMA if result["result_schema_version"] == 2 else PUBLIC_ENTITY_SCHEMA)
        properties["entities"]["items"]["additionalProperties"] = True
        if result["result_schema_version"] == 2:
            properties["relations"] = array_schema(RELATION_SCHEMA)
        validate_structure(result, historical_schema)
    output = result_metadata(result)
    output["result_schema_version"] = 4
    output["entities"], output["relations"] = public_graph(result["entities"], result["relations"])
    validate_public_result(output)
    return output
