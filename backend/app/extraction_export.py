import csv
import io

from skill4re.extraction_schema import public_result


RELATION_COLUMNS = ("关系ID", "关系", "头实体ID", "头实体", "头实体类型", "尾实体ID", "尾实体", "尾实体类型")
ENTITY_COLUMNS = ("实体ID", "实体名称", "实体类型", "别名")


def csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    dangerous = text.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n"))
    return "'" + text if dangerous else text


def write_csv(columns, rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: csv_cell(row.get(key)) for key in columns})
    return "\ufeff" + output.getvalue()


def typed_relations_to_csv(result):
    result = public_result(result)
    entities = {item["entity_id"]: item for item in result["entities"]}
    types = {item["id"]: item["name"] for item in result["extraction_schema"]["entity_types"]}
    relations = {item["id"]: item["name"] for item in result["extraction_schema"]["relation_types"]}
    rows = []
    for relation in result["relations"]:
        head, tail = entities[relation["head_id"]], entities[relation["tail_id"]]
        rows.append({"关系ID": relation["relation_id"], "关系": relations[relation["relation_type_id"]],
                     "头实体ID": head["entity_id"], "头实体": head["name"], "头实体类型": "、".join(types[value] for value in head["type_ids"]),
                     "尾实体ID": tail["entity_id"], "尾实体": tail["name"], "尾实体类型": "、".join(types[value] for value in tail["type_ids"])})
    return write_csv(RELATION_COLUMNS, rows)


def entities_to_csv(result):
    result = public_result(result)
    if "result_schema_version" not in result:
        names = {}
        for relation in result["final_relations"]:
            for name, type_name in ((relation.get("head") or relation.get("subject") or relation.get("head_entity"), relation.get("head_type") or relation.get("subject_type")),
                                    (relation.get("tail") or relation.get("object") or relation.get("tail_entity"), relation.get("tail_type") or relation.get("object_type"))):
                if name:
                    names.setdefault(name, {"实体名称": name, "实体类型": type_name or "未标注"})
        return write_csv(ENTITY_COLUMNS, names.values())
    types = {item["id"]: item["name"] for item in result["extraction_schema"]["entity_types"]}
    rows = []
    for entity in result["entities"]:
        rows.append({"实体ID": entity["entity_id"], "实体名称": entity["name"], "实体类型": "、".join(types[value] for value in entity["type_ids"]),
                     "别名": "、".join(item["name"] for item in entity["aliases"])})
    return write_csv(ENTITY_COLUMNS, rows)
