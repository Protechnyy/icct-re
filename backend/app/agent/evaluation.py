"""Offline replay, automatic metrics and human change judgments."""
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from dataclasses import replace
import json
from pathlib import Path

from . import verify_document
from .document import compact
from .relations import assign_relation_ids

CSV_FIELDS = ("document", "change_id", "type", "task_id", "relation_ids", "before", "after", "evidence", "pages", "judgment")


def relation_metrics(relations, document_text):
    source = compact(document_text)
    located = sum(bool(compact(r.get("evidence"))) and compact(r["evidence"]) in source for r in relations)
    entities = {compact(r[field]) for r in relations for field in ("head", "tail") if r.get(field)}
    return {"relation_count": len(relations), "locatable_evidence_count": located,
            "locatable_evidence_ratio": located / len(relations) if relations else 0.0,
            "entity_count": len(entities)}


def annotation_metrics(relations, annotations):
    def triples(items):
        return {tuple(compact(item.get(field)).casefold() for field in ("head", "relation", "tail")) for item in items}
    predicted, gold = triples(relations), triples(annotations)
    matched = len(predicted & gold)
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(gold) if gold else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "matched": matched, "predicted": len(predicted), "gold": len(gold)}


def comparison(before, after, document_text, agent_result, annotations=None):
    summary = agent_result.get("summary", {})
    change_counts = Counter(change["type"] for change in agent_result.get("changes", []))
    return {"status": agent_result["status"], "before": relation_metrics(before, document_text),
            "after": relation_metrics(after, document_text),
            "changes": {kind: change_counts[kind] for kind in ("correction", "deletion", "addition", "entity_merge")},
            "llm_calls": summary.get("llm_calls", 0),
            "tokens": summary.get("tokens", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
            "elapsed_seconds": summary.get("elapsed_seconds", 0),
            "annotations": {"before": annotation_metrics(before, annotations), "after": annotation_metrics(after, annotations)}
                if annotations is not None else {"skipped": True, "reason": "未提供标注关系"}}


def aggregate_metrics(documents):
    result = {"document_count": len(documents), "before": {}, "after": {}, "changes": {},
              "llm_calls": sum(d["llm_calls"] for d in documents),
              "tokens": {key: sum(d["tokens"].get(key, 0) for d in documents)
                         for key in ("input_tokens", "output_tokens", "total_tokens")},
              "elapsed_seconds": sum(d["elapsed_seconds"] for d in documents),
              "status_counts": dict(Counter(d["status"] for d in documents))}
    for stage in ("before", "after"):
        count = sum(d[stage]["relation_count"] for d in documents)
        located = sum(d[stage]["locatable_evidence_count"] for d in documents)
        result[stage] = {"relation_count": count, "locatable_evidence_count": located,
                         "locatable_evidence_ratio": located / count if count else 0,
                         "entity_count": sum(d[stage]["entity_count"] for d in documents)}
    result["changes"] = {kind: sum(d["changes"][kind] for d in documents)
                         for kind in ("correction", "deletion", "addition", "entity_merge")}
    result["entity_count_note"] = "汇总为逐文档实体数之和，不跨文档合并实体"
    return result


def export_changes(path, changes, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def safe(value):
        value = str(value)
        return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, change in enumerate(changes, 1):
            row = {"document": document, "change_id": f"c{index}", "type": change["type"],
                   "task_id": change["task_id"], "relation_ids": json.dumps(change.get("relation_ids", []), ensure_ascii=False),
                   "before": json.dumps(change.get("before", []), ensure_ascii=False),
                   "after": json.dumps(change.get("after", []), ensure_ascii=False),
                   "evidence": change.get("evidence", ""), "pages": json.dumps(change.get("pages", [])), "judgment": ""}
            writer.writerow({field: safe(value) for field, value in row.items()})


def score_judgments(paths):
    groups = defaultdict(lambda: Counter(total=0, reviewed=0, correct=0))
    true_values = {"1", "true", "yes", "correct", "正确", "是", "通过"}
    false_values = {"0", "false", "no", "incorrect", "错误", "否", "不正确"}
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                label = row.get("judgment", "").strip().lower()
                if label and label not in true_values | false_values:
                    raise ValueError(f"未知人工判定 {label!r}，位于 {path} / {row.get('change_id')}")
                for kind in ("overall", row["type"]):
                    groups[kind]["total"] += 1
                    if label:
                        groups[kind]["reviewed"] += 1
                        groups[kind]["correct"] += int(label in true_values)
    return {kind: {**values, "unreviewed": values["total"] - values["reviewed"],
                   "accuracy": values["correct"] / values["reviewed"] if values["reviewed"] else None}
            for kind, values in groups.items()}


def replay_file(input_path, output_root, config, *, annotations=None, model=None, progress_callback=None):
    source = Path(input_path).resolve()
    output_root = Path(output_root).resolve()
    if output_root == source.parent or source.parent in output_root.parents:
        raise ValueError("输出目录必须与输入结果目录独立")
    result = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(result.get("document_text"), str) or not result["document_text"].strip():
        raise ValueError("输入缺少 document_text 原文")
    before = result.get("pre_agent_relations", result.get("final_relations"))
    if not isinstance(before, list):
        raise ValueError("输入缺少 pre_agent_relations / final_relations 关系列表")
    before = assign_relation_ids(before)
    sections = result.get("relation_sections") or [{"section_id": "document", "title": "原文", "text": result["document_text"],
        "paragraphs": [{"content": paragraph, "page": None} for paragraph in result["document_text"].split("\n\n") if paragraph.strip()]}]
    after, agent_result = verify_document(before, sections, result["document_text"], replace(config, agent_enabled=True),
                                          progress_callback, model=model)
    replayed = deepcopy(result)
    replayed.update({"pre_agent_relations": before, "final_relations": after,
                     "final_relation_list": {"relation_list": after}, "agent_result": agent_result})
    if "ocr_summary" in replayed:
        replayed["ocr_summary"]["relation_count"] = len(after)
    identity = str(result.get("document_meta", {}).get("task_id") or source.parent.name)
    # Restrict user-supplied identifiers to one safe directory component.
    if identity in ("", ".", "..") or Path(identity).name != identity or "/" in identity or "\\" in identity:
        raise ValueError("输入的 task_id 不能用于输出目录")
    destination = output_root / identity
    if destination.resolve() == source.parent or source.parent in destination.resolve().parents:
        raise ValueError("输出目录必须与输入结果目录独立")
    if any((destination / filename).resolve() == source for filename in ("result.json", "metrics.json", "changes.csv")):
        raise ValueError("输出文件不能指向输入结果文件")
    destination.mkdir(parents=True, exist_ok=True)
    metrics = comparison(before, after, result["document_text"], agent_result, annotations)
    metrics.update({"document": identity, "input_path": str(source), "output_dir": str(destination)})
    (destination / "result.json").write_text(json.dumps(replayed, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    export_changes(destination / "changes.csv", agent_result.get("changes", []), identity)
    return metrics
