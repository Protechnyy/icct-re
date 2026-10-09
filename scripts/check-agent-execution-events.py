import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.agent import verify_document
from app.agent.document import DocumentIndex
from app.agent.relations import assign_relation_ids
from app.agent.trace import Trace
from app.agent.workspace import RelationWorkspace
from app.api import create_app
from app.config import AppConfig
from app.task_store import RedisTaskStore
from app.types import TaskStatus


def main():
    parser = argparse.ArgumentParser(description="使用实际结果、工具及 Redis 检查智能体执行事件")
    parser.add_argument("result", type=Path)
    parser.add_argument("--history-result", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--live", action="store_true", help="向配置的模型发送输入原文并运行核查")
    parser.add_argument("--max-calls", type=int, default=24)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    config = AppConfig.from_env()
    store = RedisTaskStore(config.redis_url)
    assert store.healthcheck()
    index = DocumentIndex(payload["relation_sections"], payload["document_text"])
    relations = assign_relation_ids(payload.get("pre_agent_relations", payload["final_relations"]))
    identity = "agent-event-check-" + uuid4().hex
    store.create_task(TaskStatus(task_id=identity, filename=payload["document_meta"]["filename"],
                                status="verifying", stage="agent_verification", progress=70),
                      {"source_result": str(args.result.resolve())})

    def progress(snapshot):
        assert store.get_agent_event(identity, snapshot["last_seq"]) is not None
        store.update_task(identity, agent_progress=snapshot)

    trace = Trace(progress, document_task_id=identity,
                  event_callback=lambda event: store.append_agent_event(identity, event))
    trace.emit("phase_start", summary="智能体校验已开始")
    section_id = next(key for key, section in index.sections.items() if len(section.get("text", "")) > 500)
    original = index.read_section(section_id)
    event = trace.emit("tool", "data-check", original, tool="read_section", args={"section_id": section_id})
    assert len(event["result"]) == 500 and event["truncated"]
    assert store.get_agent_event(identity, event["seq"])["result_data"] == original
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: trace.emit("tool", "data-check", index.search_document(relations[0]["head"]),
                                              tool="search_document"), range(40)))
    workspace = RelationWorkspace(relations, index)
    relation = next(item for item in workspace.relations if index.locate_evidence(item.get("evidence"))["matched"])
    changed = workspace.correct_relation(relation["relation_id"], json.dumps({"evidence": relation["evidence"]}), "data-check")
    assert changed["change"]["change_id"] == workspace.changes[0]["change_id"]
    rejected = workspace.correct_relation(relation["relation_id"], json.dumps({"evidence": ""}), "data-check")
    assert rejected["error"]
    trace.emit("tool", "data-check", rejected, tool="correct_relation", status="rejected")
    missing = index.search_document("")
    assert missing["matches"] == []
    trace.emit("tool", "data-check", missing, tool="search_document")
    agent = None
    if args.live:
        after, agent = verify_document(relations, payload["relation_sections"], payload["document_text"],
                                      replace(config, agent_enabled=True, agent_max_tasks=3,
                                              agent_max_steps=12, agent_max_llm_calls=args.max_calls), trace=trace)
        trace.finish(agent)
        starts = sorted(event["call_id"] for event in trace.events if event["kind"] in ("model_start", "tool_start"))
        ends = sorted(event["call_id"] for event in trace.events if event["kind"] in ("model_end", "tool" ) and event.get("call_id"))
        assert starts == ends
        assert agent["trace"][-1]["kind"] == "phase_end"
        assert agent["accepted_change_ids"] == [change["change_id"] for change in agent["changes"]]
        result = {**payload, "document_meta": {**payload["document_meta"], "task_id": identity},
                  "final_relations": after, "final_relation_list": {"relation_list": after}, "agent_result": agent}
        store.set_result(identity, result)
        store.update_task(identity, status="succeeded", stage="completed", progress=100)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    client = create_app().test_client()
    cursor = 0
    events = []
    while True:
        response = client.get(f"/api/agent/{identity}/events?after_seq={cursor}&limit=7")
        assert response.status_code == 200
        assert response.json == client.get(f"/api/agent/{identity}/events?after_seq={cursor}&limit=7").json
        page = response.json
        events.extend(page["events"])
        cursor = page["next_seq"]
        if not page["has_more"]:
            break
    assert [event["seq"] for event in events] == list(range(1, len(trace.events) + 1))
    assert len(trace.progress()["recent_events"]) == 20
    assert client.get(f"/api/agent/{identity}/events?after_seq={cursor}").json["events"] == []
    for query in ("limit=0", "limit=201", "after_seq=-1", "after_seq=abc"):
        assert client.get(f"/api/agent/{identity}/events?{query}").status_code == 400
    assert client.get("/api/agent/missing-document/events").status_code == 404
    assert client.get(f"/api/agent/{identity}/events/999999").status_code == 404
    history_id = None
    if args.history_result:
        history = json.loads(args.history_result.read_text(encoding="utf-8"))
        history_id = "agent-history-check-" + uuid4().hex
        store.create_task(TaskStatus(task_id=history_id, filename=history["document_meta"]["filename"],
                                    status="succeeded", stage="completed", progress=100),
                          {"source_result": str(args.history_result.resolve())})
        store.set_result(history_id, history)
        old = client.get(f"/api/agent/{history_id}/events").json
        assert old["version"] == history.get("agent_result", {}).get("version", 1)
        for event in old["events"]:
            detail = client.get(f"/api/agent/{history_id}/events/{event['seq']}").json
            assert detail["detail_available"] == ("result_data" in detail)
    if not args.live:
        store.update_task(identity, status="cancelled", stage="cancelled", progress=100)
    report = {"task_id": identity, "history_task_id": history_id, "source": str(args.result.resolve()),
              "events": len(events), "concurrency": True, "details": True, "pagination": True,
              "http_errors": True, "rejected": True, "not_found": True, "live": args.live,
              "agent_status": agent["status"] if agent else None}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
