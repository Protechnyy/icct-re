import Button from "@arco-design/web-react/es/Button";
import Tag from "@arco-design/web-react/es/Tag";
import { IconThunderbolt } from "@arco-design/web-react/icon";
import "@arco-design/web-react/es/Button/style/css.js";
import { memo, useEffect, useRef, useState } from "react";

// 滚动容器距底部小于该值视为接近底部，保持自动跟随
const FOLLOW_THRESHOLD = 60;

function streamStatus(stream, active) {
  if (active) return { label: stream.lastKind === "answer" ? "回答中" : "思考中", color: "arcoblue" };
  return { label: "已结束", color: "gray" };
}

function stateText(stream) {
  if (stream.done === "succeeded") return "输出完成";
  if (stream.done === "failed") return "输出中断";
  if (stream.stage || stream.streams.length) return "正在输出";
  return "等待输出";
}

function placeholder(stream) {
  if (stream.degraded) return "实时输出通道连接失败，当前展示可能不完整，任务进度与结果不受影响。";
  if (stream.noRecord) return "该任务没有保留流式输出记录。";
  return "等待模型输出…";
}

function stageText(stream) {
  const stage = stream.stage;
  if (stage?.phase === "extraction_started") return `抽取开始 · 每批并发 ${stage.concurrency} · 共 ${stage.batch_count} 批`;
  if (stage?.phase === "extraction_finished") return `抽取结束 · 输出关系 ${stage.relation_count} 条`;
  return "";
}

// 原始长度或活跃状态变化时才重渲染对应流，避免全部片段反复参与 diff
const StreamSection = memo(function StreamSection({ stream, active, thinkLen, answerLen }) {
  const [open, setOpen] = useState(true);
  const activeRef = useRef(true);
  // 该流的思考与回答都停更后自动折叠
  useEffect(() => {
    if (activeRef.current && !active) setOpen(false);
    activeRef.current = active;
  }, [active]);
  const status = streamStatus(stream, active);
  return <div className="llm-stream-section">
    <button type="button" className="llm-stream-section-head" onClick={() => setOpen((current) => !current)}>
      <span className="llm-stream-id">{stream.id}</span>
      <Tag size="small" color={status.color}>{status.label}</Tag>
      <span className="llm-stream-section-toggle">{open ? "收起" : "展开"}</span>
    </button>
    {open && <>
      {thinkLen > 0 && <div className="llm-stream-block llm-stream-block-think"><span className="llm-stream-kind-label">思考</span><div className="llm-stream-text">{stream.thinkParts.map((part) => <span key={part.key} className="llm-stream-delta">{part.text}</span>)}</div></div>}
      {answerLen > 0 && <div className="llm-stream-block llm-stream-block-answer"><span className="llm-stream-kind-label">回答</span><div className="llm-stream-text">{stream.answerParts.map((part) => <span key={part.key} className="llm-stream-delta">{part.text}</span>)}</div></div>}
    </>}
  </div>;
});

function StreamSectionList({ stream }) {
  return <>
    {stream.streams.map((item) => <StreamSection key={item.id} stream={item} active={item.active} thinkLen={item.think.length} answerLen={item.answer.length} />)}
  </>;
}

export default function StreamConsole({ stream }) {
  const [expanded, setExpanded] = useState(false);
  const scrollRef = useRef(null);
  const followRef = useRef(true);

  // 流输出结束后自动折叠
  useEffect(() => {
    if (stream.done) setExpanded(false);
  }, [stream.done]);

  // 展开态渲染后，仅在用户接近底部时自动跟随滚动
  useEffect(() => {
    if (!expanded || !followRef.current) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  });

  function toggleExpanded() {
    setExpanded((current) => {
      if (!current) followRef.current = true;
      return !current;
    });
  }

  function handleScroll() {
    const el = scrollRef.current;
    if (!el) return;
    followRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < FOLLOW_THRESHOLD;
  }

  const stageLine = stageText(stream);

  return <section className="agent-workspace" aria-label="实时抽取输出">
    <header className="agent-workspace-header">
      <div className="agent-workspace-heading"><IconThunderbolt /><h3>实时抽取输出</h3>
        <span className="agent-workspace-state" role="status">{stateText(stream)}</span></div>
      <div className="agent-workspace-controls">
        <Button type="text" size="mini" onClick={toggleExpanded} aria-expanded={expanded}>{expanded ? "折叠输出" : "展开输出"}</Button></div>
    </header>
    {expanded && <div className="agent-workspace-body">
      <div className="llm-stream-expanded" ref={scrollRef} onScroll={handleScroll}>
        {stageLine && <div className="llm-stream-stage">{stageLine}</div>}
        <StreamSectionList stream={stream} />
        {!stream.streams.length && <div className="llm-stream-placeholder">{placeholder(stream)}</div>}
      </div>
    </div>}
  </section>;
}
