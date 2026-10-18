import Button from "@arco-design/web-react/es/Button";
import Tag from "@arco-design/web-react/es/Tag";
import { IconThunderbolt } from "@arco-design/web-react/icon";
import "@arco-design/web-react/es/Button/style/css.js";
import { memo, useEffect, useRef, useState } from "react";

// 滚动容器距底部小于该值视为接近底部，保持自动跟随
const FOLLOW_THRESHOLD = 60;

// 已知技能的中文显示名，未收录的技能名保留原样
const SKILL_LABELS = {
  military: "军事",
  finance: "财务",
  legal: "法律",
  medical: "医疗",
  technology: "科技",
};

// stream_id 形如 batch-87-chunk0-legal、batch-86-routing、batch-0-proofread，可带 #retryN 后缀
function formatStreamId(id) {
  const [mainId, retryTag] = id.split("#");
  const retryText = retryTag ? `（重试 ${retryTag.replace("retry", "")}）` : "";
  const match = mainId.match(/^batch-(\d+)-(.+)$/);
  if (!match) return id;
  const batchNo = Number(match[1]) + 1;
  const tail = match[2];
  const stageLabels = { routing: "技能路由", proofread: "结果校验", summarize: "结果汇总", "targeted-proofread": "定向校验" };
  if (stageLabels[tail]) return `第 ${batchNo} 批 · ${stageLabels[tail]}${retryText}`;
  const chunkMatch = tail.match(/^chunk(\d+)-(.+)$/);
  if (!chunkMatch) return id;
  const skillLabel = SKILL_LABELS[chunkMatch[2]] || chunkMatch[2];
  const chunkText = chunkMatch[1] === "0" ? "抽取" : `抽取第 ${Number(chunkMatch[1]) + 1} 段`;
  return `第 ${batchNo} 批 · ${chunkText}（${skillLabel}）${retryText}`;
}

function streamStatus(stream, active) {
  if (active) return { label: stream.lastKind === "answer" ? "回答中" : "思考中", color: "arcoblue" };
  return { label: "已结束", color: "gray" };
}

function isRelationItem(obj) {
  return !!obj && typeof obj === "object"
    && (typeof obj.head === "string" || typeof obj.subject === "string")
    && typeof obj.relation === "string"
    && (typeof obj.tail === "string" || typeof obj.object === "string");
}

// 从未整体解析成功的回答文本中提取已写完整的关系对象；
// 输出中断时外层 relation_list 闭合不上，因此对每个闭合的 {} 片段
// 单独尝试解析，是关系条目就收集，断在半截或没写完的直接忽略
function extractRelationObjects(raw) {
  const items = [];
  const starts = [];
  let inString = false;
  let escaped = false;
  for (let i = 0; i < raw.length; i += 1) {
    const ch = raw[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') { inString = true; continue; }
    if (ch === "{") { starts.push(i); continue; }
    if (ch === "}" && starts.length) {
      const objStart = starts.pop();
      let obj = null;
      try { obj = JSON.parse(raw.slice(objStart, i + 1)); } catch { obj = null; }
      if (isRelationItem(obj)) items.push(obj);
    }
  }
  return items;
}

// 流结束后对回答做轻量解析：抽取与校验流输出含 relation_list 的 JSON，
// 路由流输出含 primary_skill 的 JSON；整体解析失败时兜底提取已完整的关系条目，
// 连一条都提不出时回退原始文本展示
function parseAnswer(answer) {
  if (!answer) return null;
  const fenced = answer.match(/```(?:json)?\s*([\s\S]*?)```/);
  const raw = fenced ? fenced[1] : answer;
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  let parsed = null;
  try { parsed = JSON.parse(raw.slice(start, end + 1)); } catch { parsed = null; }
  if (parsed && Array.isArray(parsed.relation_list)) {
    const items = parsed.relation_list.filter(isRelationItem);
    if (items.length) return { kind: "relations", items };
  }
  if (parsed && typeof parsed.primary_skill === "string") {
    return { kind: "routing", primary: parsed.primary_skill, aux: Array.isArray(parsed.aux_skills) ? parsed.aux_skills : [] };
  }
  const items = extractRelationObjects(raw);
  if (items.length) return { kind: "relations", items };
  return null;
}

function skillLabel(name) {
  return SKILL_LABELS[name] || name;
}

function ParsedAnswer({ parsed }) {
  if (parsed.kind === "relations") {
    return <ol className="llm-stream-answer-list">
      {parsed.items.map((item, index) => (
        <li key={index} className="llm-stream-answer-line">
          <span className="llm-stream-answer-head">{item.head || item.subject || "-"}</span>
          <span className="llm-stream-answer-relation">{item.relation || "-"}</span>
          <span className="llm-stream-answer-tail">{item.tail || item.object || "-"}</span>
        </li>
      ))}
    </ol>;
  }
  return <div className="llm-stream-answer-list">
    <div className="llm-stream-answer-line">
      <span className="llm-stream-answer-relation">主技能</span>
      <span className="llm-stream-answer-head">{skillLabel(parsed.primary)}</span>
      {parsed.aux.length > 0 && <span className="llm-stream-answer-tail">辅助：{parsed.aux.map(skillLabel).join("、")}</span>}
    </div>
  </div>;
}

function stateText(stream) {
  if (stream.done === "succeeded") return "输出完成";
  if (stream.done === "failed") return "输出中断";
  if (stream.done === "cancelled") return "输出已取消";
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
  const parsedAnswer = active ? null : parseAnswer(stream.answer);
  return <div className="llm-stream-section">
    <button type="button" className="llm-stream-section-head" onClick={() => setOpen((current) => !current)}>
      <span className="llm-stream-id">{formatStreamId(stream.id)}</span>
      <Tag size="small" color={status.color}>{status.label}</Tag>
      <span className="llm-stream-section-toggle">{open ? "收起" : "展开"}</span>
    </button>
    {open && <>
      {thinkLen > 0 && <div className="llm-stream-block llm-stream-block-think"><span className="llm-stream-kind-label">思考</span><div className="llm-stream-text">{stream.thinkParts.map((part) => <span key={part.key} className="llm-stream-delta">{part.text}</span>)}</div></div>}
      {answerLen > 0 && <div className="llm-stream-block llm-stream-block-answer"><span className="llm-stream-kind-label">回答</span>
        {active || !parsedAnswer
          ? <div className="llm-stream-text">{stream.answerParts.map((part) => <span key={part.key} className="llm-stream-delta">{part.text}</span>)}</div>
          : <ParsedAnswer parsed={parsedAnswer} />}
      </div>}
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
