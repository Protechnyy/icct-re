import Alert from "@arco-design/web-react/es/Alert";
import Button from "@arco-design/web-react/es/Button";
import Card from "@arco-design/web-react/es/Card";
import Empty from "@arco-design/web-react/es/Empty";
import Input from "@arco-design/web-react/es/Input";
import Message from "@arco-design/web-react/es/Message";
import Space from "@arco-design/web-react/es/Space";
import Tabs from "@arco-design/web-react/es/Tabs";
import Tag from "@arco-design/web-react/es/Tag";
import Tooltip from "@arco-design/web-react/es/Tooltip";
import Typography from "@arco-design/web-react/es/Typography";
import { IconDownload, IconFile, IconRefresh, IconSearch } from "@arco-design/web-react/icon";
import { cloneElement, useMemo, useState } from "react";
import { exportEntitiesCsv, exportTaskCsv } from "../lib/api";
import { readExtractionResult } from "../lib/extractionResult";
import useLlmStream from "../lib/useLlmStream";
import { STATUS_CONFIG } from "./TaskTable";
import ResultEmptyState from "./ResultEmptyState";
import ResizableResultTable from "./ResizableResultTable";
import StreamConsole from "./StreamConsole";
import "@arco-design/web-react/es/Table/style/css.js";
import "@arco-design/web-react/es/Tabs/style/css.js";
import "@arco-design/web-react/es/Typography/style/css.js";

function downloadFile(filename, data, type = "application/json") {
  const blob = new Blob([typeof data === "string" ? data : JSON.stringify(data, null, 2)], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function DocumentPreview({ result, preview, task }) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [zoom, setZoom] = useState(100);
  const text = result?.document_text || preview?.document_text || result?.ocr_summary?.markdown_text || result?.document_meta?.markdown_text || result?.chunks?.map((chunk) => chunk.text).join("\n\n") || "";
  const highlightedText = useMemo(() => {
    if (!query.trim()) return text;
    const parts = text.split(new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi"));
    return parts.map((part, index) => part.toLocaleLowerCase() === query.trim().toLocaleLowerCase() ? <mark key={`${part}-${index}`}>{part}</mark> : part);
  }, [query, text]);
  if (!text) return <Empty description={["failed", "cancelled", "succeeded"].includes(task.status) ? "当前任务暂无可预览的文档原文。" : task.metadata?.source_type === "text" ? "文档读取完成后显示原文。" : "文档解析完成后显示原文。"} />;
  return <div className="document-preview"><div className="document-preview-toolbar"><Space size="small"><Button size="small" icon={<IconSearch />} onClick={() => setSearchOpen((current) => !current)}>搜索原文</Button><Button size="small" onClick={() => setZoom((current) => current >= 120 ? 90 : current + 10)}>{zoom}%</Button></Space><span>文档原文预览</span></div>{searchOpen && <div className="preview-search"><Input value={query} onChange={setQuery} allowClear placeholder="输入关键词，高亮原文匹配内容" /></div>}<article style={{ fontSize: `${zoom / 100}rem` }}>{highlightedText}</article></div>;
}

const STAGE_LABELS = {
  queued: "等待任务调度",
  layout_parsing: "版面解析",
  restructure_pages: "文档重构",
  relation_extraction: "关系抽取",
  entity_recognition: "实体识别",
  entity_merge: "实体归并",
  document_merge: "结果合并",
  completed: "处理完成",
  failed: "处理失败",
};

const TIMING_LABELS = {
  total_elapsed_seconds: "总耗时",
  layout_parsing_seconds: "版面解析",
  document_restructure_seconds: "文档重构",
  relation_extraction_seconds: "关系抽取",
  entity_recognition_seconds: "实体识别",
  entity_merge_seconds: "实体归并",
  document_merge_seconds: "结果合并",
  routing_seconds: "Skill 路由",
  coref_seconds: "指代消解",
  chunk_routing_seconds: "分块路由",
  extraction_seconds: "关系识别",
  summarize_seconds: "结果汇总",
  proofreading_seconds: "结果校验",
  refinement_seconds: "结果优化",
  domain_reflection_seconds: "领域复核",
  total_seconds: "抽取总耗时",
};

function formatDateTime(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value || "-" : date.toLocaleString("zh-CN", { hour12: false });
}

function formatDuration(value) {
  const seconds = Number(value);
  return Number.isFinite(seconds) ? `${seconds.toFixed(seconds < 10 ? 3 : 2)} 秒` : "-";
}

function ExecutionLog({ task, result }) {
  const timing = result?.stage_outputs?.timing || {};
  return <div className="execution-log">
    {task.error && <Alert type="error" title="任务错误" content={task.error} closable={false} />}
    <div className="log-row"><span>任务创建时间</span><span>{formatDateTime(task.created_at)}</span><Tag color="green">已记录</Tag></div>
    <div className="log-row"><span>当前阶段</span><span>{STAGE_LABELS[task.stage] || "处理中"}</span><Tag color={task.status === "failed" ? "red" : "arcoblue"}>{statusLabel(task.status)}</Tag></div>
    {Object.entries(timing).map(([stage, value]) => <div className="log-row" key={stage}><span>{TIMING_LABELS[stage] || "处理耗时"}</span><span>{formatDuration(value)}</span><Tag>耗时</Tag></div>)}
  </div>;
}

function statusLabel(status) {
  return STATUS_CONFIG[status]?.label || (status === "succeeded" ? "已完成" : "处理中");
}

function splitModeLabel(splitMode) {
  return {
    small_section: "小节",
    chapter: "章节",
    paragraph: "段落",
    fixed_sections: "固定小节数",
  }[splitMode] || "小节";
}

export default function ResultViewer({ task, result, resultError, onRetryResult, preview, previewError, onRetryPreview }) {
  const [activeTab, setActiveTab] = useState("preview");
  const [exportingCsv, setExportingCsv] = useState(false);
  const [entityWidths, setEntityWidths] = useState({});
  const [relationWidths, setRelationWidths] = useState({});
  const extracted = useMemo(() => {
    try { return readExtractionResult(result); }
    catch (error) { return { entities: [], relations: [], error: error.message }; }
  }, [result]);
  const relationRows = useMemo(() => extracted.relations.map((item, index) => ({
    ...item,
    resultRowKey: `${task?.task_id}:${index}`,
  })), [extracted, task?.task_id]);
  const llmStream = useLlmStream(task?.task_id, task?.status);
  if (!task) return <ResultEmptyState />;
  const status = STATUS_CONFIG[task.status] || { color: "gray", label: task.status || "未知" };
  const { relations, entities } = extracted;
  const entityColumns = [
    { key: "name", title: "实体名称", render: (_, item) => <div>{item.name}{item.aliases?.length > 0 && <div className="entity-inline-types">别名：{item.aliases.map((alias) => alias.name).join("、")}</div>}</div> },
    { key: "type", title: "实体类型", dataIndex: "type", width: 180 },
  ];
  const relationColumns = [
    { key: "subject", title: "头实体", render: (_, item) => <div>{item.subject || item.head || item.head_entity || "-"}<div className="entity-inline-types">{item.head_type || item.subject_type || "未标注"}</div></div> },
    { key: "relation", title: "关系", render: (_, item) => item.relation || "-", width: 150, ellipsis: true },
    { key: "object", title: "尾实体", render: (_, item) => <div>{item.object || item.tail || item.tail_entity || "-"}<div className="entity-inline-types">{item.tail_type || item.object_type || "未标注"}</div></div> },
  ];
  const tabItems = [
    { key: "preview", title: "文档预览", content: <DocumentPreview result={result} preview={preview} task={task} /> },
    { key: "entities", title: `实体 ${entities.length ? `(${entities.length})` : ""}`, content: <ResizableResultTable rowKey="key" columns={entityColumns} widths={entityWidths} onWidthsChange={setEntityWidths} defaultColumnWidth={260} data={entities} pagination={false} noDataElement={<Empty description="暂无实体结果" />} /> },
    { key: "relations", title: `关系 ${relations.length ? `(${relations.length})` : ""}`, content: <ResizableResultTable rowKey="resultRowKey" columns={relationColumns} widths={relationWidths} onWidthsChange={setRelationWidths} data={relationRows} pagination={false} noDataElement={<Empty description="暂无关系结果" />} /> },
    { key: "logs", title: "运行日志", content: <ExecutionLog task={task} result={result} /> },
  ];
  function renderTabHeader(tabProps, DefaultTabHeader) {
    return <div role="tablist" aria-label="抽取结果" aria-orientation="horizontal" onKeyDown={(event) => {
      const tab = event.target.closest('[role="tab"]');
      if (!tab) return;
      const tabs = [...event.currentTarget.querySelectorAll('[role="tab"]')];
      const currentIndex = tabs.indexOf(tab);
      const nextIndex = { ArrowRight: (currentIndex + 1) % tabs.length, ArrowLeft: (currentIndex + tabs.length - 1) % tabs.length, Home: 0, End: tabs.length - 1 }[event.key];
      if (nextIndex !== undefined) {
        event.preventDefault();
        tabs[nextIndex].focus();
        setActiveTab(tabItems[nextIndex].key);
      } else if (event.key === " ") {
        event.preventDefault();
        setActiveTab(tabItems[currentIndex].key);
      }
    }}><DefaultTabHeader {...tabProps} /></div>;
  }
  async function handleExportCsv(entityExport = false) {
    try {
      setExportingCsv(true);
      const blob = await (entityExport ? exportEntitiesCsv : exportTaskCsv)(task.task_id);
      const stem = task.filename.replace(/\.[^.]+$/, "");
      downloadBlob(`${stem}_${entityExport ? "entities" : "relations"}.csv`, blob);
      Message.success("CSV 导出成功");
    } catch (error) {
      Message.error(`CSV 导出失败：${String(error.message || error)}`);
    } finally {
      setExportingCsv(false);
    }
  }
  return <Card className="result-workspace" bordered={false}>
    <div className="task-summary-header">
      <div className="task-summary-main"><div className="summary-file-icon"><IconFile /></div><div className="task-summary-text"><Tooltip content={task.filename}><Typography.Title heading={2} ellipsis={{ showTooltip: true }}>{task.filename}</Typography.Title></Tooltip><div className="task-summary-meta"><Tag color={status.color}>{status.label}</Tag><span>粒度：{splitModeLabel(task.metadata?.split_mode)}</span>{extracted.revision && <span>抽取配置：版本 {extracted.revision}</span>}<span>创建于 {task.created_at ? new Date(task.created_at).toLocaleString("zh-CN") : "-"}</span></div></div></div>
      <Space className="task-summary-actions" wrap><Button icon={<IconDownload />} onClick={() => result && downloadFile(`${task.filename}.json`, result)} disabled={!result}>导出 JSON</Button><Button icon={<IconDownload />} loading={exportingCsv} disabled={!result || !!extracted.error} onClick={() => handleExportCsv(true)}>导出实体 CSV</Button><Button type="primary" icon={<IconDownload />} loading={exportingCsv} disabled={!result || !!extracted.error} onClick={() => handleExportCsv(false)}>导出关系 CSV</Button></Space>
    </div>
    <StreamConsole stream={llmStream} />
    {resultError && <Alert className="result-request-error" type="error" title="读取抽取结果失败" content={resultError} action={<Button onClick={onRetryResult}>重新读取</Button>} />}
    {previewError && !result && <Alert className="result-request-error" type="error" title="读取文档预览失败" content={previewError} action={<Button onClick={onRetryPreview}>重新读取</Button>} />}
    {extracted.error && <Alert className="result-request-error" type="error" title="结果格式错误" content={extracted.error} />}
    {!result && task.status === "failed" ? <Alert type="error" title="任务执行失败" content={task.error || "请检查运行日志后重新执行任务。"} closable={false} /> : null}
    <Tabs className="result-tabs" activeTab={activeTab} onChange={setActiveTab} type="line" animation={false} destroyOnHide renderTabHeader={renderTabHeader} renderTabTitle={(node, { isActive }) => cloneElement(node, { tabIndex: isActive ? 0 : -1 })}>
      {tabItems.map((item) => <Tabs.TabPane key={item.key} title={item.title}>
        <div className="result-tab-body" tabIndex={0} aria-label={`${item.title}内容`}>
          {item.content}
        </div>
      </Tabs.TabPane>)}
    </Tabs>
  </Card>;
}
