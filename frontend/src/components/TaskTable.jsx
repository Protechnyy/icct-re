import { Button, Card, Dropdown, Empty, Input, Menu, Progress, Select, Tag, Tooltip } from "@arco-design/web-react";
import { IconFile, IconFileImage, IconFilePdf, IconMore, IconSearch } from "@arco-design/web-react/icon";
import { useMemo, useState } from "react";

export const STATUS_CONFIG = {
  queued: { color: "gray", label: "等待中", stage: "等待任务调度" },
  ocr_running: { color: "arcoblue", label: "OCR 处理中", stage: "正在识别文档内容" },
  extracting: { color: "arcoblue", label: "抽取中", stage: "正在抽取文档关系" },
  verifying: { color: "arcoblue", label: "核查中", stage: "正在核查文档关系" },
  merging: { color: "arcoblue", label: "结果整合", stage: "正在整理抽取结果" },
  succeeded: { color: "green", label: "已完成", stage: "关系抽取完成" },
  failed: { color: "red", label: "失败", stage: "任务处理失败" },
  cancelled: { color: "orange", label: "已取消", stage: "任务已取消" },
};

function taskIcon(filename = "") {
  if (/\.pdf$/i.test(filename)) return <IconFilePdf />;
  if (/\.(png|jpe?g)$/i.test(filename)) return <IconFileImage />;
  return <IconFile />;
}

function relativeTime(value) {
  if (!value) return "刚刚";
  const difference = Date.now() - new Date(value).getTime();
  if (Number.isNaN(difference) || difference < 60000) return "刚刚";
  if (difference < 3600000) return `${Math.floor(difference / 60000)} 分钟前`;
  if (difference < 86400000) return `${Math.floor(difference / 3600000)} 小时前`;
  return new Date(value).toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}

function TaskListItem({ task, active, onSelect, onAction }) {
  const [menuVisible, setMenuVisible] = useState(false);
  const [busy, setBusy] = useState(false);
  const finished = ["succeeded", "failed", "cancelled"].includes(task.status);

  async function handleAction(action) {
    setMenuVisible(false);
    setBusy(true);
    try {
      await onAction(action, task);
    } finally {
      setBusy(false);
    }
  }
  const config = STATUS_CONFIG[task.status] || { color: "gray", label: task.status || "未知", stage: task.stage || "处理中" };
  const stage = task.stage === "agent_verification" ? "文档级核查" : task.stage && task.stage !== task.status ? task.stage : config.stage;
  return (
    <div className={`task-list-item ${active ? "is-active" : ""}`}>
      <button type="button" className="task-select" title={task.filename} aria-label={`查看任务：${task.filename}`} aria-pressed={active} onClick={() => onSelect(task.task_id)} />
      <div className="task-item-topline">
        <div className="task-file-title">
          <span className="task-file-icon">{taskIcon(task.filename)}</span>
          <Tooltip content={task.filename}><span className="task-file-name">{task.filename}</span></Tooltip>
        </div>
        <Tag color={config.color} size="small">{config.label}</Tag>
      </div>
      <div className="task-stage">{stage}</div>
      {task.stage === "agent_verification" && task.agent_progress && <div className="task-stage">
        已完成 {task.agent_progress.completed_tasks || 0}/{task.agent_progress.total_tasks || 0}
        {task.agent_progress.current_task?.reason && <span> · {task.agent_progress.current_task.reason}</span>}
      </div>}
      <Progress percent={Number(task.progress) || 0} size="small" showText className="task-progress" />
      {task.error && <div className="task-error" title={task.error}>{task.error}</div>}
      <div className="task-item-footer">
        <span>{relativeTime(task.created_at)}</span>
        <span>{config.label}</span>
        <Dropdown
          trigger="click"
          position="br"
          popupVisible={menuVisible}
          onVisibleChange={setMenuVisible}
          droplist={(
            <Menu onClickMenuItem={handleAction}>
              <Menu.Item key="view">查看详情</Menu.Item>
              <Menu.Item key="refresh">刷新状态</Menu.Item>
              {task.status === "succeeded" && <Menu.Item key="export">导出 CSV</Menu.Item>}
              {task.error && <Menu.Item key="error">查看错误详情</Menu.Item>}
              {finished && <Menu.Item key="remove">从列表移除</Menu.Item>}
            </Menu>
          )}
        >
          <Button type="text" size="mini" className="task-more" icon={<IconMore />} loading={busy} disabled={busy}
            aria-label={`任务操作：${task.filename}`} aria-haspopup="menu" aria-expanded={menuVisible} />
        </Dropdown>
      </div>
    </div>
  );
}

export default function TaskTable({ tasks, onSelectTask, activeTaskId, onTaskAction }) {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("all");
  const filteredTasks = useMemo(() => tasks.filter((task) => {
    const matchName = task.filename?.toLowerCase().includes(query.trim().toLowerCase());
    return matchName && (status === "all" || task.status === status);
  }), [tasks, query, status]);

  return (
    <Card className="task-list-card" title="任务列表" bordered>
      <div className="task-filter-bar">
        <Input value={query} onChange={setQuery} prefix={<IconSearch />} placeholder="搜索文件名" allowClear />
        <Select value={status} onChange={setStatus} options={[
          { value: "all", label: "全部状态" },
          ...Object.entries(STATUS_CONFIG).map(([value, option]) => ({ value, label: option.label })),
        ]} />
      </div>
      <div className="task-list-scroll">
        {filteredTasks.length ? filteredTasks.map((task) => (
          <TaskListItem key={task.task_id} task={task} active={activeTaskId === task.task_id} onSelect={onSelectTask} onAction={onTaskAction} />
        )) : <Empty description="暂无抽取任务" />}
      </div>
    </Card>
  );
}
