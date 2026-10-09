import Alert from "@arco-design/web-react/es/Alert";
import Badge from "@arco-design/web-react/es/Badge";
import Button from "@arco-design/web-react/es/Button";
import Message from "@arco-design/web-react/es/Message";
import Modal from "@arco-design/web-react/es/Modal";
import Spin from "@arco-design/web-react/es/Spin";
import Tooltip from "@arco-design/web-react/es/Tooltip";
import {
  IconApps,
  IconRobot,
} from "@arco-design/web-react/icon";
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import ResultEmptyState from "./components/ResultEmptyState";
import TaskTable from "./components/TaskTable";
import UploadPanel from "./components/UploadPanel";
import useAgentEvents from "./lib/useAgentEvents";
import { exportTaskCsv, getHealth, getTaskResult, getTaskStatus, refreshTaskStatus, uploadFiles } from "./lib/api";

const terminalStatuses = ["succeeded", "failed", "cancelled"];
const ResultViewer = lazy(() => import("./components/ResultViewer"));
const SkillManager = lazy(() => import("./components/SkillManager"));

function readWorkspaceTaskIds() {
  const taskIds = JSON.parse(localStorage.getItem("docre-task-ids") || "[]");
  if (!Array.isArray(taskIds) || taskIds.some((id) => typeof id !== "string" || !id)) {
    throw new Error("任务记录格式无效");
  }
  const activeId = localStorage.getItem("docre-active-task");
  return [...new Set([...taskIds, ...(activeId ? [activeId] : [])])];
}

export default function App() {
  const [fileList, setFileList] = useState([]);
  const [relationOptions, setRelationOptions] = useState({
    split_mode: "small_section",
    batch_size: 1,
    fast_mode: false,
  });
  const [submitting, setSubmitting] = useState(false);
  const [tasks, setTasks] = useState([]);
  const [results, setResults] = useState({});
  const [workspaceTaskIds, setWorkspaceTaskIds] = useState(readWorkspaceTaskIds);
  const [restoring, setRestoring] = useState(false);
  const [restoreError, setRestoreError] = useState("");
  const [resultError, setResultError] = useState("");
  const [resultRetry, setResultRetry] = useState(0);
  const [activeTaskId, setActiveTaskId] = useState(() => localStorage.getItem("docre-active-task"));
  const [view, setView] = useState("tasks");
  const [health, setHealth] = useState({ loading: true, status: "checking" });

  const tasksRef = useRef(tasks);
  tasksRef.current = tasks;
  const resultsRef = useRef(results);
  resultsRef.current = results;
  const workspaceTaskIdsRef = useRef(workspaceTaskIds);
  workspaceTaskIdsRef.current = workspaceTaskIds;
  const restorationRunRef = useRef(0);

  async function restoreTasks(taskIds) {
    const run = ++restorationRunRef.current;
    setRestoring(true);
    setRestoreError("");
    const responses = await Promise.allSettled(taskIds.map(getTaskStatus));
    if (run !== restorationRunRef.current) return;
    const restoredTasks = responses.filter((response) => response.status === "fulfilled").map((response) => response.value);
    const failures = responses.flatMap((response, index) => response.status === "rejected" ? [`${taskIds[index]}：${String(response.reason.message || response.reason)}`] : []);
    setTasks((current) => [
      ...current,
      ...restoredTasks.filter((task) => workspaceTaskIdsRef.current.includes(task.task_id) && !current.some((item) => item.task_id === task.task_id)),
    ]);
    if (failures.length) setRestoreError(`部分任务读取失败：${failures.join("；")}`);
    setRestoring(false);
  }

  useEffect(() => {
    restoreTasks(workspaceTaskIdsRef.current);
    return () => { restorationRunRef.current += 1; };
  }, []);

  useEffect(() => {
    localStorage.setItem("docre-task-ids", JSON.stringify(workspaceTaskIds));
  }, [workspaceTaskIds]);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((payload) => !cancelled && setHealth({ loading: false, ...payload }))
      .catch(() => !cancelled && setHealth({ loading: false, status: "degraded" }));
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (activeTaskId) localStorage.setItem("docre-active-task", activeTaskId);
    else localStorage.removeItem("docre-active-task");
  }, [activeTaskId]);

  useEffect(() => {
    let polling = false;
    let cancelled = false;
    const timer = window.setInterval(async () => {
      if (polling) return;
      const runningTasks = tasksRef.current.filter((task) => !terminalStatuses.includes(task.status));
      if (!runningTasks.length) return;
      polling = true;
      const nextTasks = await Promise.all(runningTasks.map(refreshTaskStatus));
      polling = false;
      if (cancelled) return;
      setTasks((current) => current.map((task) => {
        const next = nextTasks.find((item) => item.task_id === task.task_id);
        return next ? { ...task, ...next, metadata: { ...task.metadata, ...next.metadata } } : task;
      }));
    }, 2500);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, []);

  const activeTask = useMemo(
    () => tasks.find((task) => task.task_id === activeTaskId) || null,
    [tasks, activeTaskId]
  );
  const agentEvents = useAgentEvents(activeTask, results[activeTaskId]);

  useEffect(() => {
    setResultError("");
    if (!activeTask || activeTask.status !== "succeeded" || resultsRef.current[activeTask.task_id]) return;
    let cancelled = false;
    getTaskResult(activeTask.task_id)
      .then((result) => {
        if (!cancelled) setResults((current) => ({ ...current, [activeTask.task_id]: result }));
      })
      .catch((error) => {
        if (!cancelled) setResultError(String(error.message || error));
      });
    return () => { cancelled = true; };
  }, [activeTask?.task_id, activeTask?.status, resultRetry]);

  async function submitFiles(rawFiles) {
    if (!rawFiles.length || submitting) return;
    try {
      setSubmitting(true);
      const response = await uploadFiles(rawFiles, relationOptions);
      const newTasks = response.tasks.map((task) => ({
        ...task,
        metadata: { ...task.metadata, split_mode: relationOptions.split_mode },
      }));
      setTasks((current) => [...newTasks, ...current]);
      setWorkspaceTaskIds((current) => [...new Set([...newTasks.map((task) => task.task_id), ...current])]);
      setActiveTaskId(response.tasks[0]?.task_id || null);
      setView("tasks");
      setFileList([]);
      Message.success(`已创建 ${response.tasks.length} 个抽取任务`);
    } catch (error) {
      Message.error(`上传失败：${String(error.message || error)}`);
    } finally {
      setSubmitting(false);
    }
  }

  function handleUpload() {
    const rawFiles = fileList.map((item) => item.originFile).filter((file) => file instanceof File);
    if (!rawFiles.length) {
      Message.error("未读取到可上传的原始文件，请重新选择文件后再试。");
      return;
    }
    submitFiles(rawFiles);
  }

  async function handleTaskAction(action, task) {
    const taskId = task.task_id;
    try {
      if (action === "view") {
        setActiveTaskId(taskId);
      } else if (action === "refresh") {
        const status = await getTaskStatus(taskId);
        setTasks((current) => current.map((item) => item.task_id === taskId
          ? { ...item, ...status, request_error: null, metadata: { ...item.metadata, ...status.metadata } } : item));
        if (status.status === "succeeded") {
          const result = await getTaskResult(taskId);
          setResults((current) => ({ ...current, [taskId]: result }));
        }
        Message.success("任务状态已刷新");
      } else if (action === "export") {
        const blob = await exportTaskCsv(taskId);
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `${task.filename.replace(/\.[^.]+$/, "")}_relations.csv`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      } else if (action === "error") {
        Modal.error({ title: "任务错误详情", content: <div style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", maxHeight: "60vh", overflowY: "auto" }}>{task.error}</div> });
      } else if (action === "remove") {
        const remaining = tasksRef.current.filter((item) => item.task_id !== taskId);
        setTasks((current) => current.filter((item) => item.task_id !== taskId));
        setWorkspaceTaskIds((current) => current.filter((id) => id !== taskId));
        setResults((current) => {
          const next = { ...current };
          delete next[taskId];
          return next;
        });
        setActiveTaskId((current) => current === taskId ? remaining[0]?.task_id || null : current);
        Message.success("已从列表移除，服务器文件保留");
      }
    } catch (error) {
      Message.error(`操作失败：${String(error.message || error)}`);
    }
  }

  function updateFiles(nextFileList) {
    setFileList(nextFileList);
  }

  function removeFile(file) {
    setFileList((current) => current.filter((item) => item.uid !== file.uid));
  }

  const healthColor = health.status === "ok" ? "success" : health.loading ? "processing" : "warning";
  const healthText = health.status === "ok" ? "服务正常" : health.loading ? "服务检查中" : "服务待检查";

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-block">
          <img className="brand-logo" src="/logo.png?v=agent-relation-blue" width={40} height={40} alt="ICCT-RE 智能体关系抽取标识" />
          <div>
            <div className="brand-title">ICCT-RE</div>
            <div className="brand-subtitle">文档级关系抽取工作台</div>
          </div>
        </div>
        <nav className="header-nav" aria-label="主导航">
          <Button type={view === "tasks" ? "secondary" : "text"} aria-current={view === "tasks" ? "page" : undefined} icon={<IconApps />} onClick={() => setView("tasks")}>任务中心</Button>
          <Button type={view === "skills" ? "secondary" : "text"} aria-current={view === "skills" ? "page" : undefined} icon={<IconRobot />} onClick={() => setView("skills")}>Skills</Button>
        </nav>
        <div className="service-status">
          <Tooltip content={health.status === "ok" ? "Redis、OCR 与推理服务可用" : "服务状态暂不可用"}>
            <Badge status={healthColor} text={healthText} />
          </Tooltip>
          <span className="model-name">Qwen / vLLM</span>
        </div>
      </header>
      <main className="app-content">
        {view === "skills" ? (
          <Suspense fallback={<div className="page-loading" role="status"><Spin />正在加载 Skills…</div>}><SkillManager /></Suspense>
        ) : (
          <>
            {restoring && <div className="workspace-loading" role="status">正在恢复任务记录…</div>}
            {restoreError && <Alert className="workspace-request-error" type="error" title="读取任务记录失败" content={restoreError} action={<Button onClick={() => restoreTasks(workspaceTaskIds)} loading={restoring}>重新读取</Button>} />}
            <div className="workbench-grid">
              <aside className="workbench-sidebar">
                <UploadPanel
                  fileList={fileList}
                  onChange={updateFiles}
                  onSubmit={handleUpload}
                  onRemove={removeFile}
                  submitting={submitting}
                  relationOptions={relationOptions}
                  onRelationOptionsChange={setRelationOptions}
                />
                <TaskTable tasks={tasks} onSelectTask={setActiveTaskId} activeTaskId={activeTaskId} onTaskAction={handleTaskAction} />
              </aside>
              <section className="workbench-main">
                {activeTask ? <Suspense fallback={<div className="result-loading" role="status"><Spin />正在加载抽取结果…</div>}>
                  <ResultViewer task={activeTask} result={activeTaskId ? results[activeTaskId] : null} agentEvents={agentEvents} resultError={resultError} onRetryResult={() => setResultRetry((current) => current + 1)} />
                </Suspense> : <ResultEmptyState />}
              </section>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
