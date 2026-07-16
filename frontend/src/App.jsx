import { Badge, Button, Message, Tooltip } from "@arco-design/web-react";
import {
  IconApps,
  IconRobot,
} from "@arco-design/web-react/icon";
import { useEffect, useMemo, useRef, useState } from "react";
import ResultViewer from "./components/ResultViewer";
import SkillManager from "./components/SkillManager";
import TaskTable from "./components/TaskTable";
import UploadPanel from "./components/UploadPanel";
import { getHealth, getTaskResult, getTaskStatus, uploadFiles } from "./lib/api";

const terminalStatuses = ["succeeded", "failed", "cancelled"];

export default function App() {
  const [fileList, setFileList] = useState([]);
  const [relationOptions, setRelationOptions] = useState({
    split_mode: "small_section",
    batch_size: 1,
  });
  const [submitting, setSubmitting] = useState(false);
  const [tasks, setTasks] = useState([]);
  const [results, setResults] = useState({});
  const [activeTaskId, setActiveTaskId] = useState(() => localStorage.getItem("docre-active-task"));
  const [view, setView] = useState("tasks");
  const [health, setHealth] = useState({ loading: true, status: "checking" });

  const tasksRef = useRef(tasks);
  tasksRef.current = tasks;
  const resultsRef = useRef(results);
  resultsRef.current = results;

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
    const timer = window.setInterval(async () => {
      const runningTasks = tasksRef.current.filter((task) => !terminalStatuses.includes(task.status));
      if (!runningTasks.length) return;
      const nextTasks = await Promise.all(runningTasks.map(async (task) => {
        try {
          const status = await getTaskStatus(task.task_id);
          if (status.status === "succeeded" && !resultsRef.current[task.task_id]) {
            const result = await getTaskResult(task.task_id);
            setResults((current) => ({ ...current, [task.task_id]: result }));
          }
          return status;
        } catch (error) {
          return { ...task, status: "failed", error: String(error) };
        }
      }));
      setTasks((current) => current.map((task) => {
        const next = nextTasks.find((item) => item.task_id === task.task_id);
        return next ? { ...task, ...next, metadata: { ...task.metadata, ...next.metadata } } : task;
      }));
    }, 2500);
    return () => window.clearInterval(timer);
  }, []);

  const activeTask = useMemo(
    () => tasks.find((task) => task.task_id === activeTaskId) || null,
    [tasks, activeTaskId]
  );

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
          <img className="brand-logo" src="/logo.png" alt="ICCT-RE Logo" />
          <div>
            <div className="brand-title">ICCT-RE</div>
            <div className="brand-subtitle">文档级关系抽取工作台</div>
          </div>
        </div>
        <nav className="header-nav" aria-label="主导航">
          <Button type={view === "tasks" ? "secondary" : "text"} icon={<IconApps />} onClick={() => setView("tasks")}>任务中心</Button>
          <Button type={view === "skills" ? "secondary" : "text"} icon={<IconRobot />} onClick={() => setView("skills")}>Skills</Button>
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
          <SkillManager />
        ) : (
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
              <TaskTable tasks={tasks} onSelectTask={setActiveTaskId} activeTaskId={activeTaskId} />
            </aside>
            <section className="workbench-main">
              <ResultViewer task={activeTask} result={activeTaskId ? results[activeTaskId] : null} />
            </section>
          </div>
        )}
      </main>
    </div>
  );
}
