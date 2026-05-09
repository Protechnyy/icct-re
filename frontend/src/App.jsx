import { Layout, message, Segmented } from "antd";
import { FileSearchOutlined, SettingOutlined } from "@ant-design/icons";
import { useEffect, useMemo, useRef, useState } from "react";
import ResultViewer from "./components/ResultViewer";
import SkillManager from "./components/SkillManager";
import TaskTable from "./components/TaskTable";
import UploadPanel from "./components/UploadPanel";
import { getTaskResult, getTaskStatus, uploadFiles } from "./lib/api";

const { Header, Content } = Layout;

export default function App() {
  const [messageApi, contextHolder] = message.useMessage();
  const [fileList, setFileList] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [tasks, setTasks] = useState([]);
  const [results, setResults] = useState({});
  const [activeTaskId, setActiveTaskId] = useState(null);
  const [view, setView] = useState("tasks");

  // Use refs to avoid recreating the polling interval when state changes
  const tasksRef = useRef(tasks);
  tasksRef.current = tasks;
  const resultsRef = useRef(results);
  resultsRef.current = results;

  useEffect(() => {
    const timer = window.setInterval(async () => {
      const currentTasks = tasksRef.current;
      const runningTasks = currentTasks.filter(
        (task) => !["succeeded", "failed"].includes(task.status)
      );
      if (!runningTasks.length) return;

      const nextTasks = await Promise.all(
        runningTasks.map(async (task) => {
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
        })
      );

      setTasks((current) =>
        current.map(
          (task) => nextTasks.find((item) => item.task_id === task.task_id) || task
        )
      );
    }, 2500);

    return () => window.clearInterval(timer);
  }, []);

  const activeTask = useMemo(
    () => tasks.find((task) => task.task_id === activeTaskId) || null,
    [tasks, activeTaskId]
  );

  async function submitFiles(rawFiles) {
    if (!rawFiles.length) return;
    try {
      setSubmitting(true);
      const response = await uploadFiles(rawFiles);
      setTasks((current) => {
        const merged = [...response.tasks, ...current];
        return merged;
      });
      setActiveTaskId(response.tasks[0]?.task_id || null);
      messageApi.success(`已创建 ${response.tasks.length} 个任务`);
      setFileList([]);
    } catch (error) {
      messageApi.error(`上传失败: ${String(error)}`);
    } finally {
      setSubmitting(false);
    }
  }

  async function handleUpload() {
    const rawFiles = fileList.map((item) => item.originFileObj).filter(Boolean);
    await submitFiles(rawFiles);
  }

  async function handleFileChange({ fileList: next }) {
    setFileList(next);
    const rawFiles = next.map((item) => item.originFileObj).filter(Boolean);
    if (!rawFiles.length || submitting) return;
    await submitFiles(rawFiles);
  }

  function handleRemove(file) {
    setFileList((prev) => prev.filter((f) => f.uid !== file.uid));
  }

  return (
    <Layout className="app-shell">
      {contextHolder}
      <Header className="app-header">
        <div>
          <h1 className="app-header-title">Document RE Workbench</h1>
          <div className="app-header-sub">PaddleOCR-VL + Skill4RE + vLLM / Qwen 文档级关系抽取工作台</div>
        </div>
        <Segmented
          value={view}
          onChange={setView}
          options={[
            { label: "任务", value: "tasks", icon: <FileSearchOutlined /> },
            { label: "Skills", value: "skills", icon: <SettingOutlined /> },
          ]}
        />
      </Header>
      <Content className="app-content">
        <div className="grid">
          <aside className="grid-sidebar">
            <UploadPanel
              fileList={fileList}
              onChange={handleFileChange}
              onSubmit={handleUpload}
              onRemove={handleRemove}
              submitting={submitting}
            />
            <TaskTable
              tasks={tasks}
              onSelectTask={setActiveTaskId}
              activeTaskId={activeTaskId}
            />
          </aside>
          <main className="grid-main">
            {view === "skills" ? (
              <SkillManager messageApi={messageApi} />
            ) : (
              <ResultViewer
                task={activeTask}
                result={activeTaskId ? results[activeTaskId] : null}
              />
            )}
          </main>
        </div>
      </Content>
    </Layout>
  );
}
