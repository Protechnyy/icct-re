import { useEffect, useRef, useState } from "react";

const TERMINAL_STATUSES = ["succeeded", "failed", "cancelled"];
// 终态任务回看时若通道为空（历史任务无流事件），超时后主动断开
const EMPTY_TIMEOUT_MS = 15000;
// delta 累积写入 ref，仅在节流间隔到期后触发渲染
const FLUSH_INTERVAL_MS = 150;
// 最近接收时间落在该窗口内的流视为活跃流
const ACTIVE_WINDOW_MS = 3000;
// 连续错误达到该次数进入降级态，静默断开（不弹全局报错）
const MAX_CONNECT_ERRORS = 5;

export default function useLlmStream(taskId, taskStatus) {
  const streamsRef = useRef(new Map());
  const stageRef = useRef(null);
  const doneRef = useRef("");
  const activeStreamIdRef = useRef("");
  const activeCountRef = useRef(0);
  const degradedRef = useRef(false);
  const noRecordRef = useRef(false);
  const dirtyRef = useRef(false);
  const taskStatusRef = useRef(taskStatus);
  taskStatusRef.current = taskStatus;
  const [, render] = useState(0);

  function flush() {
    const now = Date.now();
    let changed = dirtyRef.current;
    let latestId = "";
    let latestAt = 0;
    let activeCount = 0;
    for (const stream of streamsRef.current.values()) {
      if (stream.lastAt > latestAt) {
        latestAt = stream.lastAt;
        latestId = stream.id;
      }
      // 活跃状态翻转（流停更或恢复输出）也要触发一次渲染，供界面折叠该流
      const active = now - stream.lastAt <= ACTIVE_WINDOW_MS;
      if (stream.active !== active) {
        stream.active = active;
        changed = true;
      }
      if (active) activeCount += 1;
    }
    if (!changed) return;
    dirtyRef.current = false;
    activeStreamIdRef.current = latestId;
    activeCountRef.current = activeCount;
    render((version) => version + 1);
  }

  useEffect(() => {
    if (!taskId) return;
    streamsRef.current = new Map();
    stageRef.current = null;
    doneRef.current = "";
    activeStreamIdRef.current = "";
    activeCountRef.current = 0;
    degradedRef.current = false;
    noRecordRef.current = false;
    dirtyRef.current = false;

    function ensureStream(id) {
      let stream = streamsRef.current.get(id);
      if (!stream) {
        stream = { id, thinkParts: [], answerParts: [], think: "", answer: "", lastKind: "", lastAt: Date.now(), active: true, nextKey: 0 };
        streamsRef.current.set(id, stream);
      }
      return stream;
    }

    function handleEvent(event) {
      const data = JSON.parse(event.data);
      if (data.kind === "stage") {
        stageRef.current = { ...JSON.parse(data.text), at: Date.now() };
        dirtyRef.current = true;
        return;
      }
      if (data.kind === "done") {
        doneRef.current = data.text;
        dirtyRef.current = true;
        return;
      }
      if (data.kind !== "think" && data.kind !== "answer") return;
      const stream = ensureStream(data.stream_id);
      const parts = data.kind === "think" ? stream.thinkParts : stream.answerParts;
      parts.push({ key: stream.nextKey, text: data.text });
      stream.nextKey += 1;
      if (data.kind === "think") stream.think += data.text;
      else stream.answer += data.text;
      stream.lastKind = data.kind;
      stream.lastAt = Date.now();
      dirtyRef.current = true;
    }

    // 首连以 after_seq=0 重放通道全部历史，断线重连由浏览器携带 Last-Event-ID 续传
    const source = new EventSource(`/api/tasks/${taskId}/stream?after_seq=0`);
    let errorCount = 0;
    let closed = false;
    let emptyTimer = 0;

    source.addEventListener("delta", (event) => {
      if (emptyTimer) {
        window.clearTimeout(emptyTimer);
        emptyTimer = 0;
      }
      handleEvent(event);
    });
    source.addEventListener("end", (event) => {
      // end 事件携带 done 的同一份数据，统一入状态后关闭连接
      handleEvent(event);
      flush();
      closed = true;
      source.close();
    });
    source.onopen = () => { errorCount = 0; };
    source.onerror = () => {
      errorCount += 1;
      if (source.readyState === EventSource.CLOSED || errorCount >= MAX_CONNECT_ERRORS) {
        closed = true;
        source.close();
        degradedRef.current = true;
        flush();
      }
    };

    const flushTimer = window.setInterval(flush, FLUSH_INTERVAL_MS);

    // 终态任务挂载时通道可能为空，15 秒无事件即断开，避免长期挂着心跳
    if (TERMINAL_STATUSES.includes(taskStatusRef.current)) {
      emptyTimer = window.setTimeout(() => {
        if (!closed) {
          closed = true;
          noRecordRef.current = true;
          dirtyRef.current = true;
          flush();
          source.close();
        }
      }, EMPTY_TIMEOUT_MS);
    }

    return () => {
      closed = true;
      window.clearInterval(flushTimer);
      if (emptyTimer) window.clearTimeout(emptyTimer);
      source.close();
    };
  }, [taskId]);

  return {
    streams: [...streamsRef.current.values()],
    stage: stageRef.current,
    done: doneRef.current,
    activeStreamId: activeStreamIdRef.current,
    activeCount: activeCountRef.current,
    degraded: degradedRef.current,
    noRecord: noRecordRef.current,
  };
}
