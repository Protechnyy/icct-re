import { useCallback, useEffect, useRef, useState } from "react";
import { getAgentEvent, getAgentEvents } from "./api";

const finishedStatuses = ["completed", "partial", "failed", "skipped", "disabled"];

export default function useAgentEvents(task, result) {
  const taskId = task?.task_id;
  const eligible = Boolean(taskId && (task.agent_progress || task.stage === "agent_verification" ||
    (result?.agent_result && result.agent_result.status !== "disabled")));
  const cache = useRef({});
  const detailControllers = useRef(new Set());
  const [, render] = useState(0);
  const [retry, setRetry] = useState(0);

  useEffect(() => () => {
    for (const controller of detailControllers.current) controller.abort();
    detailControllers.current.clear();
  }, [taskId]);

  useEffect(() => {
    if (!eligible) return;
    const controller = new AbortController();
    let busy = false;
    let stopped = false;
    cache.current[taskId] ||= { events: [], details: {}, next_seq: 0 };
    async function refresh() {
      if (busy || stopped) return;
      busy = true;
      try {
        let page;
        do {
          const previous = cache.current[taskId];
          page = await getAgentEvents(taskId, previous.next_seq, controller.signal);
          if (controller.signal.aborted) return;
          const events = [...new Map([...previous.events, ...page.events].map((event) => [event.seq, event])).values()]
            .sort((first, second) => first.seq - second.seq);
          cache.current[taskId] = { ...previous, ...page, events, error: null, loading: false };
          render((version) => version + 1);
        } while (page.has_more);
        stopped = finishedStatuses.includes(page.status) && page.next_seq >= page.last_seq;
      } catch (error) {
        if (controller.signal.aborted) return;
        cache.current[taskId] = { ...cache.current[taskId], error: error.message, loading: false };
        render((version) => version + 1);
      } finally {
        busy = false;
      }
    }
    refresh();
    const timer = window.setInterval(refresh, 2500);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [taskId, eligible, retry]);

  const readDetail = useCallback(async (seq) => {
    const documentId = taskId;
    const controller = new AbortController();
    detailControllers.current.add(controller);
    const previous = cache.current[documentId] || { events: [], details: {}, next_seq: 0 };
    cache.current[documentId] = { ...previous, details: { ...previous.details, [seq]: { loading: true } } };
    render((version) => version + 1);
    try {
      const event = await getAgentEvent(documentId, seq, controller.signal);
      const current = cache.current[documentId];
      cache.current[documentId] = { ...current, details: { ...current.details, [seq]: { event } } };
    } catch (error) {
      const current = cache.current[documentId];
      if (controller.signal.aborted) {
        const details = { ...current.details };
        delete details[seq];
        cache.current[documentId] = { ...current, details };
      } else {
        cache.current[documentId] = { ...current, details: { ...current.details, [seq]: { error: error.message } } };
      }
    } finally {
      detailControllers.current.delete(controller);
    }
    render((version) => version + 1);
  }, [taskId]);

  return { ...(cache.current[taskId] || { events: [], details: {}, loading: eligible }),
           readDetail, retry: () => setRetry((version) => version + 1), eligible };
}
