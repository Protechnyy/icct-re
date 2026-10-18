const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

export async function uploadFiles(fileList, relationOptions = {}) {
  const formData = new FormData();
  fileList.forEach((file) => formData.append("files", file));
  if (relationOptions.split_mode) {
    formData.append("split_mode", relationOptions.split_mode);
  }
  if (relationOptions.batch_size) {
    formData.append("batch_size", String(relationOptions.batch_size));
  }
  formData.append("stream_thinking", String(relationOptions.stream_thinking !== false));
  formData.append("reasoning_effort", relationOptions.reasoning_effort ?? "high");
  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function getTaskStatus(taskId) {
  const response = await fetch(`${API_BASE_URL}/status/${taskId}`);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function cancelTask(taskId) {
  const response = await fetch(`${API_BASE_URL}/tasks/${encodeURIComponent(taskId)}/cancel`, { method: "POST" });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "取消任务失败");
  }
  return payload;
}

export async function refreshTaskStatus(task) {
  try {
    return { ...await getTaskStatus(task.task_id), request_error: null };
  } catch (error) {
    return { ...task, request_error: `状态读取失败，下次查询将继续尝试：${String(error.message || error)}` };
  }
}

export async function getTaskResult(taskId) {
  const response = await fetch(`${API_BASE_URL}/result/${taskId}`);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function getTaskPreview(taskId) {
  const response = await fetch(`${API_BASE_URL}/tasks/${encodeURIComponent(taskId)}/preview`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "读取文档预览失败");
  return payload;
}

export async function exportTaskCsv(taskId) {
  const response = await fetch(`${API_BASE_URL}/result/${encodeURIComponent(taskId)}/csv`);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.blob();
}

export async function getHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function getModelSettings() {
  const response = await fetch(`${API_BASE_URL}/settings/model`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "读取模型设置失败");
  }
  return payload;
}

export async function updateModelSettings(settings) {
  const response = await fetch(`${API_BASE_URL}/settings/model`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "保存模型设置失败");
  }
  return payload;
}

export async function getExtractionSchema() {
  const response = await fetch(`${API_BASE_URL}/settings/extraction-schema`);
  const value = await response.json();
  if (!response.ok) throw Object.assign(new Error(value.error || "读取配置失败"), { path: value.path, status: response.status });
  return value;
}

export async function updateExtractionSchema(config) {
  const response = await fetch(`${API_BASE_URL}/settings/extraction-schema`, {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(config),
  });
  const value = await response.json();
  if (!response.ok) throw Object.assign(new Error(value.error || "保存配置失败"), { path: value.path, status: response.status });
  return value;
}

export async function exportEntitiesCsv(taskId) {
  const response = await fetch(`${API_BASE_URL}/result/${encodeURIComponent(taskId)}/entities.csv`);
  if (!response.ok) throw new Error(await response.text());
  return response.blob();
}
