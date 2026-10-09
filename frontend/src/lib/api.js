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
  formData.append("fast_mode", String(relationOptions.fast_mode === true));
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

export async function getAgentEvents(taskId, afterSeq = 0, signal) {
  const response = await fetch(`${API_BASE_URL}/agent/${encodeURIComponent(taskId)}/events?after_seq=${afterSeq}&limit=100`, { signal });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function getAgentEvent(taskId, seq, signal) {
  const response = await fetch(`${API_BASE_URL}/agent/${encodeURIComponent(taskId)}/events/${seq}`, { signal });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
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

export async function listSkills() {
  const response = await fetch(`${API_BASE_URL}/skills`);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function createSkill(skill) {
  const response = await fetch(`${API_BASE_URL}/skills`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(skill),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function updateSkill(name, skill) {
  const response = await fetch(`${API_BASE_URL}/skills/${encodeURIComponent(name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(skill),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}
