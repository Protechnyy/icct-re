const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

export async function uploadFiles(fileList) {
  const formData = new FormData();
  fileList.forEach((file) => formData.append("files", file));
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

export async function getTaskResult(taskId) {
  const response = await fetch(`${API_BASE_URL}/result/${taskId}`);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
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
