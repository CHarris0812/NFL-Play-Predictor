export const API_BASE = "http://localhost:8000";

async function asJson(response) {
  if (!response.ok) {
    const body = await response.json();
    throw new Error(body.detail || "Request failed");
  }
  return response.json();
}

export function predictSituation(payload) {
  return fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then(asJson);
}

export function trainModels() {
  return fetch(`${API_BASE}/train`, { method: "POST" }).then(asJson);
}

export function fetchReplay(gameId) {
  return fetch(`${API_BASE}/replay/${encodeURIComponent(gameId)}`).then(asJson);
}
