/**
 * Typed fetch wrappers for all Synthoseis launcher API endpoints.
 *
 * In development the Vite dev server proxies /api → http://localhost:8000.
 * In production the FastAPI server serves the built frontend from /,
 * so relative URLs work in both environments.
 */
import type { RunRecord, SimulationConfig, RunStatus } from "../types/simulation";

const BASE = "";

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/** List available rock-physics model names. */
export async function fetchModels(): Promise<string[]> {
  const res = await fetch(`${BASE}/api/models`);
  if (!res.ok) throw new Error(`GET /api/models: ${res.status}`);
  return res.json();
}

/**
 * Open a native OS folder-picker dialog on the server machine.
 * Returns the selected absolute path, or null if the user cancelled.
 * Throws if tkinter is unavailable (headless server).
 */
export async function browseDirectory(
  initialDir?: string
): Promise<string | null> {
  const params = initialDir
    ? `?initial_dir=${encodeURIComponent(initialDir)}`
    : "";
  const res = await fetch(`${BASE}/api/browse-directory${params}`);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(`GET /api/browse-directory: ${res.status} — ${JSON.stringify(detail)}`);
  }
  const { path } = await res.json();
  return path ?? null;
}

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

/** List all simulation runs. */
export async function listRuns(): Promise<RunRecord[]> {
  const res = await fetch(`${BASE}/api/runs`);
  if (!res.ok) throw new Error(`GET /api/runs: ${res.status}`);
  return res.json();
}

/** Get a single run by ID. Returns null if not found (404). */
export async function getRun(runId: string): Promise<RunRecord | null> {
  const res = await fetch(`${BASE}/api/runs/${encodeURIComponent(runId)}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`GET /api/runs/${runId}: ${res.status}`);
  return res.json();
}

/** Submit a new simulation run. Returns {run_id, status}. */
export async function submitRun(
  config: SimulationConfig
): Promise<{ run_id: string; status: RunStatus }> {
  const res = await fetch(`${BASE}/api/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(`POST /api/runs: ${res.status} — ${JSON.stringify(detail)}`);
  }
  return res.json();
}

/** Cancel (DELETE) a running simulation. */
export async function deleteRun(runId: string): Promise<void> {
  const res = await fetch(`${BASE}/api/runs/${encodeURIComponent(runId)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`DELETE /api/runs/${runId}: ${res.status}`);
}

// ---------------------------------------------------------------------------
// SSE log streaming
// ---------------------------------------------------------------------------

/**
 * Open an SSE connection for live log output from a run.
 *
 * Usage:
 *   const es = streamLogs(runId);
 *   es.onmessage = (e) => console.log(e.data);
 *   es.addEventListener("status", (e) => console.log("Status:", e.data));
 *   // Close when done:
 *   es.close();
 */
export function streamLogs(runId: string): EventSource {
  return new EventSource(`${BASE}/api/runs/${encodeURIComponent(runId)}/logs`);
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/** Fetch the zarr manifest for a completed run. */
export async function fetchRunManifest(runId: string): Promise<unknown> {
  const res = await fetch(
    `${BASE}/api/runs/${encodeURIComponent(runId)}/manifest`
  );
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`GET /api/runs/${runId}/manifest: ${res.status}`);
  return res.json();
}
