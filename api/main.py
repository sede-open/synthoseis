"""FastAPI application for the Synthoseis simulation launcher.

Routes:
  GET  /api/models                   - List available rock-physics models
  GET  /api/runs                     - List all runs
  POST /api/runs                     - Submit a new simulation run
  GET  /api/runs/{run_id}            - Get a single run record
  GET  /api/runs/{run_id}/logs       - Stream live logs via SSE
  GET  /api/runs/{run_id}/manifest   - Return the run's zarr manifest JSON
  DELETE /api/runs/{run_id}          - Cancel a running simulation

Static files (production):
  If webapp/dist/index.html exists, the built frontend is served at /.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import subprocess
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.config import SimulationConfig
from api.run_manager import (
    cancel_run,
    get_run,
    init_db,
    insert_run,
    launch_run,
    list_runs,
    mark_orphans_failed,
    stream_logs,
)

REPO_ROOT = pathlib.Path(__file__).parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: clean up orphaned RUNNING records on startup."""
    init_db()
    mark_orphans_failed()
    yield
    # Shutdown: no explicit cleanup needed


app = FastAPI(title="Synthoseis Launcher API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Rock-physics model discovery
# ---------------------------------------------------------------------------

@app.get("/api/models", response_model=list[str])
def get_models() -> list[str]:
    """Return a list of available rock-physics model names."""
    rpm_dir = REPO_ROOT / "rockphysics"
    exclude = {"__init__", "RockPropertyModels"}
    return [p.stem for p in sorted(rpm_dir.glob("*.py")) if p.stem not in exclude]


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

@app.get("/api/runs")
def api_list_runs() -> list[dict]:
    return list_runs()


@app.get("/api/runs/{run_id}")
def api_get_run(run_id: str) -> dict:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run


@app.post("/api/runs")
async def api_create_run(config: SimulationConfig) -> dict:
    """Validate config, persist a RUNNING record, and launch the subprocess."""
    run_id = config.run_id or str(uuid.uuid4())

    # Write config JSON into the project folder
    project_folder = pathlib.Path(config.project_folder)
    project_folder.mkdir(parents=True, exist_ok=True)
    config_path = project_folder / f"config_{run_id}.json"
    config_dict = config.to_config_json()
    config_dict["run_id"] = run_id
    config_path.write_text(json.dumps(config_dict, indent=2))

    # Insert RUNNING record
    insert_run(run_id, json.dumps(config_dict))

    # Fire-and-forget — create_task schedules on the running event loop.
    # do not await so the HTTP response returns immediately.
    asyncio.create_task(launch_run(run_id, str(config_path), REPO_ROOT))

    return {"run_id": run_id, "status": "RUNNING"}


@app.delete("/api/runs/{run_id}")
def api_cancel_run(run_id: str) -> dict:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    cancel_run(run_id)
    return {"run_id": run_id, "status": "FAILED"}


# ---------------------------------------------------------------------------
# SSE log streaming
# ---------------------------------------------------------------------------

@app.get("/api/runs/{run_id}/logs")
async def api_run_logs(run_id: str) -> StreamingResponse:
    """Stream live stdout from the simulation as Server-Sent Events."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return StreamingResponse(
        stream_logs(run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@app.get("/api/runs/{run_id}/manifest")
def api_run_manifest(run_id: str) -> dict:
    """Return the zarr manifest for the completed run's output folder."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    output_folder = run.get("output_folder")
    if not output_folder:
        raise HTTPException(status_code=404, detail="output_folder not set for this run")

    result = subprocess.run(
        ["python", "scripts/generate_manifest.py", output_folder],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Static frontend (production only)
# ---------------------------------------------------------------------------

_dist = REPO_ROOT / "webapp" / "dist"
if (_dist / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")
