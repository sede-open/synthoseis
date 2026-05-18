"""FastAPI application for the Synthoseis simulation launcher.

Routes:
  GET  /api/models                   - List available rock-physics models
  GET  /api/browse-directory         - Open native OS folder-picker dialog
  GET  /api/manifest                 - Combined manifest for all COMPLETE runs
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
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
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
    # Strip the conventional "rpm_" prefix so the returned names match the
    # values expected by select_rpm (e.g. "rpm_example.py" -> "example").
    stems = [p.stem for p in sorted(rpm_dir.glob("*.py")) if p.stem not in exclude]
    return [s[len("rpm_"):] if s.startswith("rpm_") else s for s in stems]


@app.get("/api/browse-directory", response_model=dict)
def browse_directory(initial_dir: str | None = None) -> dict:
    """Open a native OS folder-picker dialog and return the selected path.

    Uses platform-native subprocesses — no Python UI toolkit required, so it
    works correctly from a FastAPI thread on macOS (tkinter crashes with
    NSInternalInconsistencyException when called off the main thread):

      macOS  : osascript (AppleScript) — always available, no thread restriction
      Linux  : zenity (GNOME) or kdialog (KDE), whichever is on PATH
      Windows: PowerShell FolderBrowserDialog

    Returns {"path": "<absolute path>"} or {"path": null} if the user cancels.
    Raises HTTP 501 if no supported dialog tool is found.
    """
    import shutil

    start = str(pathlib.Path(initial_dir).expanduser().resolve()) if initial_dir else str(pathlib.Path.home())

    chosen: str | None = None

    if sys.platform == "darwin":
        # AppleScript: runs in its own process, no Cocoa main-thread requirement.
        script = (
            f'POSIX path of (choose folder '
            f'with prompt "Select folder" '
            f'default location POSIX file "{start}")'
        )
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            chosen = result.stdout.strip().rstrip("/") or None

    elif sys.platform.startswith("linux"):
        if shutil.which("zenity"):
            result = subprocess.run(
                ["zenity", "--file-selection", "--directory",
                 "--title=Select folder", f"--filename={start}/"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                chosen = result.stdout.strip() or None
        elif shutil.which("kdialog"):
            result = subprocess.run(
                ["kdialog", "--getexistingdirectory", start],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                chosen = result.stdout.strip() or None
        else:
            raise HTTPException(
                status_code=501,
                detail="No folder picker found. Install zenity (GNOME) or kdialog (KDE).",
            )

    elif sys.platform == "win32":
        ps_script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
            f"$f.SelectedPath = '{start}'; "
            "if ($f.ShowDialog() -eq 'OK') {{ Write-Output $f.SelectedPath }}"
        )
        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            chosen = result.stdout.strip() or None

    else:
        raise HTTPException(
            status_code=501,
            detail=f"Folder picker not supported on platform: {sys.platform}",
        )

    return {"path": chosen}


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

@app.get("/api/manifest")
def api_manifest(project_folder: str | None = None) -> list:
    """Return the combined manifest for completed runs.

    If *project_folder* is supplied (a server-side path, ``~`` is expanded),
    that folder is scanned directly via generate_manifest and the result is
    returned immediately (manifest.json is always regenerated so the viewer
    reflects the current state on disk).

    Without *project_folder* the endpoint falls back to the runs DB: it
    collects unique output_folder values from COMPLETE runs and merges their
    manifests (cached via overwrite=False).
    """
    import os as _os
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT))
    from scripts.generate_manifest import generate_manifest

    if project_folder:
        import re as _re
        folder = str(pathlib.Path(_os.path.expanduser(project_folder)).resolve())
        if not pathlib.Path(folder).exists():
            raise HTTPException(
                status_code=404,
                detail=f"project_folder does not exist: {folder}",
            )
        # Auto-detect: if the user passed a seismic__* run folder directly,
        # step up to the parent so generate_manifest can scan all siblings.
        _run_folder_re = _re.compile(r"^seismic__(\d{8}|\d{4}_\d{4})_.+$")
        if _run_folder_re.match(pathlib.Path(folder).name):
            folder = str(pathlib.Path(folder).parent)
        try:
            return generate_manifest(folder, overwrite=True)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # --- DB fallback ---------------------------------------------------------
    runs = list_runs()
    seen_folders: set[str] = set()
    entries: list = []
    for run in runs:
        if run.get("status") != "COMPLETE":
            continue
        raw_folder = run.get("output_folder")
        if not raw_folder:
            continue
        folder = str(pathlib.Path(_os.path.expanduser(raw_folder)).resolve())
        if folder in seen_folders or not pathlib.Path(folder).exists():
            continue
        seen_folders.add(folder)
        try:
            entries.extend(generate_manifest(folder, overwrite=False))
        except Exception as exc:
            print(f"WARNING: could not generate manifest for {folder}: {exc}")
    return entries


@app.get("/api/runs/{run_id}/manifest")
def api_run_manifest(run_id: str) -> list:
    """Return the zarr manifest entries for a single completed run's output folder."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    raw_folder = run.get("output_folder")
    if not raw_folder:
        raise HTTPException(status_code=404, detail="output_folder not set for this run")

    import os as _os
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT))
    from scripts.generate_manifest import generate_manifest

    folder = str(pathlib.Path(_os.path.expanduser(raw_folder)).resolve())
    if not pathlib.Path(folder).exists():
        raise HTTPException(status_code=404, detail=f"output_folder does not exist: {folder}")
    try:
        return generate_manifest(folder, overwrite=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Zarr file server — serves arbitrary filesystem paths for the browser zarr client
# ---------------------------------------------------------------------------

@app.get("/api/zarr/{file_path:path}")
def serve_zarr_file(file_path: str) -> FileResponse:
    """Serve zarr chunk files and metadata from the filesystem.

    The browser-side zarrita client (FetchStore) fetches zarr.json metadata and
    chunk files via HTTP. This endpoint maps URL path segments back to absolute
    filesystem paths so those files can be reached.

    Example:
      GET /api/zarr/Users/alice/synthoseis_output/run/store.zarr/zarr.json
      → serves /Users/alice/synthoseis_output/run/store.zarr/zarr.json
    """
    import mimetypes as _mimetypes
    full_path = pathlib.Path("/" + file_path)
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail=f"Not found: {full_path}")
    media_type = _mimetypes.guess_type(str(full_path))[0] or "application/octet-stream"
    return FileResponse(str(full_path), media_type=media_type)


# ---------------------------------------------------------------------------
# Static frontend (production only)
# ---------------------------------------------------------------------------

_dist = REPO_ROOT / "webapp" / "dist"
if (_dist / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")
