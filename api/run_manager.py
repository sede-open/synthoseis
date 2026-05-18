"""SQLite CRUD, subprocess management, and SSE log streaming for simulation runs.

NOTE: This module uses a module-level `_processes` dict to track live subprocesses.
Multi-worker deployments (workers > 1) would split this registry across OS processes,
breaking DELETE /api/runs/{id} and SSE log streaming for cross-worker runs.
Always run with workers=1.
"""
from __future__ import annotations

import asyncio
import os
import signal
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

# DB_PATH is module-level so tests can monkeypatch it.
DB_PATH: Path = Path(__file__).parent.parent / "runs.db"

# In-memory registry of live subprocesses (run_id → Process).
# Must remain a single-worker deployment — see module docstring.
_processes: dict[str, asyncio.subprocess.Process] = {}

# Heartbeat interval for SSE streams (seconds)
HEARTBEAT_INTERVAL = 15

# num_runs is always 1 per submission — not user-configurable
NUM_RUNS = 1


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the runs table if it doesn't already exist."""
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id       TEXT PRIMARY KEY,
            status       TEXT NOT NULL,
            config_json  TEXT,
            started_at   TEXT,
            ended_at     TEXT,
            output_folder TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def mark_orphans_failed() -> None:
    """Mark any RUNNING rows as FAILED (called on app startup to clean up orphans)."""
    conn = _connect()
    conn.execute(
        "UPDATE runs SET status='FAILED', ended_at=datetime('now') WHERE status='RUNNING'"
    )
    conn.commit()
    conn.close()


def insert_run(run_id: str, config_json: str) -> None:
    conn = _connect()
    conn.execute(
        "INSERT INTO runs (run_id, status, config_json, started_at) VALUES (?, 'RUNNING', ?, datetime('now'))",
        (run_id, config_json),
    )
    conn.commit()
    conn.close()


def get_run(run_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def list_runs() -> list[dict]:
    conn = _connect()
    rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _update_status(run_id: str, status: str, output_folder: str | None = None) -> None:
    conn = _connect()
    if status in ("COMPLETE", "FAILED"):
        if output_folder is not None:
            conn.execute(
                "UPDATE runs SET status=?, ended_at=datetime('now'), output_folder=? WHERE run_id=?",
                (status, output_folder, run_id),
            )
        else:
            conn.execute(
                "UPDATE runs SET status=?, ended_at=datetime('now') WHERE run_id=?",
                (status, run_id),
            )
    else:
        conn.execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))
    conn.commit()
    conn.close()


async def launch_run(run_id: str, config_json_path: str, repo_root: Path) -> None:
    """Launch the Synthoseis simulation as an async subprocess.

    Args:
        run_id: Unique identifier for this run (also passed as --run_id).
        config_json_path: Absolute path to the written config JSON file.
        repo_root: Repository root — used as cwd so `python main.py` resolves.
    """
    import json as _json
    # Derive the expected output folder from the config so we can persist it
    # to runs.db once the run completes (needed by GET /api/runs/{id}/manifest).
    try:
        _cfg = _json.loads(pathlib.Path(config_json_path).read_text())
        _project_folder = _cfg.get("project_folder", "")
        _run_id_suffix = f"_{run_id}" if run_id else ""
        # Parameters.py names the subfolder seismic__{datestamp}_{runid};
        # we don't know the datestamp yet, so we store the project_folder root
        # and let generate_manifest.py discover the exact subfolder.
        _output_folder = _project_folder
    except Exception:
        _output_folder = None

    process = await asyncio.create_subprocess_exec(
        "python",
        "main.py",
        "--config",
        str(config_json_path),
        "--run_id",
        run_id,
        cwd=str(repo_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={
            **os.environ,
            # log.py falls back to /scratch when GC_LOG_DIR is unset, which is
            # read-only on macOS. Point it at the project folder instead so the
            # geocrawler.log file lands alongside the run output.
            "GC_LOG_DIR": _output_folder or str(repo_root),
        },
    )
    _processes[run_id] = process

    # Wait for the process and update status + output_folder on completion
    return_code = await process.wait()
    _processes.pop(run_id, None)
    final_status = "COMPLETE" if return_code == 0 else "FAILED"
    _update_status(run_id, final_status, output_folder=_output_folder)


async def stream_logs(run_id: str) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE-formatted strings for a run's stdout.

    Sends a heartbeat comment every HEARTBEAT_INTERVAL seconds while waiting
    for the next line of output.
    """
    process = _processes.get(run_id)

    if process is None or process.stdout is None:
        # Process already finished or was never started — emit a quick status
        run = get_run(run_id)
        final_status = run["status"] if run else "FAILED"
        yield f"event: status\ndata: {final_status}\n\n"
        return

    async def _read_line() -> bytes | None:
        try:
            return await asyncio.wait_for(
                process.stdout.readline(), timeout=HEARTBEAT_INTERVAL
            )
        except asyncio.TimeoutError:
            return None

    while True:
        line = await _read_line()
        if line is None:
            # Timeout — send heartbeat
            yield ": heartbeat\n\n"
            continue
        if line == b"":
            # EOF — process finished
            break
        yield f"data: {line.decode(errors='replace').rstrip()}\n\n"

    # Process finished — report final status
    final_status = "COMPLETE" if (process.returncode == 0) else "FAILED"
    yield f"event: status\ndata: {final_status}\n\n"


def cancel_run(run_id: str) -> None:
    """Send SIGTERM to a running process and mark it FAILED in the DB."""
    process = _processes.get(run_id)
    if process is not None:
        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        _processes.pop(run_id, None)
    _update_status(run_id, "FAILED")
