"""Failing tests for FastAPI routes (api/main.py + api/run_manager.py).

Run: pytest tests/test_api_routes.py -v
These should FAIL until api/ is fully implemented.
"""
import asyncio
import json
import os
import pathlib
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_CONFIG = {
    "project": "example",
    "project_folder": "/tmp/synthoseis_test",
    "work_folder": "/tmp",
    "cube_shape": [300, 300, 1250],
    "incident_angles": [7, 15, 24],
    "digi": 4,
    "infill_factor": 10,
    "initial_layer_stdev": [7.0, 25.0],
    "thickness_min": 2,
    "thickness_max": 12,
    "seabed_min_depth": [20, 50],
    "signal_to_noise_ratio_db": [7.5, 12.5, 17.5],
    "bandwidth_low": [3.0, 6.0],
    "bandwidth_high": [20.0, 35.0],
    "bandwidth_ord": 4,
    "dip_factor_max": 2.0,
    "min_number_faults": 1,
    "max_number_faults": 6,
    "pad_samples": 10,
    "max_column_height": [150.0, 150.0],
    "closure_types": ["simple", "faulted", "onlap"],
    "min_closure_voxels_simple": 500,
    "min_closure_voxels_faulted": 2500,
    "min_closure_voxels_onlap": 500,
    "sand_layer_thickness": 2,
    "sand_layer_fraction": {"min": 0.05, "max": 0.25},
    "extra_qc_plots": True,
    "verbose": True,
    "partial_voxels": True,
    "variable_shale_ng": False,
    "basin_floor_fans": False,
    "include_channels": False,
    "include_salt": True,
    "write_to_hdf": False,
    "broadband_qc_volume": False,
    "model_qc_volumes": True,
    "multiprocess_bp": True,
    "model_store_in_memory": False,
    "cleanup_intermediates": True,
}


@pytest.fixture()
def test_db(tmp_path, monkeypatch):
    """Redirect the SQLite database to a temp file for isolation."""
    db_path = tmp_path / "runs_test.db"
    monkeypatch.setenv("SYNTHOSEIS_DB_PATH", str(db_path))
    # Patch the module-level DB_PATH in run_manager if already imported
    try:
        import api.run_manager as rm
        monkeypatch.setattr(rm, "DB_PATH", db_path)
        rm.init_db()
    except ImportError:
        pass
    yield db_path


@pytest.fixture()
def client(test_db, monkeypatch):
    """Synchronous TestClient that skips the real lifespan startup side-effects."""
    from fastapi.testclient import TestClient

    import api.run_manager as rm
    monkeypatch.setattr(rm, "DB_PATH", test_db)
    rm.init_db()
    rm._processes.clear()

    # Patch mark_orphans_failed so lifespan doesn't interfere
    monkeypatch.setattr(rm, "mark_orphans_failed", lambda: None)

    from api.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /api/models
# ---------------------------------------------------------------------------

def test_get_models_returns_list(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert all(isinstance(m, str) for m in data)


def test_get_models_includes_example(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    # "rpm_example" RPM should be discoverable (file is rockphysics/rpm_example.py)
    models = resp.json()
    assert any("example" in m for m in models), f"No example model found in {models}"


# ---------------------------------------------------------------------------
# GET /api/runs (empty)
# ---------------------------------------------------------------------------

def test_list_runs_empty_on_fresh_db(client):
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# POST /api/runs
# ---------------------------------------------------------------------------

def _mock_launch(monkeypatch):
    """Patch launch_run so no real subprocess is spawned."""
    import api.run_manager as rm

    async def fake_launch(run_id, config_path, repo_root):
        # Immediately mark complete so tests don't hang
        import sqlite3
        conn = sqlite3.connect(rm.DB_PATH)
        conn.execute(
            "UPDATE runs SET status='COMPLETE', ended_at=datetime('now') WHERE run_id=?",
            (run_id,),
        )
        conn.commit()
        conn.close()

    monkeypatch.setattr(rm, "launch_run", fake_launch)


def test_post_runs_valid_config_returns_run_id(client, monkeypatch, tmp_path):
    """POST /api/runs with valid config → {run_id, status: RUNNING}."""
    import api.run_manager as rm
    monkeypatch.setattr(rm, "launch_run", AsyncMock())

    cfg = {**VALID_CONFIG, "project_folder": str(tmp_path)}
    resp = client.post("/api/runs", json=cfg)
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    assert data["status"] == "RUNNING"


def test_post_runs_invalid_config_returns_422(client):
    """POST /api/runs with invalid config → 422."""
    bad = {**VALID_CONFIG, "closure_types": []}
    resp = client.post("/api/runs", json=bad)
    assert resp.status_code == 422


def test_post_runs_thickness_invalid_returns_422(client):
    bad = {**VALID_CONFIG, "thickness_min": 10, "thickness_max": 5}
    resp = client.post("/api/runs", json=bad)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/runs/{run_id}
# ---------------------------------------------------------------------------

def test_get_run_by_id(client, monkeypatch, tmp_path):
    import api.run_manager as rm
    monkeypatch.setattr(rm, "launch_run", AsyncMock())

    cfg = {**VALID_CONFIG, "project_folder": str(tmp_path)}
    post_resp = client.post("/api/runs", json=cfg)
    run_id = post_resp.json()["run_id"]

    get_resp = client.get(f"/api/runs/{run_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["run_id"] == run_id


def test_get_run_nonexistent_returns_404(client):
    resp = client.get("/api/runs/nonexistent-run-id")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/runs/{run_id}
# ---------------------------------------------------------------------------

def test_delete_running_run_returns_200(client, monkeypatch, tmp_path):
    import api.run_manager as rm
    monkeypatch.setattr(rm, "launch_run", AsyncMock())

    # Create a run
    cfg = {**VALID_CONFIG, "project_folder": str(tmp_path)}
    post_resp = client.post("/api/runs", json=cfg)
    run_id = post_resp.json()["run_id"]

    # Mock cancel so no real SIGTERM
    monkeypatch.setattr(rm, "cancel_run", lambda rid: None)

    resp = client.delete(f"/api/runs/{run_id}")
    assert resp.status_code == 200


def test_delete_nonexistent_run_returns_404(client, monkeypatch):
    import api.run_manager as rm
    monkeypatch.setattr(rm, "cancel_run", lambda rid: None)

    resp = client.delete("/api/runs/nonexistent-run-id")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Orphan cleanup on startup
# ---------------------------------------------------------------------------

def test_orphan_cleanup_marks_running_as_failed(test_db, monkeypatch):
    """RUNNING rows in DB become FAILED when mark_orphans_failed() is called."""
    import sqlite3
    import api.run_manager as rm

    monkeypatch.setattr(rm, "DB_PATH", test_db)
    rm.init_db()

    # Insert a fake RUNNING row
    conn = sqlite3.connect(test_db)
    conn.execute(
        "INSERT INTO runs (run_id, status, config_json, started_at) VALUES (?, ?, ?, datetime('now'))",
        ("orphan-run", "RUNNING", "{}"),
    )
    conn.commit()
    conn.close()

    rm.mark_orphans_failed()

    conn = sqlite3.connect(test_db)
    row = conn.execute(
        "SELECT status FROM runs WHERE run_id=?", ("orphan-run",)
    ).fetchone()
    conn.close()
    assert row[0] == "FAILED"


# ---------------------------------------------------------------------------
# SSE — content-type and heartbeat
# ---------------------------------------------------------------------------

def test_logs_endpoint_returns_event_stream_content_type(client, monkeypatch, tmp_path):
    """GET /api/runs/{run_id}/logs → content-type: text/event-stream."""
    import api.run_manager as rm
    monkeypatch.setattr(rm, "launch_run", AsyncMock())

    cfg = {**VALID_CONFIG, "project_folder": str(tmp_path)}
    post_resp = client.post("/api/runs", json=cfg)
    run_id = post_resp.json()["run_id"]

    async def fake_stream(run_id):
        yield ": heartbeat\n\n"
        yield "event: status\ndata: COMPLETE\n\n"

    monkeypatch.setattr("api.main.stream_logs", fake_stream)

    with client.stream("GET", f"/api/runs/{run_id}/logs") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


def test_logs_endpoint_sends_heartbeat(client, monkeypatch, tmp_path):
    """SSE stream includes a heartbeat comment line."""
    import api.run_manager as rm
    monkeypatch.setattr(rm, "launch_run", AsyncMock())

    cfg = {**VALID_CONFIG, "project_folder": str(tmp_path)}
    post_resp = client.post("/api/runs", json=cfg)
    run_id = post_resp.json()["run_id"]

    async def fake_stream(run_id):
        yield ": heartbeat\n\n"
        yield "event: status\ndata: COMPLETE\n\n"

    # Must patch api.main.stream_logs since main.py imports it directly
    monkeypatch.setattr("api.main.stream_logs", fake_stream)

    with client.stream("GET", f"/api/runs/{run_id}/logs") as resp:
        content = b""
        for chunk in resp.iter_bytes():
            content += chunk
        assert b"heartbeat" in content


def test_logs_endpoint_nonexistent_run_returns_404(client):
    resp = client.get("/api/runs/nonexistent/logs")
    assert resp.status_code == 404
