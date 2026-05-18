"""End-to-end dashboard integration tests.

Exercises the full 100×100×500 simulation pipeline in two ways:

  1. ``test_simulation_runs_end_to_end``
       Calls ``build_model()`` directly — no HTTP layer — and verifies that
       every expected zarr output is written to disk.

  2. ``test_dashboard_api_run_lifecycle``
       Starts a real uvicorn server in a background thread, POSTs a run via
       ``POST /api/runs``, polls ``GET /api/runs/{run_id}`` until the status
       transitions to COMPLETE, then checks the manifest endpoint is usable.

Both tests are tagged ``@pytest.mark.slow`` because a 100×100×500 cube can
take several minutes.  All other tests continue to run in the normal suite:

    pytest tests/ -m "not slow"          # fast suite only
    pytest tests/test_dashboard_simulation.py -v -s   # integration tests
"""
from __future__ import annotations

import json
import pathlib
import socket
import sys
import threading
import time
import uuid

import httpx
import pytest

# ---------------------------------------------------------------------------
# Shared simulation config — 100×100×500, trimmed for speed
# ---------------------------------------------------------------------------

SMOKE_CONFIG: dict = {
    # Geometry
    "cube_shape": [100, 100, 500],
    "incident_angles": [7, 15, 24],
    "digi": 4,
    "infill_factor": 10,
    # Stratigraphy
    "initial_layer_stdev": [7.0, 25.0],
    "thickness_min": 2,
    "thickness_max": 12,
    "seabed_min_depth": [20, 50],
    # Seismic
    "signal_to_noise_ratio_db": [7.5, 12.5, 17.5],
    "bandwidth_low": [3.0, 6.0],
    "bandwidth_high": [20.0, 35.0],
    "bandwidth_ord": 4,
    "dip_factor_max": 2.0,
    # Faults — keep small to stay within the volume
    "min_number_faults": 1,
    "max_number_faults": 3,
    "pad_samples": 10,
    # Closures
    "max_column_height": [150.0, 150.0],
    "closure_types": ["simple"],
    "min_closure_voxels_simple": 100,
    "min_closure_voxels_faulted": 500,
    "min_closure_voxels_onlap": 100,
    # Rock properties
    "sand_layer_thickness": 2,
    "sand_layer_fraction": {"min": 0.05, "max": 0.25},
    # Feature flags — disable everything expensive or non-deterministic
    "extra_qc_plots": False,
    "verbose": False,
    "partial_voxels": True,
    "variable_shale_ng": False,
    "basin_floor_fans": False,
    "include_channels": False,
    "include_salt": False,
    "broadband_qc_volume": False,
    "model_qc_volumes": False,
    "multiprocess_bp": False,
    "model_store_in_memory": False,
    # Keep outputs so post-run assertions can inspect them
    "cleanup_intermediates": False,
}

# Maximum seconds to wait for the simulation subprocess to finish via the API
SIMULATION_TIMEOUT_S = 900  # 15 min


# ---------------------------------------------------------------------------
# Test 1 — direct build_model() smoke test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_simulation_runs_end_to_end(tmp_path: pathlib.Path) -> None:
    """Full 100×100×500 pipeline via build_model(); verifies zarr output tree."""
    from main import build_model  # noqa: PLC0415

    config = {
        **SMOKE_CONFIG,
        "project": "dashboard_smoke_direct",
        "project_folder": str(tmp_path / "project"),
        "work_folder": str(tmp_path / "work"),
    }
    config_path = tmp_path / "smoke_config.json"
    config_path.write_text(json.dumps(config))

    run_id = str(uuid.uuid4())
    output_folder = build_model(str(config_path), run_id, seed=42)

    assert output_folder is not None, "build_model() returned None — check logs"

    out = pathlib.Path(output_folder)
    assert out.exists(), f"Output folder not created: {output_folder}"

    # Core geology zarr stores
    assert (out / "geology" / "geologic_age.zarr").exists(), \
        "geologic_age.zarr missing"
    assert (out / "geology" / "faulted_lithology.zarr").exists(), \
        "faulted_lithology.zarr missing"

    # Horizon depth maps
    assert (out / "horizons" / "depth_maps.zarr").exists(), \
        "depth_maps.zarr missing"

    # At least one seismic angle stack was written
    seismic_zarrs = list(out.glob("seismic/*.zarr"))
    assert seismic_zarrs, f"No seismic zarr stores found under {out}/seismic/"


# ---------------------------------------------------------------------------
# Helpers for the live-server fixture
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Return an ephemeral port number that is free at call-time."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(base_url: str, timeout: float = 15.0) -> None:
    """Block until the server responds to GET /api/runs or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get(f"{base_url}/api/runs", timeout=2).raise_for_status()
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.25)
    raise RuntimeError(f"Server at {base_url} did not become ready in {timeout}s")


# ---------------------------------------------------------------------------
# Live-server fixture — real uvicorn, isolated DB, background thread
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_server(tmp_path_factory: pytest.TempPathFactory):
    """
    Start a real uvicorn server in a background daemon thread.

    * Uses a temporary SQLite DB so it doesn't touch ``runs.db``.
    * Picks a free OS port to avoid conflicts.
    * Yields the base URL string; shuts down cleanly after the module finishes.
    """
    import uvicorn  # noqa: PLC0415
    import api.run_manager as rm  # noqa: PLC0415

    db_path = tmp_path_factory.mktemp("live_server_db") / "test_runs.db"

    # Redirect the module-level DB path before the app starts
    original_db = rm.DB_PATH
    rm.DB_PATH = db_path

    port = _free_port()
    config = uvicorn.Config(
        "api.main:app",
        host="127.0.0.1",
        port=port,
        loop="asyncio",
        log_level="warning",
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    _wait_for_server(base_url)

    yield base_url

    # Graceful shutdown
    server.should_exit = True
    thread.join(timeout=10)

    # Restore module state
    rm.DB_PATH = original_db


# ---------------------------------------------------------------------------
# Test 2 — full API lifecycle: POST → poll GET → COMPLETE → manifest
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_dashboard_api_run_lifecycle(
    live_server: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """
    Submit a 100×100×500 run through the dashboard API and verify it completes.

    Steps
    -----
    1. POST /api/runs  →  {run_id, status: RUNNING}
    2. Poll GET /api/runs/{run_id} every 10 s until COMPLETE or FAILED.
    3. Assert status is COMPLETE.
    4. Call GET /api/runs/{run_id}/manifest and assert it returns a JSON object.
    """
    tmp = tmp_path_factory.mktemp("api_sim")
    base = live_server

    config = {
        **SMOKE_CONFIG,
        "project": "dashboard_smoke_api",
        "project_folder": str(tmp / "project"),
        "work_folder": str(tmp / "work"),
    }

    # ── Step 1: submit ──────────────────────────────────────────────────────
    post_resp = httpx.post(f"{base}/api/runs", json=config, timeout=30)
    assert post_resp.status_code == 200, (
        f"POST /api/runs failed ({post_resp.status_code}): {post_resp.text}"
    )
    submitted = post_resp.json()
    assert "run_id" in submitted, f"No run_id in response: {submitted}"
    assert submitted["status"] == "RUNNING", f"Unexpected initial status: {submitted}"
    run_id = submitted["run_id"]

    # ── Step 2: poll until terminal status ─────────────────────────────────
    deadline = time.monotonic() + SIMULATION_TIMEOUT_S
    status = "RUNNING"
    while time.monotonic() < deadline:
        poll = httpx.get(f"{base}/api/runs/{run_id}", timeout=10)
        assert poll.status_code == 200, f"GET /api/runs/{run_id} → {poll.status_code}"
        status = poll.json()["status"]
        if status in ("COMPLETE", "FAILED"):
            break
        time.sleep(10)
    else:
        pytest.fail(
            f"Simulation did not finish within {SIMULATION_TIMEOUT_S}s "
            f"(last status: {status})"
        )

    # ── Step 3: assert success ──────────────────────────────────────────────
    assert status == "COMPLETE", (
        f"Simulation ended with status '{status}' — check server logs for errors"
    )

    # ── Step 4: manifest endpoint ───────────────────────────────────────────
    manifest_resp = httpx.get(f"{base}/api/runs/{run_id}/manifest", timeout=30)
    assert manifest_resp.status_code == 200, (
        f"GET /api/runs/{run_id}/manifest → {manifest_resp.status_code}: "
        f"{manifest_resp.text}"
    )
    manifest = manifest_resp.json()
    assert isinstance(manifest, dict), (
        f"Expected manifest to be a JSON object, got: {type(manifest)}"
    )
