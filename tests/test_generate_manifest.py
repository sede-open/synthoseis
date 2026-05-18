"""Failing tests for scripts/generate_manifest.py.

Tests use real zarr v3 stores created via output_writer.write_volume_to_zarr()
and real SQLite files. All tests should fail until generate_manifest.py exists.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from datagenerator.output_writer import write_volume_to_zarr

# ---------------------------------------------------------------------------
# Helpers to import the script as a module regardless of sys.path
# ---------------------------------------------------------------------------

def _load_generate_manifest():
    """Dynamically load scripts/generate_manifest.py as a module."""
    script_path = Path(__file__).parents[1] / "scripts" / "generate_manifest.py"
    spec = importlib.util.spec_from_file_location("generate_manifest", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_params_db(run_dir: Path, params: dict) -> None:
    """Create a parameters.db SQLite file inside *run_dir*."""
    db_path = run_dir / "parameters.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS model_parameters (key TEXT, value TEXT)"
    )
    conn.executemany(
        "INSERT INTO model_parameters VALUES (?, ?)",
        [(k, v) for k, v in params.items()],
    )
    conn.commit()
    conn.close()


DATESTAMP = "20260517"
SHAPE = (10, 10, 20)


def _make_run_dir(project_folder: Path, run_id: str = "test_run") -> Path:
    """Return a freshly-created run subdirectory."""
    run_dir = project_folder / f"seismic__{DATESTAMP}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _make_seismic_store(run_dir: Path, angle: int = 7) -> Path:
    store_path = (
        run_dir
        / "seismic"
        / f"seismicCubes_RFC_{angle}_degrees_{DATESTAMP}.zarr"
    )
    store_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.rand(*SHAPE).astype("float32")
    write_volume_to_zarr(
        arr,
        str(store_path),
        name="amplitude",
        dims=("inline", "crossline", "time"),
        attrs={"angle_deg": angle, "sample_rate_ms": 4},
    )
    return store_path


def _make_geology_store(run_dir: Path, name: str = "geologic_age", var: str = "age") -> Path:
    store_path = run_dir / "geology" / f"{name}.zarr"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.rand(*SHAPE).astype("float32")
    write_volume_to_zarr(
        arr, str(store_path), name=var, dims=("inline", "crossline", "time")
    )
    return store_path


def _make_horizons_store(run_dir: Path) -> Path:
    store_path = run_dir / "horizons" / "depth_maps.zarr"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.rand(*SHAPE).astype("float32")
    write_volume_to_zarr(
        arr,
        str(store_path),
        name="depth",
        dims=("inline", "crossline", "horizon"),
    )
    return store_path


def _make_closure_store(run_dir: Path, fname: str = "gas") -> Path:
    store_path = run_dir / "closures" / f"{fname}.zarr"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(SHAPE, dtype="uint8")
    write_volume_to_zarr(
        arr, str(store_path), name="label", dims=("inline", "crossline", "time")
    )
    return store_path


def _make_qc_store(run_dir: Path, fname: str = "fault_segments") -> Path:
    store_path = run_dir / f"{fname}_{DATESTAMP}.zarr"
    arr = np.zeros(SHAPE, dtype="float32")
    write_volume_to_zarr(
        arr, str(store_path), name="data", dims=("inline", "crossline", "time")
    )
    return store_path


# ===========================================================================
# Tests
# ===========================================================================


class TestSchemaValidation:
    """Manifest output satisfies the expected JSON schema."""

    def test_manifest_schema_fields(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir)
        _make_closure_store(run_dir)
        _write_params_db(run_dir, {"cube_shape": str(list(SHAPE))})

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        assert isinstance(manifest, list), "Manifest must be a list"
        assert len(manifest) == 1
        entry = manifest[0]

        for field in ("run_id", "folder", "datestamp", "cube_shape", "volumes", "parameters"):
            assert field in entry, f"Missing field: {field}"

        assert isinstance(entry["run_id"], str)
        assert isinstance(entry["folder"], str)
        assert isinstance(entry["datestamp"], str)
        assert isinstance(entry["cube_shape"], list)
        assert isinstance(entry["volumes"], list)
        assert isinstance(entry["parameters"], dict)

    def test_manifest_run_id_and_folder(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "my_run")
        _make_seismic_store(run_dir)
        _write_params_db(run_dir, {})

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        entry = manifest[0]
        assert entry["folder"] == f"seismic__{DATESTAMP}_my_run"
        assert entry["run_id"] == "my_run"

    def test_manifest_datestamp_extracted(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        assert manifest[0]["datestamp"] == DATESTAMP

    def test_manifest_cube_shape_from_volumes(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        assert manifest[0]["cube_shape"] == list(SHAPE)

    def test_volume_schema_fields(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))
        vol = manifest[0]["volumes"][0]

        for field in ("name", "store_path", "variable", "group", "shape", "dtype", "dims", "chunks", "compressor", "attrs"):
            assert field in vol, f"Volume missing field: {field}"

    def test_volume_shape_dtype_dims(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))
        vol = manifest[0]["volumes"][0]

        assert vol["shape"] == list(SHAPE)
        assert vol["dtype"] == "float32"
        assert vol["dims"] == ["inline", "crossline", "time"]

    def test_volume_compressor_string(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))
        vol = manifest[0]["volumes"][0]

        # Should be a non-empty string describing the codec
        assert isinstance(vol["compressor"], str)
        assert len(vol["compressor"]) > 0

    def test_manifest_written_to_disk(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir)

        mod = _load_generate_manifest()
        mod.generate_manifest(str(tmp_path))

        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists(), "manifest.json not written to project_folder"
        with open(manifest_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1


class TestVolumeDiscovery:
    """Scanner discovers stores across all four subdirectories."""

    def test_seismic_group_discovered(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        groups = {v["group"] for v in manifest[0]["volumes"]}
        assert "Seismic" in groups

    def test_geology_group_discovered(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_geology_store(run_dir, "geologic_age", "age")

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        groups = {v["group"] for v in manifest[0]["volumes"]}
        assert "Geology" in groups

    def test_horizons_group_discovered(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_horizons_store(run_dir)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        groups = {v["group"] for v in manifest[0]["volumes"]}
        assert "Horizons" in groups

    def test_closures_group_discovered(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_closure_store(run_dir, "gas")

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        groups = {v["group"] for v in manifest[0]["volumes"]}
        assert "Closures" in groups

    def test_all_four_subdirs(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir, 7)
        _make_geology_store(run_dir, "geologic_age", "age")
        _make_geology_store(run_dir, "faulted_lithology", "lithology")
        _make_horizons_store(run_dir)
        _make_closure_store(run_dir, "gas")
        _make_closure_store(run_dir, "oil")

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        groups = {v["group"] for v in manifest[0]["volumes"]}
        assert groups == {"Seismic", "Geology", "Horizons", "Closures"}

    def test_multiple_seismic_angles(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        for angle in (7, 15, 24):
            _make_seismic_store(run_dir, angle)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        seismic_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Seismic"]
        assert len(seismic_vols) == 3

    def test_multiple_closures(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        for name in ("gas", "oil", "brine", "hc_labels"):
            _make_closure_store(run_dir, name)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        closure_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Closures"]
        assert len(closure_vols) == 4


class TestVariableNames:
    """Each subdirectory yields the correct variable name."""

    def test_seismic_variable_amplitude(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir, 7)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        seismic_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Seismic"]
        assert all(v["variable"] == "amplitude" for v in seismic_vols)

    def test_geology_age_variable(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_geology_store(run_dir, "geologic_age", "age")

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        geo_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Geology"]
        assert any(v["variable"] == "age" for v in geo_vols)

    def test_geology_lithology_variable(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_geology_store(run_dir, "faulted_lithology", "lithology")

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        geo_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Geology"]
        assert any(v["variable"] == "lithology" for v in geo_vols)

    def test_horizons_depth_variable(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_horizons_store(run_dir)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        horiz_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Horizons"]
        assert all(v["variable"] == "depth" for v in horiz_vols)

    def test_closures_label_variable(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_closure_store(run_dir, "gas")

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        closure_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Closures"]
        assert all(v["variable"] == "label" for v in closure_vols)


class TestDatestampHandling:
    """Datestamp rules: seismic filenames include stamp; geology/closures/horizons do not."""

    def test_seismic_store_path_contains_datestamp(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir, 7)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        seismic_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Seismic"]
        for vol in seismic_vols:
            assert DATESTAMP in vol["store_path"], (
                f"Expected datestamp in seismic store_path: {vol['store_path']}"
            )

    def test_geology_store_path_no_datestamp(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_geology_store(run_dir, "geologic_age", "age")

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        geo_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Geology"]
        for vol in geo_vols:
            assert DATESTAMP not in vol["store_path"], (
                f"Unexpected datestamp in geology store_path: {vol['store_path']}"
            )

    def test_closures_store_path_no_datestamp(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_closure_store(run_dir, "gas")

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        closure_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Closures"]
        for vol in closure_vols:
            assert DATESTAMP not in vol["store_path"], (
                f"Unexpected datestamp in closures store_path: {vol['store_path']}"
            )

    def test_horizons_store_path_no_datestamp(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_horizons_store(run_dir)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        horiz_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Horizons"]
        for vol in horiz_vols:
            assert DATESTAMP not in vol["store_path"], (
                f"Unexpected datestamp in horizons store_path: {vol['store_path']}"
            )


class TestAttrsExtraction:
    """Attributes are extracted correctly from each store."""

    def test_seismic_angle_deg_attr(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir, 15)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        seismic_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Seismic"]
        assert len(seismic_vols) == 1
        assert seismic_vols[0]["attrs"].get("angle_deg") == 15

    def test_seismic_sample_rate_attr(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir, 7)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        seismic_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Seismic"]
        assert seismic_vols[0]["attrs"].get("sample_rate_ms") == 4

    def test_geology_attrs_empty(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_geology_store(run_dir, "geologic_age", "age")

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        geo_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Geology"]
        assert geo_vols[0]["attrs"] == {}

    def test_horizons_depth_dims(self, tmp_path):
        """depth_maps.zarr has 'horizon' as third dim — must be recorded correctly."""
        run_dir = _make_run_dir(tmp_path)
        _make_horizons_store(run_dir)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        horiz_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Horizons"]
        assert len(horiz_vols) == 1
        assert horiz_vols[0]["dims"] == ["inline", "crossline", "horizon"]


class TestQCVolumes:
    """Root-level QC volumes are scanned and placed in the QC group."""

    def test_qc_volume_discovered(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_qc_store(run_dir, "fault_segments")

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        qc_vols = [v for v in manifest[0]["volumes"] if v["group"] == "QC"]
        assert len(qc_vols) == 1

    def test_qc_volume_variable_is_data(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_qc_store(run_dir, "fault_segments")

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        qc_vols = [v for v in manifest[0]["volumes"] if v["group"] == "QC"]
        assert qc_vols[0]["variable"] == "data"

    def test_qc_store_path_contains_datestamp(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_qc_store(run_dir, "sealed_label")

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        qc_vols = [v for v in manifest[0]["volumes"] if v["group"] == "QC"]
        assert DATESTAMP in qc_vols[0]["store_path"]

    def test_multiple_qc_volumes(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        for name in ("fault_segments", "sealed_label", "fault_plane_throw"):
            _make_qc_store(run_dir, name)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        qc_vols = [v for v in manifest[0]["volumes"] if v["group"] == "QC"]
        assert len(qc_vols) == 3


class TestParametersDb:
    """parameters.db is read and included in the manifest."""

    def test_parameters_from_db(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir)
        _write_params_db(
            run_dir,
            {
                "cube_shape": "[300, 300, 1250]",
                "incident_angles": "[7, 15, 24]",
                "include_salt": "True",
            },
        )

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        params = manifest[0]["parameters"]
        assert params["cube_shape"] == "[300, 300, 1250]"
        assert params["incident_angles"] == "[7, 15, 24]"
        assert params["include_salt"] == "True"

    def test_missing_parameters_db_empty_dict(self, tmp_path):
        """Missing parameters.db → parameters: {}, no crash."""
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir)
        # No parameters.db created

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        assert manifest[0]["parameters"] == {}

    def test_missing_parameters_db_exit_0(self, tmp_path):
        """Missing parameters.db does not raise an exception."""
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir)

        mod = _load_generate_manifest()
        # Should not raise
        manifest = mod.generate_manifest(str(tmp_path))
        assert len(manifest) == 1


class TestEdgeCases:
    """Edge cases and error-recovery behaviour."""

    def test_empty_project_folder(self, tmp_path):
        """No seismic__* dirs → valid manifest with empty list, no crash."""
        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        assert isinstance(manifest, list)
        assert len(manifest) == 0

    def test_empty_manifest_written_to_disk(self, tmp_path):
        """Empty manifest is still written as a JSON array."""
        mod = _load_generate_manifest()
        mod.generate_manifest(str(tmp_path))

        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            data = json.load(f)
        assert data == []

    def test_two_run_folders(self, tmp_path):
        for run_id in ("run_a", "run_b"):
            run_dir = _make_run_dir(tmp_path, run_id)
            _make_seismic_store(run_dir)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        assert len(manifest) == 2
        folders = {e["folder"] for e in manifest}
        assert f"seismic__{DATESTAMP}_run_a" in folders
        assert f"seismic__{DATESTAMP}_run_b" in folders

    def test_subdirectory_empty_no_volumes(self, tmp_path):
        """Subdirectory present but contains no zarr stores → no crash, no volumes for that group."""
        run_dir = _make_run_dir(tmp_path)
        # Create geology/ with a non-zarr file
        (run_dir / "geology").mkdir()
        (run_dir / "geology" / "readme.txt").write_text("empty")
        _make_seismic_store(run_dir)

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        # Should have Seismic volumes but no Geology volumes
        geo_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Geology"]
        assert geo_vols == []

    def test_corrupted_zarr_store_skipped(self, tmp_path):
        """Zarr store missing .zmetadata is skipped; other volumes included."""
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir, 7)

        # Create a "broken" zarr store without .zmetadata
        broken = run_dir / "geology" / "broken.zarr"
        broken.mkdir(parents=True)
        (broken / "array.0.0").write_bytes(b"\x00" * 16)
        # No .zmetadata → open_consolidated will fail

        mod = _load_generate_manifest()
        manifest = mod.generate_manifest(str(tmp_path))

        # Seismic volume is still present
        seismic_vols = [v for v in manifest[0]["volumes"] if v["group"] == "Seismic"]
        assert len(seismic_vols) == 1
        # Broken store not in manifest
        all_paths = [v["store_path"] for v in manifest[0]["volumes"]]
        assert not any("broken" in p for p in all_paths)


class TestOverwriteFlag:
    """--overwrite flag controls whether an existing manifest is replaced."""

    def test_overwrite_false_skips_existing(self, tmp_path):
        """Without --overwrite, an existing manifest.json is left unchanged."""
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir)

        # Write a sentinel manifest
        manifest_path = tmp_path / "manifest.json"
        sentinel = [{"sentinel": True}]
        manifest_path.write_text(json.dumps(sentinel))

        mod = _load_generate_manifest()
        result = mod.generate_manifest(str(tmp_path), overwrite=False)

        # The file on disk should still be the sentinel
        with open(manifest_path) as f:
            data = json.load(f)
        assert data == sentinel

    def test_overwrite_true_replaces_existing(self, tmp_path):
        """With overwrite=True, an existing manifest.json is replaced."""
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir)

        # Write a sentinel manifest
        manifest_path = tmp_path / "manifest.json"
        sentinel = [{"sentinel": True}]
        manifest_path.write_text(json.dumps(sentinel))

        mod = _load_generate_manifest()
        mod.generate_manifest(str(tmp_path), overwrite=True)

        with open(manifest_path) as f:
            data = json.load(f)
        assert data != sentinel
        assert len(data) == 1
        assert "run_id" in data[0]


class TestCLI:
    """CLI invocation via subprocess."""

    def test_cli_exits_zero(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir)

        script = Path(__file__).parents[1] / "scripts" / "generate_manifest.py"
        result = subprocess.run(
            [sys.executable, str(script), "--project-folder", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CLI stderr:\n{result.stderr}"

    def test_cli_writes_manifest_json(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir)

        script = Path(__file__).parents[1] / "scripts" / "generate_manifest.py"
        subprocess.run(
            [sys.executable, str(script), "--project-folder", str(tmp_path)],
            capture_output=True,
            check=True,
        )

        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_cli_overwrite_flag(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        _make_seismic_store(run_dir)

        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text('[{"sentinel":true}]')

        script = Path(__file__).parents[1] / "scripts" / "generate_manifest.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--project-folder",
                str(tmp_path),
                "--overwrite",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        with open(manifest_path) as f:
            data = json.load(f)
        assert data != [{"sentinel": True}]

    def test_cli_empty_project_exits_zero(self, tmp_path):
        script = Path(__file__).parents[1] / "scripts" / "generate_manifest.py"
        result = subprocess.run(
            [sys.executable, str(script), "--project-folder", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
