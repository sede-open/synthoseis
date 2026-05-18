"""
Tests for the zarr storage backend added to Parameters.
Verifies:
  - setup_model_store() returns zarr.Group on LocalStore by default
  - setup_model_store(in_memory=True) uses MemoryStore
  - create_array() returns zarr.Array with correct shape/dtype and overwrite semantics
"""
import os
import tempfile
import pytest
import zarr
import zarr.storage
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parameters(tmp_path, in_memory=False):
    """Create a minimal Parameters instance wired to a temp directory."""
    import json
    import pathlib

    config = {
        "project": "test",
        "project_folder": str(tmp_path / "project"),
        "work_folder": str(tmp_path),
        "cube_shape": [50, 50, 500],
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
        "dip_factor_max": 2,
        "min_number_faults": 1,
        "max_number_faults": 6,
        "pad_samples": 10,
        "max_column_height": [150.0, 150.0],
        "closure_types": ["simple"],
        "min_closure_voxels_simple": 500,
        "min_closure_voxels_faulted": 2500,
        "min_closure_voxels_onlap": 500,
        "sand_layer_thickness": 2,
        "sand_layer_fraction": {"min": 0.05, "max": 0.25},
        "extra_qc_plots": False,
        "verbose": False,
        "partial_voxels": True,
        "variable_shale_ng": False,
        "basin_floor_fans": False,
        "include_channels": False,
        "include_salt": False,
        "broadband_qc_volume": False,
        "model_qc_volumes": False,
        "model_store_in_memory": in_memory,
        "cleanup_intermediates": True,
    }

    config_path = tmp_path / "test_config.json"
    config_path.write_text(json.dumps(config))

    from datagenerator.Parameters import Parameters
    p = Parameters(str(config_path), test_mode=50)
    p.setup_model(seed=42)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_setup_model_store_local(tmp_path):
    """Default store should be a zarr.Group backed by LocalStore."""
    p = _make_parameters(tmp_path, in_memory=False)
    p.setup_model_store()

    assert isinstance(p.model_store, zarr.Group), (
        f"Expected zarr.Group, got {type(p.model_store)}"
    )
    assert isinstance(p.model_store.store, zarr.storage.LocalStore), (
        f"Expected LocalStore, got {type(p.model_store.store)}"
    )


def test_setup_model_store_memory(tmp_path):
    """in_memory=True should produce a zarr.Group backed by MemoryStore."""
    p = _make_parameters(tmp_path, in_memory=True)
    p.setup_model_store(in_memory=True)

    assert isinstance(p.model_store, zarr.Group)
    assert isinstance(p.model_store.store, zarr.storage.MemoryStore), (
        f"Expected MemoryStore, got {type(p.model_store.store)}"
    )


def test_create_array_shape_dtype(tmp_path):
    """create_array() must return zarr.Array with correct shape and dtype."""
    p = _make_parameters(tmp_path, in_memory=True)
    p.setup_model_store(in_memory=True)

    shape = (50, 50, 510)
    arr = p.create_array("test_volume", shape=shape, dtype="float32")

    assert isinstance(arr, zarr.Array), f"Expected zarr.Array, got {type(arr)}"
    assert arr.shape == shape, f"Expected shape {shape}, got {arr.shape}"
    assert arr.dtype == np.dtype("float32"), f"Expected float32, got {arr.dtype}"


def test_create_array_overwrite(tmp_path):
    """Calling create_array twice with the same name should not raise (overwrite=True)."""
    p = _make_parameters(tmp_path, in_memory=True)
    p.setup_model_store(in_memory=True)

    shape = (10, 10, 100)
    p.create_array("dup", shape=shape, dtype="float32")
    arr2 = p.create_array("dup", shape=(20, 20, 200), dtype="float64")
    assert arr2.shape == (20, 20, 200)


def test_create_array_default_dtype(tmp_path):
    """Default dtype for create_array should be float32."""
    p = _make_parameters(tmp_path, in_memory=True)
    p.setup_model_store(in_memory=True)

    arr = p.create_array("defaultdtype", shape=(5, 5, 50))
    assert arr.dtype == np.dtype("float32")
