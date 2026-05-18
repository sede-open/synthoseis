"""
Reproducibility test: two build_model() calls with seed=42, test_mode=50
must produce identical faulted_depth_maps arrays.

This test is RED until SeedSequence-based seeding lands in Phase 1.
"""
import json
import pathlib
import tempfile
import numpy as np
import pytest


def _run_model(tmp_path, run_index: int, seed: int = 42, test_mode: int = 50):
    """Run a single model build and return the faulted_depth_maps array."""
    config = {
        "project": "repro_test",
        "project_folder": str(tmp_path / f"project_{run_index}"),
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
        "model_store_in_memory": True,
        "cleanup_intermediates": False,
    }
    config_path = tmp_path / f"config_{run_index}.json"
    config_path.write_text(json.dumps(config))

    from datagenerator.Parameters import Parameters
    p = Parameters(str(config_path), test_mode=test_mode)
    p.setup_model(seed=seed)
    p.setup_model_store(in_memory=True)

    from datagenerator.Horizons import build_unfaulted_depth_maps
    depth_maps, onlap_list, fan_list, fan_thicknesses = build_unfaulted_depth_maps(p)
    return depth_maps


def test_reproducibility_seed42(tmp_path):
    """Two runs with seed=42 must yield bit-identical faulted_depth_maps."""
    maps_a = _run_model(tmp_path, run_index=0, seed=42)
    maps_b = _run_model(tmp_path, run_index=1, seed=42)
    np.testing.assert_array_equal(
        maps_a,
        maps_b,
        err_msg="faulted_depth_maps differ between two seed=42 runs",
    )


def test_different_seeds_differ(tmp_path):
    """Two runs with different seeds should produce different depth maps."""
    maps_a = _run_model(tmp_path, run_index=2, seed=42)
    maps_b = _run_model(tmp_path, run_index=3, seed=99)
    assert not np.array_equal(maps_a, maps_b), (
        "Expected different seeds to produce different depth_maps"
    )
