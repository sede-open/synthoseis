"""Failing tests for SimulationConfig Pydantic v2 model (api/config.py).

Run: pytest tests/test_api_config.py -v
These should FAIL until api/config.py is implemented.
"""
import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CONFIG = {
    "project": "example",
    "project_folder": "/tmp/synthoseis_example",
    "work_folder": "/tmp/synthoseis_work",
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


# ---------------------------------------------------------------------------
# Happy-path
# ---------------------------------------------------------------------------

def test_valid_config_round_trips():
    """Valid config parses without error and produces expected fields."""
    from api.config import SimulationConfig

    cfg = SimulationConfig(**VALID_CONFIG)
    assert cfg.project == "example"
    assert cfg.cube_shape == [300, 300, 1250]
    assert cfg.closure_types == ["simple", "faulted", "onlap"]
    assert cfg.sand_layer_fraction.min == pytest.approx(0.05)


def test_valid_config_json_serialisation():
    """to_config_json() returns a plain dict that matches the source."""
    from api.config import SimulationConfig

    cfg = SimulationConfig(**VALID_CONFIG)
    d = cfg.to_config_json()
    assert isinstance(d, dict)
    assert d["project"] == "example"
    assert d["cube_shape"] == [300, 300, 1250]
    assert "run_id" not in d
    assert d["sand_layer_fraction"] == {"min": 0.05, "max": 0.25}


def test_to_config_json_matches_example_json_keys():
    """Keys returned by to_config_json() are a superset of config/example.json keys
    (ignoring extra keys in example.json that we intentionally omit)."""
    import json, pathlib
    from api.config import SimulationConfig

    example = json.loads(
        (pathlib.Path(__file__).parent.parent / "config" / "example.json").read_text()
    )
    cfg = SimulationConfig(**VALID_CONFIG)
    d = cfg.to_config_json()
    # All our keys must be present; example.json may have extras we don't model
    for key in d:
        assert key in example or key == "write_to_hdf" or key == "multiprocess_bp", (
            f"Key '{key}' in to_config_json() not found in example.json"
        )


# ---------------------------------------------------------------------------
# thickness_min / thickness_max cross-field validator
# ---------------------------------------------------------------------------

def test_thickness_min_equal_max_raises():
    from api.config import SimulationConfig

    bad = {**VALID_CONFIG, "thickness_min": 5, "thickness_max": 5}
    with pytest.raises(ValidationError, match="thickness"):
        SimulationConfig(**bad)


def test_thickness_min_greater_than_max_raises():
    from api.config import SimulationConfig

    bad = {**VALID_CONFIG, "thickness_min": 10, "thickness_max": 5}
    with pytest.raises(ValidationError, match="thickness"):
        SimulationConfig(**bad)


# ---------------------------------------------------------------------------
# cube_shape validator
# ---------------------------------------------------------------------------

def test_cube_shape_non_positive_raises():
    from api.config import SimulationConfig

    bad = {**VALID_CONFIG, "cube_shape": [0, 300, 1250]}
    with pytest.raises(ValidationError):
        SimulationConfig(**bad)


def test_cube_shape_negative_raises():
    from api.config import SimulationConfig

    bad = {**VALID_CONFIG, "cube_shape": [300, -1, 1250]}
    with pytest.raises(ValidationError):
        SimulationConfig(**bad)


def test_cube_shape_wrong_length_raises():
    from api.config import SimulationConfig

    bad = {**VALID_CONFIG, "cube_shape": [300, 300]}
    with pytest.raises(ValidationError):
        SimulationConfig(**bad)


# ---------------------------------------------------------------------------
# closure_types validator
# ---------------------------------------------------------------------------

def test_closure_types_empty_raises():
    from api.config import SimulationConfig

    bad = {**VALID_CONFIG, "closure_types": []}
    with pytest.raises(ValidationError):
        SimulationConfig(**bad)


def test_closure_types_invalid_value_raises():
    from api.config import SimulationConfig

    bad = {**VALID_CONFIG, "closure_types": ["simple", "invalid_type"]}
    with pytest.raises(ValidationError):
        SimulationConfig(**bad)


# ---------------------------------------------------------------------------
# incident_angles validator
# ---------------------------------------------------------------------------

def test_incident_angles_empty_raises():
    from api.config import SimulationConfig

    bad = {**VALID_CONFIG, "incident_angles": []}
    with pytest.raises(ValidationError):
        SimulationConfig(**bad)


# ---------------------------------------------------------------------------
# sand_layer_fraction cross-field validator
# ---------------------------------------------------------------------------

def test_sand_fraction_min_equals_max_raises():
    from api.config import SimulationConfig

    bad = {**VALID_CONFIG, "sand_layer_fraction": {"min": 0.1, "max": 0.1}}
    with pytest.raises(ValidationError):
        SimulationConfig(**bad)


def test_sand_fraction_min_greater_than_max_raises():
    from api.config import SimulationConfig

    bad = {**VALID_CONFIG, "sand_layer_fraction": {"min": 0.5, "max": 0.1}}
    with pytest.raises(ValidationError):
        SimulationConfig(**bad)


# ---------------------------------------------------------------------------
# Extra fields forbidden
# ---------------------------------------------------------------------------

def test_extra_fields_forbidden():
    from api.config import SimulationConfig

    bad = {**VALID_CONFIG, "unknown_field": "should_fail"}
    with pytest.raises(ValidationError):
        SimulationConfig(**bad)
