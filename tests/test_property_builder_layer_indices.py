"""Parity tests for the Task 8 per-layer index precomputation.

`_precompute_layer_indices` replaces five per-layer ``np.where(...)`` scans
in ``build_property_models_randomised_depth_v3`` with a single sort + slice.
The tests below check index-array equality against the reference scans.
"""
import numpy as np
import pytest

from datagenerator.Seismic import _compute_class_ids, _precompute_layer_indices


def _reference_layer_indices(integer_faulted_age, net_to_gross, class_id, z_max):
    """Reference path that the builder used pre-Task 8."""
    out = {}
    for z in range(1, z_max):
        layer_mask = integer_faulted_age == z
        out[z] = {
            "layer": np.where(layer_mask),
            "shale": np.where(layer_mask & (net_to_gross >= 0.0)),
            "brine": np.where(layer_mask & (class_id == 2)),
            "oil": np.where(layer_mask & (class_id == 3)),
            "gas": np.where(layer_mask & (class_id == 4)),
        }
    return out


def _assert_bucket_equal(a, b):
    assert len(a) == len(b) == 3
    for arr_a, arr_b in zip(a, b):
        np.testing.assert_array_equal(arr_a, arr_b)


def _assert_per_layer_equal(a, b):
    assert set(a.keys()) == set(b.keys())
    for z in a:
        for bucket in ("layer", "shale", "brine", "oil", "gas"):
            _assert_bucket_equal(a[z][bucket], b[z][bucket])


def _make_fixture(seed, shape=(12, 14, 20), z_max=8, sand_prob=0.35,
                  oil_prob=0.08, gas_prob=0.08):
    rng = np.random.default_rng(seed)
    age = rng.integers(0, z_max, size=shape).astype(np.int32)
    ng = rng.uniform(-0.2, 1.0, size=shape).astype(np.float32)
    # Force some voxels to behave like water (ng < 0) so the shale bucket
    # is not equal to the layer bucket.
    ng[rng.uniform(size=shape) < 0.1] = -1.0
    is_sand = rng.uniform(size=shape) < sand_prob
    ng[~is_sand & (age > 0)] = 0.0  # shale voxels
    oil = (rng.uniform(size=shape) < oil_prob).astype(np.float32)
    gas = (rng.uniform(size=shape) < gas_prob).astype(np.float32)
    # Mask oil/gas to sand voxels; overlap is allowed on purpose so
    # _compute_class_ids exercises its mutual-exclusion rule.
    oil[~is_sand | (age == 0)] = 0.0
    gas[~is_sand | (age == 0)] = 0.0
    return age, ng, oil, gas, z_max


@pytest.mark.parametrize("seed", [0, 1, 42, 2026])
def test_precompute_layer_indices_matches_where_loop(seed):
    age, ng, oil, gas, z_max = _make_fixture(seed)
    class_id = _compute_class_ids(age, ng, oil, gas)
    reference = _reference_layer_indices(age, ng, class_id, z_max)
    vectorised = _precompute_layer_indices(age, ng, class_id, z_max)
    _assert_per_layer_equal(reference, vectorised)


def test_precompute_layer_indices_empty_cube():
    """All-zero age volume: every bucket must be empty, no crashes."""
    age = np.zeros((5, 6, 7), dtype=np.int32)
    ng = np.zeros_like(age, dtype=np.float32)
    oil = np.zeros_like(age, dtype=np.float32)
    gas = np.zeros_like(age, dtype=np.float32)
    class_id = _compute_class_ids(age, ng, oil, gas)
    result = _precompute_layer_indices(age, ng, class_id, z_max=4)
    assert set(result.keys()) == {1, 2, 3}
    for z in result:
        for bucket in ("layer", "shale", "brine", "oil", "gas"):
            i, j, k = result[z][bucket]
            assert i.size == 0
            assert j.size == 0
            assert k.size == 0


def test_precompute_layer_indices_preserves_c_order():
    """Index arrays must come out in the same C-order np.where produces,
    so downstream 1D slicing (k + delta_z) matches voxel-by-voxel."""
    age, ng, oil, gas, z_max = _make_fixture(seed=7, shape=(8, 8, 12))
    class_id = _compute_class_ids(age, ng, oil, gas)
    result = _precompute_layer_indices(age, ng, class_id, z_max)
    for z in range(1, z_max):
        i, j, k = result[z]["layer"]
        if i.size == 0:
            continue
        # C-order: flat index increasing monotonically
        flat = (i * age.shape[1] * age.shape[2]) + (j * age.shape[2]) + k
        assert np.all(np.diff(flat) > 0), f"layer {z} not in C-order"
