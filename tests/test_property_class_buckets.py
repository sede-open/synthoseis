"""Task 4.1 — class-bucket precompute for the property builder.

Class conventions (see ``datagenerator.Seismic._compute_class_ids``):

* 0 = unprocessed sand-wise (water / out-of-range / shale-only / oil+gas
  overlap); handled by the separate ``ng >= 0`` shale-baseline pass
* 2 = brine sand (``ng > 0`` & neither closure)
* 3 = oil sand (``ng > 0`` & oil-only closure)
* 4 = gas sand (``ng > 0`` & gas-only closure)

Class 1 is reserved by convention but not emitted.
"""
from __future__ import annotations

import numpy as np

from datagenerator.Seismic import _compute_class_ids


def test_class_ids_cover_all_sand_buckets():
    integer_age = np.array(
        [[[0, 1, 2, 3, 4]], [[1, 1, 1, 1, 1]]], dtype=np.int32
    )
    ng = np.array(
        [[[0.0, 0.0, 0.5, 0.5, 0.5]], [[-1.0, 0.0, 0.5, 0.5, 0.5]]],
        dtype=np.float32,
    )
    oil = np.array(
        [[[0, 0, 0, 1, 0]], [[0, 0, 0, 1, 0]]], dtype=np.float32
    )
    gas = np.array(
        [[[0, 0, 0, 0, 1]], [[0, 0, 0, 0, 1]]], dtype=np.float32
    )

    class_id = _compute_class_ids(integer_age, ng, oil, gas)

    assert class_id.dtype == np.int8
    assert class_id.shape == integer_age.shape
    np.testing.assert_array_equal(class_id[0, 0], [0, 0, 2, 3, 4])
    np.testing.assert_array_equal(class_id[1, 0], [0, 0, 2, 3, 4])


def test_class_ids_oil_gas_overlap_is_unprocessed():
    integer_age = np.ones((1, 1, 1), dtype=np.int32)
    ng = np.full((1, 1, 1), 0.5, dtype=np.float32)
    oil = np.ones((1, 1, 1), dtype=np.float32)
    gas = np.ones((1, 1, 1), dtype=np.float32)

    class_id = _compute_class_ids(integer_age, ng, oil, gas)
    assert class_id[0, 0, 0] == 0


def test_class_ids_out_of_range_marks_zero():
    integer_age = np.zeros((1, 1, 3), dtype=np.int32)
    ng = np.array([[[0.5, -1.0, 0.0]]], dtype=np.float32)
    oil = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    gas = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)

    class_id = _compute_class_ids(integer_age, ng, oil, gas)
    np.testing.assert_array_equal(class_id[0, 0], [0, 0, 0])


def test_class_ids_matches_naive_np_where_per_bucket():
    rng = np.random.default_rng(123)
    shape = (8, 8, 16)
    integer_age = rng.integers(0, 6, size=shape, dtype=np.int32)
    ng = rng.uniform(-0.2, 1.0, size=shape).astype(np.float32)
    oil = rng.choice([0.0, 1.0], size=shape, p=[0.8, 0.2]).astype(np.float32)
    gas = rng.choice([0.0, 1.0], size=shape, p=[0.85, 0.15]).astype(np.float32)

    class_id = _compute_class_ids(integer_age, ng, oil, gas)

    valid = integer_age > 0
    brine = valid & (ng > 0.0) & (oil == 0.0) & (gas == 0.0)
    oil_only = valid & (ng > 0.0) & (oil == 1.0) & (gas == 0.0)
    gas_only = valid & (ng > 0.0) & (oil == 0.0) & (gas == 1.0)

    np.testing.assert_array_equal(class_id == 2, brine)
    np.testing.assert_array_equal(class_id == 3, oil_only)
    np.testing.assert_array_equal(class_id == 4, gas_only)
    np.testing.assert_array_equal(class_id[~valid], 0)
    assert np.count_nonzero(class_id == 1) == 0
