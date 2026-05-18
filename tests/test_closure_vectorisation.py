"""Task 7 — parity tests for vectorised Closures helpers.

Each test exercises a vectorised helper in ``datagenerator._closures_vectorised``
against an inlined Python-loop reference copied verbatim from
``datagenerator.Closures`` pre-Task-7. Deterministic synthetic fixtures
with controlled label layouts stand in for the production
``skimage.measure.label`` output.
"""
from __future__ import annotations

import numpy as np
import pytest

from datagenerator._closures_vectorised import (
    assign_fluid_types_vectorised,
    bbox_for_label_and_fault,
    bincount_label_sizes,
    closure_size_filter_sizes,
    parse_closure_codes_vectorised,
    relabel_consecutive,
)


def _make_labels(shape=(32, 32, 64), n_labels=7, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, dtype=np.int32)
    # plant n_labels non-overlapping "blobs" with label ids 1..n
    for lid in range(1, n_labels + 1):
        origin = rng.integers(0, np.array(shape) - 6, size=3)
        size = rng.integers(2, 6, size=3)
        sl = tuple(slice(int(o), int(o + s)) for o, s in zip(origin, size))
        # only overwrite background so blobs do not merge
        chunk = labels[sl]
        chunk[chunk == 0] = lid
        labels[sl] = chunk
    return labels


# --------------------------------------------------------------------------- #
# bincount_label_sizes
# --------------------------------------------------------------------------- #


def test_bincount_matches_per_label_scan():
    labels = _make_labels()
    expected = np.array(
        [labels[labels == x].size for x in range(int(labels.max()) + 1)],
        dtype=np.int64,
    )
    got = bincount_label_sizes(labels)
    np.testing.assert_array_equal(got, expected)


def test_bincount_accepts_float_labels():
    """parse_closure_codes casts labels to float32 before its loop; the helper
    must accept that dtype."""
    labels = _make_labels().astype(np.float32)
    got = bincount_label_sizes(labels)
    assert got.dtype == np.int64
    assert int(got.sum()) == labels.size


# --------------------------------------------------------------------------- #
# closure_size_filter_sizes
# --------------------------------------------------------------------------- #


def test_closure_size_filter_sizes_matches_original():
    labels = _make_labels()
    labels_after = labels.copy()
    # simulate remove_small_objects zeroing label 3
    labels_after[labels_after == 3] = 0

    s_expected = [labels[labels == x].size for x in range(1, 1 + int(np.max(labels)))]
    t_expected = [
        labels_after[labels_after == x].size
        for x in range(1, 1 + int(np.max(labels_after)))
    ]

    s_got, t_got = closure_size_filter_sizes(labels, labels_after)
    assert s_got == s_expected
    assert t_got == t_expected


# --------------------------------------------------------------------------- #
# relabel_consecutive
# --------------------------------------------------------------------------- #


def _reference_parse_label_values_and_counts(labels_clean: np.ndarray):
    """Byte-for-byte copy of the pre-refactor Closures.parse_label_values_and_counts."""
    next_label = 0
    label_values = [0]
    label_counts = [labels_clean[labels_clean == 0].size]
    for _ in range(1, int(labels_clean.max()) + 1):
        try:
            next_label = labels_clean[labels_clean > next_label].min()
        except (TypeError, ValueError):
            break
        label_values.append(next_label)
        label_counts.append(labels_clean[labels_clean == next_label].size)
    for i, ilabel in enumerate(label_values):
        labels_clean[labels_clean == ilabel] = i
        label_values[i] = i
    label_values.remove(0)
    return label_values, labels_clean


def test_relabel_consecutive_with_gaps():
    # Sparse label ids to mimic post-remove_small_objects input
    labels = _make_labels()
    # Introduce gaps: drop labels 2 and 5
    labels[labels == 2] = 0
    labels[labels == 5] = 0

    ref = labels.copy()
    got_values, got_labels = relabel_consecutive(labels.copy())
    ref_values, ref_labels = _reference_parse_label_values_and_counts(ref)

    assert got_values == ref_values
    np.testing.assert_array_equal(got_labels, ref_labels)


def test_relabel_consecutive_preserves_dtype():
    labels = _make_labels().astype(np.int16)
    _, new_labels = relabel_consecutive(labels)
    assert new_labels.dtype == np.int16


# --------------------------------------------------------------------------- #
# parse_closure_codes_vectorised
# --------------------------------------------------------------------------- #


def _reference_parse_closure_codes(hc_closure_codes, labels, num, code=0.1):
    labels = labels.astype("float32").copy()
    if num > 0:
        for x in range(1, num + 1):
            y = code + labels[labels == x].size
            labels[labels == x] = y
        hc_closure_codes = hc_closure_codes + labels
    return hc_closure_codes


def test_parse_closure_codes_matches_original():
    labels = _make_labels()
    num = int(labels.max())
    hc_init = np.zeros_like(labels, dtype=np.float32)

    ref = _reference_parse_closure_codes(hc_init.copy(), labels.copy(), num, code=0.2)
    got = parse_closure_codes_vectorised(hc_init.copy(), labels.copy(), num, code=0.2)

    np.testing.assert_array_equal(got, ref)


def test_parse_closure_codes_zero_labels_returns_input():
    labels = np.zeros((4, 4, 4), dtype=np.int32)
    hc = np.ones_like(labels, dtype=np.float32) * 7.0
    got = parse_closure_codes_vectorised(hc.copy(), labels, 0, code=0.1)
    np.testing.assert_array_equal(got, hc)


# --------------------------------------------------------------------------- #
# assign_fluid_types_vectorised
# --------------------------------------------------------------------------- #


def _reference_assign_fluid_types(labels_clean, closure_segments, fluid_type_code):
    brine = np.zeros_like(labels_clean, dtype=np.uint8)
    oil = np.zeros_like(labels_clean, dtype=np.uint8)
    gas = np.zeros_like(labels_clean, dtype=np.uint8)
    for i in range(1, int(labels_clean.max()) + 1):
        if fluid_type_code[i] == 0:
            brine[np.logical_and(labels_clean == i, closure_segments > 0)] = 1
        elif fluid_type_code[i] == 1:
            oil[labels_clean == i] = 1
        elif fluid_type_code[i] == 2:
            gas[labels_clean == i] = 1
    return oil, gas, brine


@pytest.mark.parametrize("force_code", [None, 2])
def test_assign_fluid_types_matches_original(force_code):
    labels = _make_labels()
    closure_segments = (labels > 0).astype(np.uint8)
    # knock out a few voxels in closure_segments to exercise the brine gate
    closure_segments[labels == 3] = 0

    n = int(labels.max()) + 1
    if force_code is None:
        rng = np.random.default_rng(123)
        ftc = rng.integers(3, size=n)
    else:
        ftc = np.full(n, force_code, dtype=np.int64)

    ref_oil, ref_gas, ref_brine = _reference_assign_fluid_types(
        labels, closure_segments, ftc
    )
    got_oil, got_gas, got_brine = assign_fluid_types_vectorised(
        labels, closure_segments, ftc
    )

    np.testing.assert_array_equal(got_oil, ref_oil)
    np.testing.assert_array_equal(got_gas, ref_gas)
    np.testing.assert_array_equal(got_brine, ref_brine)


# --------------------------------------------------------------------------- #
# bbox_for_label_and_fault
# --------------------------------------------------------------------------- #


def test_bbox_tight_then_padded():
    labels = np.zeros((20, 20, 40), dtype=np.int32)
    labels[5:8, 6:10, 10:15] = 4
    fault = np.zeros_like(labels, dtype=np.float32)
    fault[5:8, 6:10, 10:15] = 12.0

    sl = bbox_for_label_and_fault(labels, 4, fault, 12.0, fault_tol=0.25, pad=2)
    assert sl is not None
    # tight box 5..8, 6..10, 10..15 padded by 2 and clipped at array edges
    assert sl == (slice(3, 10), slice(4, 12), slice(8, 17))


def test_bbox_returns_none_when_empty():
    labels = np.zeros((10, 10, 10), dtype=np.int32)
    fault = np.zeros_like(labels, dtype=np.float32)
    assert bbox_for_label_and_fault(labels, 7, fault, 0.0) is None
