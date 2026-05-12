"""Task 4.1 — ``fix_zero_values_at_base`` forward-fill parity + contract.

The refactor replaces the nested Python z-loop:

    for z in range(1, vol.shape[-1]):
        mask = vol[:, :, z] == 0.0
        vol[:, :, z][mask] = vol[:, :, z - 1][mask]

with a vectorised ``np.maximum.accumulate``-based forward-fill along
the last axis (see technical-spec §4). The rewrite must:

* leave leading zeros (before the first non-zero per trace) untouched
  — there is nothing shallower to pull from,
* forward-fill every internal zero with the nearest non-zero above,
* propagate the last non-zero sample to every trailing zero at the
  base of the cube,
* operate on all six property volumes (``rho``, ``vp``, ``vs``,
  ``rho_ff``, ``vp_ff``, ``vs_ff``).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from datagenerator.Seismic import SeismicVolume


def _reference_fill(vol):
    out = vol.copy()
    for z in range(1, out.shape[-1]):
        mask = out[:, :, z] == 0.0
        out[:, :, z][mask] = out[:, :, z - 1][mask]
    return out


def _make_props(rho):
    p = SimpleNamespace()
    p.rho = rho.copy()
    p.vp = rho.copy() * 2.0
    p.vs = rho.copy() * 0.5
    p.rho_ff = rho.copy()
    p.vp_ff = rho.copy() * 2.0
    p.vs_ff = rho.copy() * 0.5
    return p


def test_fix_zero_values_trailing_zeros_get_last_nonzero():
    vol = np.zeros((2, 2, 8), dtype=np.float32)
    vol[..., :4] = np.arange(1, 5, dtype=np.float32)  # [1,2,3,4,0,0,0,0]
    props = _make_props(vol)

    out = SeismicVolume.fix_zero_values_at_base(props)

    # After the fill, samples 4..7 must equal sample 3 (value 4)
    for name in ("rho", "vp", "vs", "rho_ff", "vp_ff", "vs_ff"):
        cube = getattr(out, name)
        expected_tail = np.broadcast_to(cube[..., 3:4], cube[..., 4:].shape)
        np.testing.assert_array_equal(cube[..., 4:], expected_tail)


def test_fix_zero_values_leading_zeros_preserved():
    vol = np.zeros((2, 2, 8), dtype=np.float32)
    vol[..., 3:] = np.arange(5, 10, dtype=np.float32)  # [0,0,0,5,6,7,8,9]
    props = _make_props(vol)

    out = SeismicVolume.fix_zero_values_at_base(props)

    for name in ("rho", "vp", "vs", "rho_ff", "vp_ff", "vs_ff"):
        cube = getattr(out, name)
        # Leading zeros stay zero (nothing shallower to pull from)
        assert np.all(cube[..., :3] == 0.0), f"{name} lost leading zeros"
        # Existing non-zero samples unchanged
        np.testing.assert_array_equal(cube[..., 3:], getattr(props, name)[..., 3:])


def test_fix_zero_values_internal_holes_forward_filled():
    vol = np.zeros((1, 1, 6), dtype=np.float32)
    vol[0, 0] = np.array([1.0, 0.0, 3.0, 0.0, 0.0, 6.0], dtype=np.float32)
    props = _make_props(vol)

    out = SeismicVolume.fix_zero_values_at_base(props)

    np.testing.assert_array_equal(
        out.rho[0, 0], np.array([1.0, 1.0, 3.0, 3.0, 3.0, 6.0], dtype=np.float32)
    )


def test_fix_zero_values_matches_reference_on_random_input():
    rng = np.random.default_rng(0)
    vol = rng.uniform(0.5, 5.0, size=(4, 5, 32)).astype(np.float32)
    # Knock out random samples, with heavier knockout at the base to
    # exercise the trailing-zeros path.
    vol[rng.random(vol.shape) < 0.2] = 0.0
    vol[..., -6:][rng.random((4, 5, 6)) < 0.7] = 0.0
    # Leave at least one leading-zero trace unchanged
    vol[0, 0, :2] = 0.0
    props = _make_props(vol)

    out = SeismicVolume.fix_zero_values_at_base(props)

    expected = _reference_fill(vol)
    np.testing.assert_array_equal(out.rho, expected)
    np.testing.assert_array_equal(out.rho_ff, expected)
    np.testing.assert_array_equal(out.vp, _reference_fill(vol * 2.0))
    np.testing.assert_array_equal(out.vs, _reference_fill(vol * 0.5))


def test_fix_zero_values_preserves_dtype():
    vol = np.zeros((2, 2, 8), dtype=np.float32)
    vol[..., :4] = 1.0
    props = _make_props(vol)

    out = SeismicVolume.fix_zero_values_at_base(props)
    for name in ("rho", "vp", "vs", "rho_ff", "vp_ff", "vs_ff"):
        assert getattr(out, name).dtype == np.float32, f"{name} dtype changed"
