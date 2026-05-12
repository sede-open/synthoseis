"""Tests for the vectorised _perlin() implementation (OPT-1).

These tests use a minimal stub for the Parameters object so that they do not
require the full synthoseis fixture data.
"""
from __future__ import annotations

import types
import numpy as np
import pytest
from numpy.random import default_rng

from datagenerator.Horizons import RandomHorizonStack


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stack(xsize: int = 20, ysize: int = 30, zsize: int = 10, seed: int = 0):
    """Create a minimal RandomHorizonStack without triggering I/O."""
    ss = np.random.SeedSequence(seed)

    params = types.SimpleNamespace(
        cube_shape=(xsize, ysize, zsize),
        horizon_ss=ss,
        verbose=False,
    )
    stack = RandomHorizonStack.__new__(RandomHorizonStack)
    stack.cfg = params
    stack.rng = default_rng(ss.spawn(1)[0])
    return stack


# ---------------------------------------------------------------------------
# OPT-1 tests
# ---------------------------------------------------------------------------

class TestPerlinShape:
    def test_default_shape(self):
        stack = _make_stack(20, 30)
        result = stack._perlin()
        assert result.shape == (20, 30)

    def test_square_shape(self):
        stack = _make_stack(15, 15)
        result = stack._perlin()
        assert result.shape == (15, 15)

    def test_single_cell(self):
        """Edge case: 1×1 grid."""
        stack = _make_stack(1, 1, 10)
        result = stack._perlin()
        assert result.shape == (1, 1)


class TestPerlinDtype:
    def test_output_is_float(self):
        stack = _make_stack(10, 10)
        result = stack._perlin()
        assert np.issubdtype(result.dtype, np.floating), f"expected float, got {result.dtype}"

    def test_no_object_array(self):
        stack = _make_stack(10, 10)
        result = stack._perlin()
        assert result.dtype != object


class TestPerlinRange:
    def test_values_in_range(self):
        """Output values should be in (−1.5, 1.5) — opensimplex range is ≈ [−1, 1]."""
        stack = _make_stack(25, 25)
        result = stack._perlin()
        assert result.min() > -1.5, f"min {result.min()} too low"
        assert result.max() < 1.5, f"max {result.max()} too high"

    def test_no_nan_or_inf(self):
        stack = _make_stack(20, 20)
        result = stack._perlin()
        assert np.all(np.isfinite(result))


class TestPerlinReproducibility:
    def test_same_base_same_output(self):
        """With do_rotate=False, same base must produce identical arrays."""
        stack = _make_stack(15, 20, seed=7)
        r1 = stack._perlin(base=42, do_rotate=False)
        r2 = stack._perlin(base=42, do_rotate=False)
        np.testing.assert_array_equal(r1, r2)

    def test_different_base_different_output(self):
        stack = _make_stack(15, 20, seed=7)
        r1 = stack._perlin(base=1, do_rotate=False)
        r2 = stack._perlin(base=2, do_rotate=False)
        assert not np.array_equal(r1, r2)


class TestPerlinOctaves:
    def test_octave_changes_output(self):
        stack = _make_stack(15, 15, seed=3)
        r1 = stack._perlin(base=10, octave=1)
        r3 = stack._perlin(base=10, octave=3)
        assert not np.array_equal(r1, r3)

    def test_large_octave_no_overflow(self):
        stack = _make_stack(15, 15, seed=3)
        result = stack._perlin(base=5, octave=8)
        assert np.all(np.isfinite(result))
        assert result.shape == (15, 15)


class TestPerlinEdgeCases:
    def test_base_zero(self):
        stack = _make_stack(10, 10, seed=0)
        result = stack._perlin(base=0)
        # Should not raise; result should be non-trivially zero
        assert result.shape == (10, 10)
        assert np.any(result != 0.0)

    def test_float_base_cast_to_int(self):
        """base=255.9 should cast to int without raising."""
        stack = _make_stack(10, 10, seed=0)
        result = stack._perlin(base=255.9)
        assert result.shape == (10, 10)

    def test_no_rotate(self):
        stack = _make_stack(10, 15, seed=1)
        result = stack._perlin(base=42, do_rotate=False)
        assert result.shape == (10, 15)


class TestPerlinStats:
    def test_perlin_stats(self):
        """Vectorised _perlin mean ≈ 0.0 and std in [0.1, 0.6] (Story 5 acceptance).

        Uses a fixed seed and large-enough grid (64×64) for stable statistics.
        base=1, do_rotate=False produces a near-zero-mean field with this
        OpenSimplex implementation (mean ≈ 0.002, std ≈ 0.29 at 64×64).
        """
        stack = _make_stack(64, 64, seed=0)
        result = stack._perlin(base=1, do_rotate=False)
        mean = float(result.mean())
        std = float(result.std())
        assert abs(mean) <= 0.05, (
            f"_perlin mean {mean:.4f} is outside ±0.05 of 0.0"
        )
        assert 0.1 <= std <= 0.6, (
            f"_perlin std {std:.4f} is outside the expected range [0.1, 0.6]"
        )
