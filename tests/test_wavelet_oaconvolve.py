"""Task 3.1 — ``apply_wavelet`` must use ``scipy.signal.oaconvolve``.

Overlap-add is algorithmically equivalent to the previous ``fftconvolve``
but cheaper for the long z-axis / short wavelet shapes we run.  We assert:

* output shape equals input shape,
* result matches a direct ``oaconvolve(cube, wavelet_3d, mode="same", axes=-1)``,
* the module source no longer imports or calls ``fftconvolve``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import oaconvolve

from datagenerator.Seismic import apply_wavelet


SEISMIC_SRC = Path(__file__).resolve().parent.parent / "datagenerator" / "Seismic.py"


def test_apply_wavelet_matches_oaconvolve():
    rng = np.random.default_rng(0)
    cube = rng.standard_normal((4, 8, 256)).astype(np.float64)
    wavelet = rng.standard_normal(65).astype(np.float64)

    got = apply_wavelet(cube, wavelet)
    expected = oaconvolve(
        cube, wavelet[np.newaxis, np.newaxis, :], mode="same", axes=-1
    )

    assert got.shape == cube.shape
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=1e-12)


def test_seismic_module_uses_oaconvolve_not_fftconvolve():
    src = SEISMIC_SRC.read_text()
    assert "oaconvolve" in src, "apply_wavelet must use oaconvolve (Task 3.4)"
    assert "fftconvolve" not in src, "fftconvolve must be removed (Task 3.4)"
