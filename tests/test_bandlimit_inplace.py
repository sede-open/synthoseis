"""Task 3.1 — parity + contract tests for the refactored bandlimit path.

Covers:
* ``apply_butterworth_bandpass`` no longer depends on ``np.nan_to_num`` or
  ``method="gust"``; it must produce the same output as a direct serial
  ``filtfilt(b, a, data, method="pad")`` call.
* ``apply_bandlimits`` mutates its input cube in place and returns the
  same object (no multiprocessing ``Pool`` round-trip that re-allocates).
* The ``multiprocess_bp`` flag is gone from the module source — neither
  read nor referenced in control flow.
* The ``Pool`` import is gone from the module source.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.signal import filtfilt

from datagenerator.Seismic import (
    SeismicVolume,
    apply_butterworth_bandpass,
    derive_butterworth_bandpass,
)


SEISMIC_SRC = Path(__file__).resolve().parent.parent / "datagenerator" / "Seismic.py"


def _seismic_src() -> str:
    return SEISMIC_SRC.read_text()


def test_apply_butterworth_bandpass_matches_plain_filtfilt_pad():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((4, 8, 64)).astype(np.float64)
    b, a = derive_butterworth_bandpass(4.0, 90.0, 4.0, order=4)

    got = apply_butterworth_bandpass(data, b, a)
    expected = filtfilt(b, a, data, method="pad")

    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=1e-12)


def test_apply_butterworth_bandpass_does_not_use_nan_to_num_or_gust():
    src = _seismic_src()
    marker = "def apply_butterworth_bandpass("
    idx = src.index(marker)
    # Slice to end of function: take everything up to the next top-level
    # ``def `` / ``class `` or EOF.
    after = src[idx + len(marker):]
    next_def = len(after)
    for needle in ("\ndef ", "\nclass "):
        pos = after.find(needle)
        if pos != -1 and pos < next_def:
            next_def = pos
    body = after[:next_def]
    # Look for actual call syntax, not docstring references to removed
    # names. The function body must be free of every call pattern we
    # intentionally dropped in Task 3.3b.
    assert "nan_to_num(" not in body, "nan_to_num() call must be removed per Task 3.3b"
    assert 'method="gust"' not in body, 'method="gust" must be removed per Task 3.3b'
    assert "method='gust'" not in body
    assert "irlen=" not in body, "irlen= kwarg must be removed per Task 3.3b"


def test_apply_bandlimits_is_in_place():
    """``apply_bandlimits`` must mutate its input cube and return it."""
    rng = np.random.default_rng(7)
    cube_shape = (8, 16, 128)
    data = rng.standard_normal((3, *cube_shape)).astype(np.float32)

    cfg = SimpleNamespace(
        digi=4.0,
        infill_factor=1,
        pad_samples=0,
        cube_shape=cube_shape,
        lowfreq=4.0,
        highfreq=90.0,
        order=4,
        verbose=False,
    )
    seismic = SeismicVolume.__new__(SeismicVolume)
    seismic.cfg = cfg

    before_id = id(data)
    out = seismic.apply_bandlimits(data)
    assert id(out) == before_id, "apply_bandlimits must return input array unchanged"
    assert out.shape == data.shape
    assert np.all(np.isfinite(out))


def test_seismic_module_has_no_multiprocess_bp_or_pool():
    src = _seismic_src()
    assert "multiprocess_bp" not in src, (
        "multiprocess_bp flag must be deleted from Seismic.py (Task 3.2)"
    )
    assert "from multiprocessing import Pool" not in src, (
        "Pool import must be deleted from Seismic.py (Task 3.2)"
    )
    # Also ensure the staticmethod wrapper for the Pool starmap is gone.
    assert "_run_bandpass_on_cubes" not in src, (
        "_run_bandpass_on_cubes staticmethod is now dead code (Task 3.2)"
    )
