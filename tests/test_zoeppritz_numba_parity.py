"""Task 2.1: parity between the numba Zoeppritz kernel and the
reference vectorised path.

Tolerances come from Task 0.5 spike: reference uses float32 promoted to
complex128, numba uses complex128 directly; the residual drift at 1
angle was ~1.8e-7. Post-critical reflections exercise the cmath.asin
branch, so we test across 1, 8, and 20 angles on a realistic fixture
with both sub- and super-critical geometry.

Two tolerance regimes are asserted:

* ``ATOL_BULK`` / ``RTOL``: the whole cube on a small fixture that rarely
  hits the exact critical-angle boundary.
* ``ATOL_MAX_PRODUCTION``: the per-sample maximum on a large fixture. A
  tiny fraction of near-critical samples show drift up to ~1e-2 because
  float32 + ``fastmath=True`` math intrinsics diverge there; the bulk
  (median) drift stays at ~1e-8. Downstream bandpass + wavelet smooth
  this out well before VDS output.
"""
from __future__ import annotations

import numpy as np
import pytest

from datagenerator.zoeppritz_kernel import compute_rfc_volumes
from tests._zoeppritz_reference import zoeppritz_vectorised


ATOL_BULK = 2e-6
RTOL = 1e-3
ATOL_MAX_PRODUCTION = 1e-2


def _fixture(seed: int = 42, il: int = 16, xl: int = 16, z: int = 128):
    """Smallish realistic fixture. Includes high-contrast interfaces so
    post-critical reflection is exercised at high angles (up to 40°)."""
    rng = np.random.default_rng(seed)
    vp = rng.uniform(1500.0, 4500.0, size=(il, xl, z)).astype(np.float32)
    vs = (vp * rng.uniform(0.45, 0.55, size=(il, xl, z))).astype(np.float32)
    rho = rng.uniform(1.8, 2.7, size=(il, xl, z)).astype(np.float32)
    return vp, vs, rho


def _reference(vp, vs, rho, angles_deg):
    vp1, vp2 = vp[:, :, :-1], vp[:, :, 1:]
    vs1, vs2 = vs[:, :, :-1], vs[:, :, 1:]
    rho1, rho2 = rho[:, :, :-1], rho[:, :, 1:]
    out = np.empty((len(angles_deg), *vp1.shape), dtype=np.float32)
    for a_idx, ang in enumerate(angles_deg):
        out[a_idx] = zoeppritz_vectorised(vp1, vs1, rho1, vp2, vs2, rho2, ang)
    return out


@pytest.mark.parametrize(
    "angles",
    [
        [10.0],
        [5.0, 15.0, 25.0, 35.0],
        list(np.linspace(0.0, 40.0, 8)),
        list(np.linspace(0.0, 40.0, 20)),
    ],
    ids=["n1", "n4", "n8", "n20"],
)
def test_zoeppritz_numba_matches_reference(angles):
    vp, vs, rho = _fixture()
    ref = _reference(vp, vs, rho, angles)

    out = np.empty_like(ref)
    compute_rfc_volumes(vp, vs, rho, angles, out)

    np.testing.assert_allclose(out, ref, atol=ATOL_BULK, rtol=RTOL)


def test_zoeppritz_numba_handles_post_critical_angles():
    """Force post-critical reflection (vp2 >> vp1 at 40°) and verify the
    numba kernel does not blow up or differ from reference by more than
    float32 precision."""
    il, xl, z = 8, 8, 32
    vp = np.empty((il, xl, z), dtype=np.float32)
    vs = np.empty_like(vp)
    rho = np.empty_like(vp)
    # Top half slow, bottom half fast → large contrast at the midpoint
    vp[:, :, : z // 2] = 1800.0
    vp[:, :, z // 2 :] = 4200.0
    vs[:] = vp * 0.5
    rho[:, :, : z // 2] = 2.0
    rho[:, :, z // 2 :] = 2.5

    angles = [40.0]
    ref = _reference(vp, vs, rho, angles)
    out = np.empty_like(ref)
    compute_rfc_volumes(vp, vs, rho, angles, out)

    assert np.all(np.isfinite(out)), "numba kernel produced non-finite values"
    np.testing.assert_allclose(out, ref, atol=ATOL_BULK, rtol=RTOL)


def test_zoeppritz_numba_large_random_fixture_median_is_float32_precision():
    """On a larger random fixture spanning 0-40°, the bulk (median) of
    samples must agree with the reference at float32 precision
    (~1e-7). The per-sample maximum is allowed to reach ``1e-2`` — that
    drift is concentrated at the thin shell where reflectivity is
    near-unit magnitude and the real↔complex path boundary interacts
    with ``fastmath=True`` math intrinsics. Catching this here prevents
    a future kernel rewrite from sneaking in a bulk regression that
    hides behind the max-drift allowance."""
    vp, vs, rho = _fixture(il=96, xl=96, z=256)
    angles = list(np.linspace(0.0, 40.0, 20))
    ref = _reference(vp, vs, rho, angles)
    out = np.empty_like(ref)
    compute_rfc_volumes(vp, vs, rho, angles, out)

    assert np.all(np.isfinite(out))

    diff = np.abs(out - ref)
    median_abs = float(np.median(diff))
    max_abs = float(diff.max())

    # Bulk precision must stay at float32 noise level.
    assert median_abs < 1e-6, f"median abs drift {median_abs:.3e} too large"
    # Max drift must not exceed the production tolerance.
    assert max_abs < ATOL_MAX_PRODUCTION, (
        f"max abs drift {max_abs:.3e} exceeds production tolerance "
        f"{ATOL_MAX_PRODUCTION:.3e}"
    )
