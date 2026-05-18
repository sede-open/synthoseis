"""Numba Zoeppritz PP reflectivity kernel.

Replaces the per-angle Python loop in `SeismicVolume.create_rfc_volumes`
with a single parallel kernel that walks every trace once and fills all
incident angles in one pass.

Contract
--------
Inputs are 3D float32 cubes shaped ``(il, xl, z - 1)`` containing upper
and lower layer properties:

    vp1, vs1, rho1, vp2, vs2, rho2

and a 1D float64 array ``angles_rad`` of incident angles in radians. The
output array ``out`` must be pre-allocated with shape
``(n_angles, il, xl, z - 1)`` and dtype float32.

Implementation notes
--------------------
* The formulation matches :func:`tests._zoeppritz_reference.zoeppritz_vectorised`
  term-for-term. That reference performs **complex** arithmetic (the
  incident angle is promoted to ``complex128`` at the top) and only
  takes the real part of the final PP reflectivity. Complex arithmetic
  is required whenever ``p * vp2``, ``p * vs1``, or ``p * vs2`` exceeds
  1 (i.e. post-critical reflection), where ``arcsin`` of a real value
  above 1 returns a complex number with a non-zero imaginary part. A
  previous spike that clipped the ``arcsin`` argument to ``[-1, 1]``
  produced parity errors up to 1.24 on realistic 0-40° geometry — too
  large to use in production.
* To keep the kernel fully jittable in ``nopython`` mode while matching
  the reference, the trace/z inner loop uses scalar ``complex128``
  arithmetic with ``cmath.asin``, ``cmath.sin``, ``cmath.cos``. Only the
  real part of the final ratio is stored.
* The outer loop is parallel across traces (``prange``); angles run
  serially inside each trace so the inner hot path stays cache-friendly.
"""
from __future__ import annotations

import cmath

import numba
import numpy as np


@numba.njit(fastmath=False, cache=True, boundscheck=False, inline="always")
def _zoep_pp_complex(
    _vp1, _vs1, _rho1, _vp2, _vs2, _rho2, theta_c
):
    """Slow complex-arithmetic PP reflectivity for post-critical angles.

    ``theta_c`` is a complex128 scalar (the real incident angle in
    radians, promoted). ``fastmath=False`` keeps cmath branch cuts
    intact.
    """
    sin_theta = cmath.sin(theta_c)
    p = sin_theta / _vp1
    theta2 = cmath.asin(p * _vp2)
    phi1 = cmath.asin(p * _vs1)
    phi2 = cmath.asin(p * _vs2)

    sp1 = cmath.sin(phi1) ** 2
    sp2 = cmath.sin(phi2) ** 2
    ct = cmath.cos(theta_c)
    ct2 = cmath.cos(theta2)
    cp1 = cmath.cos(phi1)
    cp2 = cmath.cos(phi2)

    aa = _rho2 * (1.0 - 2.0 * sp2) - _rho1 * (1.0 - 2.0 * sp1)
    bb = _rho2 * (1.0 - 2.0 * sp2) + 2.0 * _rho1 * sp1
    cc = _rho1 * (1.0 - 2.0 * sp1) + 2.0 * _rho2 * sp2
    dd = 2.0 * (_rho2 * _vs2 * _vs2 - _rho1 * _vs1 * _vs1)

    ee = bb * ct / _vp1 + cc * ct2 / _vp2
    ff = bb * cp1 / _vs1 + cc * cp2 / _vs2
    gg = aa - dd * ct / _vp1 * cp2 / _vs2
    hh = aa - dd * ct2 / _vp2 * cp1 / _vs1

    det = ee * ff + gg * hh * p * p
    num = (
        ff * (bb * ct / _vp1 - cc * ct2 / _vp2)
        - hh * p * p * (aa + det * ct / _vp1 * cp2 / _vs2)
    )
    return (num / det).real


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def zoeppritz_all_angles(
    vp1: np.ndarray,
    vs1: np.ndarray,
    rho1: np.ndarray,
    vp2: np.ndarray,
    vs2: np.ndarray,
    rho2: np.ndarray,
    angles_rad: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fill ``out[a, i, j, k]`` with PP reflectivity for all angles.

    Parameters
    ----------
    vp1, vs1, rho1, vp2, vs2, rho2
        3D float32 arrays of upper/lower layer properties, shape
        ``(il, xl, zm1)``.
    angles_rad
        1D float64 array of incident angles in radians.
    out
        Pre-allocated float32 array of shape ``(n_ang, il, xl, zm1)``.
    """
    il, xl, zm1 = vp1.shape
    n_ang = angles_rad.shape[0]
    n_traces = il * xl

    # Pre-compute per-angle scalars once. `sin(theta)` and `cos(theta)`
    # are identical for every sample at a given angle — factor them out
    # of the hot loop.
    sin_theta_arr = np.empty(n_ang, dtype=np.float64)
    cos_theta_arr = np.empty(n_ang, dtype=np.float64)
    for a in range(n_ang):
        sin_theta_arr[a] = np.sin(angles_rad[a])
        cos_theta_arr[a] = np.cos(angles_rad[a])

    for t in numba.prange(n_traces):
        i = t // xl
        j = t - i * xl
        for k in range(zm1):
            _vp1 = np.float64(vp1[i, j, k])
            _vs1 = np.float64(vs1[i, j, k])
            _rho1 = np.float64(rho1[i, j, k])
            _vp2 = np.float64(vp2[i, j, k])
            _vs2 = np.float64(vs2[i, j, k])
            _rho2 = np.float64(rho2[i, j, k])

            # Critical-angle detector: the three arcsin arguments are
            # ``sin_theta * v / vp1``; post-critical iff that exceeds 1
            # for any of {vp2, vs1, vs2}. Precompute the worst-case
            # ratio so the per-angle check is one multiply and compare.
            inv_vp1 = 1.0 / _vp1
            vmax_ratio = _vp2 * inv_vp1
            v = _vs1 * inv_vp1
            if v > vmax_ratio:
                vmax_ratio = v
            v = _vs2 * inv_vp1
            if v > vmax_ratio:
                vmax_ratio = v

            # Precompute sample-only quantities that do not depend on
            # the incident angle — saves work proportional to n_angles.
            dd = 2.0 * (_rho2 * _vs2 * _vs2 - _rho1 * _vs1 * _vs1)

            for a in range(n_ang):
                sin_theta = sin_theta_arr[a]
                ct = cos_theta_arr[a]

                if sin_theta * vmax_ratio < 1.0:
                    # --- Real fast path (inlined) ------------------
                    p = sin_theta / _vp1
                    theta2 = np.arcsin(p * _vp2)
                    phi1 = np.arcsin(p * _vs1)
                    phi2 = np.arcsin(p * _vs2)

                    sp1 = np.sin(phi1) ** 2
                    sp2 = np.sin(phi2) ** 2
                    ct2 = np.cos(theta2)
                    cp1 = np.cos(phi1)
                    cp2 = np.cos(phi2)

                    aa = _rho2 * (1.0 - 2.0 * sp2) - _rho1 * (1.0 - 2.0 * sp1)
                    bb = _rho2 * (1.0 - 2.0 * sp2) + 2.0 * _rho1 * sp1
                    cc = _rho1 * (1.0 - 2.0 * sp1) + 2.0 * _rho2 * sp2

                    ee = bb * ct / _vp1 + cc * ct2 / _vp2
                    ff = bb * cp1 / _vs1 + cc * cp2 / _vs2
                    gg = aa - dd * ct / _vp1 * cp2 / _vs2
                    hh = aa - dd * ct2 / _vp2 * cp1 / _vs1

                    det = ee * ff + gg * hh * p * p
                    num = (
                        ff * (bb * ct / _vp1 - cc * ct2 / _vp2)
                        - hh * p * p * (aa + det * ct / _vp1 * cp2 / _vs2)
                    )
                    out[a, i, j, k] = np.float32(num / det)
                else:
                    # --- Complex fallback for post-critical --------
                    theta_c = complex(angles_rad[a], 0.0)
                    r = _zoep_pp_complex(
                        _vp1, _vs1, _rho1, _vp2, _vs2, _rho2, theta_c,
                    )
                    out[a, i, j, k] = np.float32(r)


def compute_rfc_volumes(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
    angles_deg,
    out: np.ndarray,
) -> None:
    """Fill ``out`` with per-angle PP reflectivity.

    Shapes: ``vp/vs/rho`` = ``(il, xl, z)``; ``out`` = ``(n_ang, il, xl, z-1)``.

    ``angles_deg`` is any iterable of degree values. Inputs are coerced
    to contiguous float32; the degree-to-radian conversion happens here
    so the kernel stays angle-representation agnostic.
    """
    vp1 = np.ascontiguousarray(vp[:, :, :-1], dtype=np.float32)
    vp2 = np.ascontiguousarray(vp[:, :, 1:], dtype=np.float32)
    vs1 = np.ascontiguousarray(vs[:, :, :-1], dtype=np.float32)
    vs2 = np.ascontiguousarray(vs[:, :, 1:], dtype=np.float32)
    rho1 = np.ascontiguousarray(rho[:, :, :-1], dtype=np.float32)
    rho2 = np.ascontiguousarray(rho[:, :, 1:], dtype=np.float32)
    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    zoeppritz_all_angles(vp1, vs1, rho1, vp2, vs2, rho2, angles_rad, out)
