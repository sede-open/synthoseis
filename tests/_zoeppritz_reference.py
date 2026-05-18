"""Reference oracle for the numba Zoeppritz kernel parity tests.

This module lives under ``tests/`` because the original vectorised
numpy implementation is used *only* as the oracle by:

* ``tests/test_zoeppritz_numba_parity.py``
* ``scripts/bench_zoeppritz.py``

Task 5.6 (performance refactor purge) relocated it out of
``datagenerator.Seismic`` so the runtime import surface no longer
ships the dead reference loop.
"""
from __future__ import annotations

import numpy as np


def zoeppritz_vectorised(vp1, vs1, rho1, vp2, vs2, rho2, angle_deg):
    """Vectorised Zoeppritz PP reflectivity for a single angle.

    All inputs are broadcastable arrays (typically 3D: il × xl × z).
    angle_deg is a scalar angle in degrees.
    Returns real-valued PP reflectivity with the same shape as vp1.
    """
    theta = np.deg2rad(np.float64(angle_deg)) + 0j  # complex scalar
    p = np.sin(theta) / vp1
    theta2 = np.arcsin(p * vp2)
    phi1 = np.arcsin(p * vs1)
    phi2 = np.arcsin(p * vs2)

    sin_phi1_sq = np.sin(phi1) ** 2
    sin_phi2_sq = np.sin(phi2) ** 2
    cos_theta = np.cos(theta)
    cos_theta2 = np.cos(theta2)
    cos_phi1 = np.cos(phi1)
    cos_phi2 = np.cos(phi2)

    a = rho2 * (1 - 2 * sin_phi2_sq) - rho1 * (1 - 2 * sin_phi1_sq)
    b = rho2 * (1 - 2 * sin_phi2_sq) + 2 * rho1 * sin_phi1_sq
    c = rho1 * (1 - 2 * sin_phi1_sq) + 2 * rho2 * sin_phi2_sq
    d = 2 * (rho2 * vs2 ** 2 - rho1 * vs1 ** 2)

    e = b * cos_theta / vp1 + c * cos_theta2 / vp2
    f = b * cos_phi1 / vs1 + c * cos_phi2 / vs2
    g = a - d * cos_theta / vp1 * cos_phi2 / vs2
    h = a - d * cos_theta2 / vp2 * cos_phi1 / vs1

    det = e * f + g * h * p ** 2

    zoep_pp = (
        f * (b * cos_theta / vp1 - c * cos_theta2 / vp2)
        - h * p ** 2 * (a + det * cos_theta / vp1 * cos_phi2 / vs2)
    ) / det

    return np.real(zoep_pp).astype(np.float32, copy=True)
