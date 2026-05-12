"""
_noise_shim.py — opensimplex-backed drop-in for the `noise` package.

Provides:
    pnoise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0, repeatx=1024,
            repeaty=1024, base=0)

This shim is used when the `noise` package is not installed.
"""
import math
import numpy as np

try:
    import opensimplex as _osx
except ImportError as exc:
    raise ImportError(
        "Neither 'noise' nor 'opensimplex' is available. "
        "Install opensimplex>=0.4.5.1 to use the noise shim."
    ) from exc


def pnoise2(
    x: float,
    y: float,
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    repeatx: int = 1024,
    repeaty: int = 1024,
    base: int = 0,
) -> float:
    """Fractal Brownian Motion via opensimplex2."""
    _osx.seed(base)
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0
    for _ in range(octaves):
        value += _osx.noise2(x * frequency, y * frequency) * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return value / max_amplitude if max_amplitude > 0 else 0.0
