"""xarray/zarr output writer for synthoseis final volumes.

Every final pipeline output is written as a zarr store that conforms to
the xarray/CF-conventions layout, readable with xarray.open_zarr(path).

Compression note (OPT-2):
    zarr 3.1.6 (zarr-v3 format, verified 2026-05-06) requires the zarr-native
    codec API for the encoding dict.  Specifically:
      - encoding key: ``"compressors"`` (list of BytesBytesCodec objects)
      - codec class:  ``zarr.codecs.BloscCodec``
    The legacy ``numcodecs.Blosc`` / ``"compressor"`` (singular) key is
    deprecated in zarr 3.x and raises a type error when used directly.
"""
from __future__ import annotations

import warnings
import numpy as np
import xarray as xr
from zarr.codecs import BloscCodec, BloscShuffle

# ---------------------------------------------------------------------------
# Module-level default compressor (OPT-2)
# zarr 3.x BloscCodec with zstd + bitshuffle gives ≈3–4× ratio on float32
# seismic data at clevel=5 with negligible CPU overhead.
# ---------------------------------------------------------------------------
_DEFAULT_COMPRESSOR = BloscCodec(
    cname="zstd",
    clevel=5,
    shuffle=BloscShuffle.bitshuffle,
)


def write_volume_to_zarr(
    arr: np.ndarray,
    path: str,
    name: str = "data",
    dims: tuple[str, ...] = ("inline", "crossline", "time"),
    coords: dict | None = None,
    attrs: dict | None = None,
    chunks: dict | None = None,
    compressor: BloscCodec | None = _DEFAULT_COMPRESSOR,
) -> None:
    """Write a numpy array as a CF-convention xarray zarr store.

    Parameters
    ----------
    arr : np.ndarray
        Array to write (any shape, any dtype).
    path : str
        Output zarr path (directory). Created if absent.
    name : str
        Variable name inside the dataset (default "data").
    dims : tuple[str, ...]
        Dimension names, length must match arr.ndim.
    coords : dict | None
        Coordinate arrays keyed by dim name. Optional.
    attrs : dict | None
        Global dataset attributes (e.g. angle_deg, sample_rate_ms).
    chunks : dict | None
        Chunk sizes keyed by dim name. None = xarray auto-chunking.
    compressor : BloscCodec | None
        zarr 3.x codec to apply when writing chunks.  Defaults to
        ``BloscCodec(cname="zstd", clevel=5, shuffle=bitshuffle)``.
        Pass ``None`` to write without compression (useful for benchmarks).
    """
    da = xr.DataArray(np.asarray(arr), dims=dims, coords=coords or {}, attrs=attrs or {})
    ds = da.to_dataset(name=name)

    # Build encoding dict combining chunk spec and compressor.
    encoding: dict = {}
    if chunks:
        encoding.setdefault(name, {})["chunks"] = [chunks.get(d, -1) for d in dims]
    if compressor is not None:
        # zarr 3.x uses "compressors" (list) not legacy "compressor" (scalar).
        encoding.setdefault(name, {})["compressors"] = [compressor]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Cc]onsolidated metadata.*", category=UserWarning)
        ds.to_zarr(path, mode="w", consolidated=True, encoding=encoding or None)


def read_volume_from_zarr(path: str, name: str = "data") -> np.ndarray:
    """Read a zarr volume back as a numpy array.

    Parameters
    ----------
    path : str
        Path to the zarr store written by write_volume_to_zarr.
    name : str
        Variable name inside the dataset.
    """
    ds = xr.open_zarr(path)
    return ds[name].values
