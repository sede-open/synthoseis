"""Generate a manifest.json for a synthoseis project folder.

Scans a project folder for completed synthoseis run subdirectories
(named ``seismic__{datestamp}_{run_id}/``) and for each run discovers
all zarr v3 stores across four fixed subdirectories plus optional
root-level QC volumes.

CLI usage::

    python scripts/generate_manifest.py --project-folder /path/to/project [--overwrite]

The manifest is written to ``{project_folder}/manifest.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Any

import xarray as xr
import zarr
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Subdirectory → group name mapping (order matters for display)
_SUBDIR_GROUP: dict[str, str] = {
    "seismic": "Seismic",
    "geology": "Geology",
    "horizons": "Horizons",
    "closures": "Closures",
}

# Pattern for run folder names produced by Parameters.py
# Supports both legacy format (8-digit: "seismic__20260517_my_run_id")
# and current MMDD_HHMM format ("seismic__0517_2351_my_run_id").
_RUN_FOLDER_RE = re.compile(r"^seismic__(\d{8}|\d{4}_\d{4})_(.+)$")


# ---------------------------------------------------------------------------
# Codec string helpers
# ---------------------------------------------------------------------------


def _compressor_string(za: zarr.Array) -> str:
    """Return a human-readable compressor description for a zarr Array.

    Tries to extract BloscCodec fields; falls back to repr of the last codec.
    """
    try:
        codecs = za.metadata.codecs  # tuple of BytesBytesCodec
        for codec in reversed(codecs):
            cname = getattr(codec, "cname", None)
            if cname is not None:
                # BloscCodec
                cname_str = str(cname.value) if hasattr(cname, "value") else str(cname)
                clevel = getattr(codec, "clevel", "?")
                shuffle = getattr(codec, "shuffle", None)
                shuffle_str = (
                    str(shuffle.value) if hasattr(shuffle, "value") else str(shuffle)
                )
                return f"blosc:{cname_str}:{clevel}:{shuffle_str}"
        # Fallback: use repr of last non-bytes codec, or "none"
        return "none"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Store scanning helpers
# ---------------------------------------------------------------------------


def _scan_zarr_store(store_path: Path, group_name: str, run_dir: Path) -> dict[str, Any] | None:
    """Open a zarr v3 store and extract volume metadata.

    Returns a VolumeInfo dict or None if the store cannot be read.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            g = zarr.open_consolidated(str(store_path), zarr_format=3)
    except Exception as exc:
        logger.warning("Skipping %s — cannot open consolidated zarr: %s", store_path, exc)
        return None

    try:
        ds = xr.open_zarr(str(store_path))
    except Exception as exc:
        logger.warning("Skipping %s — cannot open with xarray: %s", store_path, exc)
        return None

    # First data variable
    if not ds.data_vars:
        logger.warning("Skipping %s — no data variables found", store_path)
        return None

    var_name = next(iter(ds.data_vars))
    da = ds[var_name]

    # Chunk shape from encoding (tuple of ints)
    chunks_raw = da.encoding.get("chunks", None)
    if chunks_raw is not None:
        chunks = list(int(c) for c in chunks_raw)
    else:
        chunks = list(da.shape)

    # Compressor from zarr metadata
    compressor = "none"
    if var_name in g:
        compressor = _compressor_string(g[var_name])

    # Store path relative to the run dir
    rel_store_path = store_path.relative_to(run_dir)

    return {
        "name": store_path.stem,  # e.g. "seismicCubes_RFC_7_degrees_20260517"
        "store_path": str(rel_store_path),
        "variable": var_name,
        "group": group_name,
        "shape": list(da.shape),
        "dtype": str(da.dtype),
        "dims": list(da.dims),
        "chunks": chunks,
        "compressor": compressor,
        "attrs": dict(da.attrs),
    }


def _read_parameters_db(run_dir: Path) -> dict[str, str]:
    """Read model_parameters table from parameters.db.

    Returns empty dict if the file is missing or unreadable.
    """
    db_path = run_dir / "parameters.db"
    if not db_path.exists():
        logger.warning("No parameters.db found in %s", run_dir)
        return {}
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT key, value FROM model_parameters"
        ).fetchall()
        conn.close()
        return {str(k): str(v) for k, v in rows}
    except Exception as exc:
        logger.warning("Could not read parameters.db in %s: %s", run_dir, exc)
        return {}


def _scan_run_dir(run_dir: Path) -> dict[str, Any]:
    """Scan a single run directory and return its manifest entry."""
    folder_name = run_dir.name
    m = _RUN_FOLDER_RE.match(folder_name)
    if m:
        datestamp = m.group(1)
        run_id = m.group(2)
    else:
        # Fallback: use full folder name as run_id, unknown datestamp
        datestamp = ""
        run_id = folder_name

    volumes: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # 1. Scan the four named subdirectories
    # ------------------------------------------------------------------ #
    for subdir_name, group_name in _SUBDIR_GROUP.items():
        subdir = run_dir / subdir_name
        if not subdir.is_dir():
            continue
        for store_path in sorted(subdir.iterdir()):
            if store_path.suffix != ".zarr":
                continue
            vol = _scan_zarr_store(store_path, group_name, run_dir)
            if vol is not None:
                volumes.append(vol)

    # ------------------------------------------------------------------ #
    # 2. Scan root-level QC zarr stores
    #    Convention: {fname}_{datestamp}.zarr at run_dir root
    # ------------------------------------------------------------------ #
    for store_path in sorted(run_dir.iterdir()):
        if store_path.suffix != ".zarr":
            continue
        # A zarr store at the root must be a QC volume
        vol = _scan_zarr_store(store_path, "QC", run_dir)
        if vol is not None:
            volumes.append(vol)

    # ------------------------------------------------------------------ #
    # 3. Determine cube_shape from first 3-D volume; else empty list
    # ------------------------------------------------------------------ #
    cube_shape: list[int] = []
    for vol in volumes:
        if len(vol["shape"]) == 3:
            cube_shape = vol["shape"]
            break

    # ------------------------------------------------------------------ #
    # 4. Parameters
    # ------------------------------------------------------------------ #
    parameters = _read_parameters_db(run_dir)

    return {
        "run_id": run_id,
        "folder": folder_name,
        "datestamp": datestamp,
        "cube_shape": cube_shape,
        "volumes": volumes,
        "parameters": parameters,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_manifest(
    project_folder: str,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Scan *project_folder* for synthoseis runs and write ``manifest.json``.

    Parameters
    ----------
    project_folder:
        Path to the directory that contains ``seismic__*/`` run folders.
    overwrite:
        If ``False`` (default) and ``manifest.json`` already exists, skip
        generation and return the existing manifest.  If ``True``, always
        regenerate.

    Returns
    -------
    list[dict]
        The manifest as a Python list of run entry dicts.
    """
    project_path = Path(project_folder)
    manifest_path = project_path / "manifest.json"

    if manifest_path.exists() and not overwrite:
        logger.info("manifest.json already exists; use --overwrite to regenerate.")
        with open(manifest_path) as f:
            return json.load(f)

    # Discover run folders
    run_dirs = sorted(
        d
        for d in project_path.iterdir()
        if d.is_dir() and _RUN_FOLDER_RE.match(d.name)
    )

    manifest: list[dict[str, Any]] = []
    for run_dir in tqdm(run_dirs, desc="Scanning runs", unit="run"):
        entry = _scan_run_dir(run_dir)
        manifest.append(entry)

    # Write manifest.json
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote %s (%d run(s))", manifest_path, len(manifest))
    return manifest


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate manifest.json for a synthoseis project folder."
    )
    parser.add_argument(
        "--project-folder",
        required=True,
        help="Path to the synthoseis project folder containing seismic__*/ run subdirs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite an existing manifest.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    args = _build_parser().parse_args(argv)
    generate_manifest(args.project_folder, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
