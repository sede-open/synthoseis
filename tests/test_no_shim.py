"""
AST-scan datagenerator/ and main.py to ensure no HDF5/PyTables shim symbols remain.
This test will be RED until the HDF5 -> zarr migration is complete.
"""
import ast
import os
import pathlib
import pytest

BANNED_NAMES = {
    "h5file",
    "hdf_init",
    "hdf_setup",
    "hdf_master",
    "hdf_store",
    "hdf_remove_node_list",
    "hdf_node_list",
    "write_data_to_hdf",
}

BANNED_IMPORT_MODULES = {"tables"}

WORKTREE = pathlib.Path(__file__).resolve().parents[1]
SCAN_DIRS = [WORKTREE / "datagenerator", WORKTREE / "rockphysics"]
SCAN_FILES = [WORKTREE / "main.py"]


def _collect_python_files():
    files = list(SCAN_FILES)
    for d in SCAN_DIRS:
        files.extend(sorted(d.glob("**/*.py")))
    return [f for f in files if f.exists()]


def _find_banned_in_file(filepath: pathlib.Path):
    """Return list of (lineno, symbol, source_line) for banned symbols."""
    hits = []
    source = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return hits

    for node in ast.walk(tree):
        # Check attribute access: x.h5file, cfg.hdf_init(...)
        if isinstance(node, ast.Attribute) and node.attr in BANNED_NAMES:
            line = source.splitlines()[node.lineno - 1].strip()
            hits.append((node.lineno, node.attr, line))
        # Check plain name usage
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            line = source.splitlines()[node.lineno - 1].strip()
            hits.append((node.lineno, node.id, line))
        # Check function calls by name: hdf_init(...)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_NAMES:
                line = source.splitlines()[node.func.lineno - 1].strip()
                hits.append((node.func.lineno, node.func.id, line))
        # Check imports: import tables
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BANNED_IMPORT_MODULES:
                    line = source.splitlines()[node.lineno - 1].strip()
                    hits.append((node.lineno, f"import {alias.name}", line))
        # Check from-imports: from tables import ...
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in BANNED_IMPORT_MODULES:
                line = source.splitlines()[node.lineno - 1].strip()
                hits.append((node.lineno, f"from {node.module} import ...", line))

    return hits


@pytest.mark.parametrize("filepath", _collect_python_files())
def test_no_hdf5_shim_symbols(filepath):
    """Each source file must contain zero HDF5/PyTables shim symbols."""
    rel = filepath.relative_to(WORKTREE)
    hits = _find_banned_in_file(filepath)
    assert hits == [], (
        f"Banned HDF5 symbols found in {rel}:\n"
        + "\n".join(f"  line {ln}: [{sym}]  {src}" for ln, sym, src in hits)
    )
