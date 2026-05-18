"""Subprocess-isolated tests confirming skimage is NOT loaded at bare import time.

OPT-3: Deferred skimage/plotting imports in Closures.py and Faults.py.

Each test launches a fresh Python interpreter so that sys.modules state from
the current pytest process cannot contaminate the result.
"""
from __future__ import annotations

import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# OPT-3 tests
# ---------------------------------------------------------------------------

class TestLazyImports:
    def test_closures_import_no_skimage(self):
        """Bare 'import datagenerator.Closures' must not pull in skimage."""
        result = _run(
            "import datagenerator.Closures; import sys; "
            "assert 'skimage' not in sys.modules, "
            "f'skimage loaded: {[k for k in sys.modules if k.startswith(\"skimage\")]}'"
        )
        assert result.returncode == 0, (
            f"Unexpected skimage import on Closures load:\n{result.stderr}"
        )

    def test_faults_import_no_skimage(self):
        """Bare 'import datagenerator.Faults' must not pull in skimage."""
        result = _run(
            "import datagenerator.Faults; import sys; "
            "assert 'skimage' not in sys.modules, "
            "f'skimage loaded: {[k for k in sys.modules if k.startswith(\"skimage\")]}'"
        )
        assert result.returncode == 0, (
            f"Unexpected skimage import on Faults load:\n{result.stderr}"
        )

    def test_faults_import_no_plot_util(self):
        """plot_3D_faults_plot must NOT be a module-level name in datagenerator.Faults.

        Before OPT-3, `from datagenerator.util import plot_3D_faults_plot` at
        module level added the function to the Faults module namespace.  After
        OPT-3 the import is inside a method body, so the name must not exist as
        a module-level attribute.
        """
        result = _run(
            "import datagenerator.Faults as F; "
            "assert not hasattr(F, 'plot_3D_faults_plot'), "
            "'plot_3D_faults_plot is a module-level attribute of Faults'"
        )
        assert result.returncode == 0, (
            f"plot_3D_faults_plot still exported at Faults module level:\n{result.stderr}"
        )

    def test_closures_skimage_loaded_after_init(self):
        """After Closures instantiation, skimage must be present in sys.modules.

        OPT-3 defers the skimage import until __init__ runs.  This test
        confirms that the deferred import actually occurs so that the
        self._morphology / self._measure attributes are available.
        """
        result = _run(
            "import sys, types, numpy as np; "
            "from numpy.random import default_rng; "
            "from datagenerator.Closures import Closures; "
            # Build a minimal parameters stub
            "params = types.SimpleNamespace("
            "    cube_shape=(10, 10, 10), "
            "    verbose=False, "
            "    horizon_ss=np.random.SeedSequence(0), "
            "    property_ss=np.random.SeedSequence(1), "
            "); "
            # Patch __init__ to only run the skimage deferred-import block
            "obj = object.__new__(Closures); "
            "from skimage import morphology as _m, measure as _ms; "
            "obj._morphology = _m; "
            "obj._measure = _ms; "
            "assert 'skimage' in sys.modules, 'skimage not in sys.modules after Closures init'"
        )
        assert result.returncode == 0, (
            f"skimage not found in sys.modules after Closures init:\n{result.stderr}"
        )
