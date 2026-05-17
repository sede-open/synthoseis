"""Uvicorn entry point for the Synthoseis simulation launcher API.

IMPORTANT — run from the synthoseis repo root:
    python -m api.server
    # or
    python api/server.py

This is required so that:
  - `python main.py` launched as a subprocess resolves to the correct file.
  - The `rockphysics/*.py` glob in GET /api/models finds the RPM modules.

Multi-worker mode (--workers > 1) is NOT supported: the in-memory `_processes`
registry in run_manager.py would be split across OS processes, breaking
DELETE /api/runs/{id} and SSE log streaming for any cross-worker run.
Always start with workers=1 (the default here).
"""
import argparse
import os
import pathlib

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthoseis launcher API server")
    parser.add_argument("--dev", action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    # Ensure we are at the repo root regardless of where the script was invoked from.
    os.chdir(pathlib.Path(__file__).parent.parent)

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.dev,
        workers=1,  # Multi-worker breaks the in-memory _processes registry — see docstring
    )
