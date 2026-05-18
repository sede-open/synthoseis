# Synthoseis for agents and LLMs

This page is written for AI coding assistants, autonomous agents, and LLMs
working on this codebase. It provides the dense, structured context needed to
navigate and extend the project without hallucinating structure.

---

## Repo map (non-generated files)

```
synthoseis/
├── main.py                      # CLI entry point → build_model()
├── pyproject.toml               # deps, pytest markers
├── config/
│   └── example.json             # canonical config example
├── datagenerator/               # core generation library
│   ├── Parameters.py            # config loading + global state
│   ├── Horizons.py              # stratigraphic layer construction
│   ├── Geomodels.py             # elastic property volumes
│   ├── Faults.py                # fault geometry + application
│   ├── Closures.py              # trap identification + fluid fill
│   ├── Salt.py                  # salt body injection
│   ├── Seismic.py               # Zoeppritz + wavelet convolution
│   ├── Augmentation.py          # post-stack augmentations
│   ├── output_writer.py         # Zarr serialisation
│   └── zoeppritz_kernel.py      # Numba-compiled Zoeppritz kernel
├── rockphysics/
│   ├── RockPropertyModels.py    # base class + HDF5 helpers
│   └── rpm_example.py           # example RPM (copy to add new ones)
├── api/
│   ├── main.py                  # FastAPI app + all route handlers
│   ├── config.py                # SimulationConfig Pydantic model
│   ├── run_manager.py           # SQLite DB + subprocess launch + SSE
│   └── server.py                # uvicorn entry point
├── webapp/                      # React + Vite frontend
│   └── src/
│       ├── App.tsx
│       ├── components/          # UI components (LaunchPanel, RunViewer, …)
│       ├── hooks/               # useManifest, useZarrSlice
│       └── types/               # TypeScript types
├── scripts/
│   ├── dev.sh                   # start API + webapp
│   └── generate_manifest.py     # scan output folder → manifest.json
└── tests/                       # pytest test suite
```

---

## Key invariants

- **`main.py` must be run from the repo root.** `Parameters` resolves RPM
  paths relative to CWD. The API enforces this via `os.chdir` in `server.py`.
- **API is always single-worker.** The `_processes` dict in `run_manager.py`
  is in-process. Multi-worker mode silently breaks log streaming and
  cancellation.
- **Zarr v3 only.** All output uses `zarr>=3`. Do not introduce HDF5 writes.
- **`uv` is the package manager.** Do not use `pip` directly in scripts or
  CI — use `uv run` or `uv sync`.
- **Config validation is Pydantic v2.** `SimulationConfig` uses
  `model_config = ConfigDict(extra="forbid")`. Unknown keys are rejected.
- **RPM naming convention:** file `rpm_<name>.py` → class `RPM<Name>` →
  config `"project": "<name>"`.

---

## How to add a feature (agent checklist)

1. **Understand the data flow**: config JSON → `Parameters` → datagenerator
   stages → Zarr output → manifest → API → webapp.
2. **Backend changes** in `datagenerator/` require no API or webapp changes
   unless a new config key is needed.
3. **New config key**: add to `SimulationConfig` in `api/config.py` AND to
   `Parameters.__init__` in `datagenerator/Parameters.py`. Update
   `config/example.json`.
4. **New API route**: add to `api/main.py`. Update `reference/api-endpoints.md`.
5. **New RPM**: copy `rpm_example.py`, rename class. No other file changes
   needed — discovery is automatic.
6. **Frontend changes**: components live in `webapp/src/components/`. The API
   client is `webapp/src/api/client.ts`.
7. **Tests**: add to `tests/`. Mark slow end-to-end tests with
   `@pytest.mark.slow`.

---

## Running the project

```bash
# Install
uv sync

# Generate one model (CLI)
uv run python main.py --config config/example.json --num_runs 1 --run_id test

# Start the dashboard
./scripts/dev.sh          # API → :8000, webapp → :5173

# Run fast tests
uv run pytest

# Run all tests including slow
uv run pytest -m slow
```

---

## Common pitfalls

| Pitfall | Correct approach |
|---------|-----------------|
| Running `python main.py` from a subdirectory | Always run from repo root, or use `uv run python main.py` |
| Starting API with `--workers 2` | Always use default `workers=1` |
| Writing to HDF5 | Write to Zarr via `output_writer.write_volume_to_zarr` |
| Using `pip install` | Use `uv add <pkg>` to update `pyproject.toml` |
| Accessing `Parameters` attributes before `setup_model()` | Call `p.setup_model()` before any attribute access |

---

## `llms.txt`

A machine-readable site index is available at `/llms.txt` (generated during
the Sphinx build). It lists all documentation URLs for ingestion by LLM
retrieval pipelines.
