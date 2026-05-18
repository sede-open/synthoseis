# Architecture

## Components

```
┌──────────────────────────────────────────────────────┐
│                    User interface                     │
│  CLI (main.py)          Dashboard (webapp + api/)     │
└────────────┬──────────────────────┬──────────────────┘
             │                      │ HTTP / SSE
             ▼                      ▼
┌────────────────────┐   ┌──────────────────────────┐
│  datagenerator/    │   │  api/                    │
│  (Python library)  │◄──│  FastAPI + uvicorn        │
└────────────────────┘   └──────────────────────────┘
             │
             ▼
┌────────────────────┐
│  rockphysics/      │
│  (RPM plug-ins)    │
└────────────────────┘
             │
             ▼
┌────────────────────┐
│  Output (Zarr)     │
│  project_folder/   │
└────────────────────┘
```

## `datagenerator/`

The core library. Contains no web or API code. Each module maps to one
geologic process:

| Module | Responsibility |
|--------|---------------|
| `Parameters` | Load config, build directory structure, hold global state |
| `Horizons` | Build the unfaulted stratigraphic layer stack |
| `Geomodels` | Compute lithology, net-to-gross, and fluid volumes on the layer stack |
| `Faults` | Generate fault geometries and apply them to all volumes |
| `Closures` | Flood-fill the faulted age model to identify hydrocarbon traps |
| `Salt` | Add salt diapirs as masked bodies |
| `Seismic` | Compute angle-dependent reflection coefficients and convolve with a wavelet |
| `Augmentation` | Apply post-stack geophysical augmentations (smoothing, integration, RMO) |
| `output_writer` | Serialise volumes to Zarr stores |

## `rockphysics/`

Plug-in modules that define depth trends for Vp, Vs, and ρ for each
fluid/facies combination. `Parameters` selects the correct module by matching
`config["project"]` to the file name (`rpm_<project>.py`).

## `api/`

A thin FastAPI layer that:
1. Validates the config with Pydantic (`api/config.py`)
2. Persists run state in a local SQLite database (`runs.db`)
3. Launches `main.py` as a subprocess
4. Streams stdout back to the client via SSE

The API is single-worker by design. The subprocess registry is held in memory
and cannot be shared across OS processes.

## `webapp/`

A React + Vite single-page application. In development it runs on port 5173
and proxies API calls to port 8000. In production the compiled bundle is served
directly by the API at `/`.

## `scripts/`

Utility scripts not part of the library:

| Script | Purpose |
|--------|---------|
| `dev.sh` | Start API + webapp together |
| `generate_manifest.py` | Scan output folder and write `manifest.json` |
