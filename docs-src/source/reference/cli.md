# CLI reference

## `main.py`

Entry point for building synthetic models from the command line.

```
uv run python main.py [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | path | required | JSON config file. |
| `--num_runs` | int | `1` | Number of independent models to generate. |
| `--run_id` | str | `None` | Label appended to the output directory name. |
| `--test_mode` | int | `None` | Shrink the cube to N × N (must be ≥ 50). |
| `--seed` | int | `None` | Fix the random seed for reproducibility. |

---

## `api/server.py`

Uvicorn entry point for the REST API.

```
uv run python -m api.server [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dev` | `false` | Enable uvicorn auto-reload. |
| `--host` | `0.0.0.0` | Bind address. |
| `--port` | `8000` | Bind port. |

:::{note}
Always start with the default `workers=1`. Multi-worker mode breaks the
in-memory process registry used for log streaming and run cancellation.
:::

---

## `scripts/dev.sh`

Start the API and webapp together.

```
./scripts/dev.sh
```

No flags. Press `Ctrl-C` to stop both services.

---

## `scripts/generate_manifest.py`

Scan a project folder and write `manifest.json`.

```
uv run python scripts/generate_manifest.py <project_folder> [--overwrite]
```

| Argument | Description |
|----------|-------------|
| `project_folder` | Root output directory to scan. |
| `--overwrite` | Regenerate `manifest.json` even if it already exists. |
