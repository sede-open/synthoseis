# API endpoints reference

Base URL: `http://localhost:8000` (default).

All request/response bodies are JSON. Errors follow the FastAPI default:
`{"detail": "<message>"}`.

---

## Rock-physics models

### `GET /api/models`

Returns the list of available rock-physics model names discovered in
`rockphysics/`.

**Response** `200 OK`

```json
["example", "myproject"]
```

---

## Runs

### `GET /api/runs`

List all simulation runs.

**Response** `200 OK` — array of run objects:

```json
[
  {
    "run_id":        "abc123",
    "status":        "COMPLETE",
    "config":        "{...}",
    "output_folder": "/home/user/synthoseis_output/example/abc123",
    "created_at":    "2024-05-01T12:00:00"
  }
]
```

**Status values:** `RUNNING` | `COMPLETE` | `FAILED`

---

### `POST /api/runs`

Submit a new simulation run.

**Request body** — [SimulationConfig](config-schema.md) JSON object.

**Response** `200 OK`

```json
{"run_id": "abc123", "status": "RUNNING"}
```

The run is launched asynchronously. Poll `GET /api/runs/{run_id}` or stream
`GET /api/runs/{run_id}/logs` to follow progress.

---

### `GET /api/runs/{run_id}`

Get a single run record.

**Response** `200 OK` — run object (see above).  
**Response** `404 Not Found` if the run does not exist.

---

### `DELETE /api/runs/{run_id}`

Cancel a running simulation. Sends `SIGTERM` to the subprocess and marks the
run `FAILED`.

**Response** `200 OK`

```json
{"run_id": "abc123", "status": "FAILED"}
```

---

### `GET /api/runs/{run_id}/logs`

Stream live stdout from the simulation as **Server-Sent Events**.

**Response** `200 OK`, `Content-Type: text/event-stream`

Each event:

```
data: <log line>
```

The stream closes when the subprocess exits.

---

### `GET /api/runs/{run_id}/manifest`

Return the Zarr manifest for a single completed run's output folder.

**Response** `200 OK` — array of manifest entries (see `/api/manifest`).  
**Response** `404 Not Found` if the run or its output folder does not exist.

---

## Manifest

### `GET /api/manifest`

Return the combined manifest for completed runs.

**Query parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `project_folder` | `string` (optional) | Server-side path to scan. `~` is expanded. If omitted, all `COMPLETE` runs in the DB are scanned. |

**Response** `200 OK` — array of manifest entries:

```json
[
  {
    "run_id":  "abc123",
    "volumes": ["seismic_angle_07", "seismic_angle_15", "geologic_age", "..."],
    "path":    "/absolute/path/to/abc123"
  }
]
```

---

## Utilities

### `GET /api/browse-directory`

Open a native OS folder-picker dialog and return the selected path.

**Query parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `initial_dir` | `string` (optional) | Starting directory for the picker. |

**Response** `200 OK`

```json
{"path": "/home/user/selected/folder"}
```

Returns `{"path": null}` if the user cancels. Returns `501 Not Implemented` if
no supported dialog tool is available.

**Platform support:** macOS (osascript), Linux (zenity or kdialog), Windows
(PowerShell FolderBrowserDialog).
