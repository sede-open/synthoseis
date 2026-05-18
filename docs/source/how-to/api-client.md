# How to call the API programmatically

The REST API is a standard FastAPI app. Use any HTTP client.

## Submit a run

```python
import httpx, json

config = {
    "project": "example",
    "project_folder": "/tmp/synthoseis_output",
    "work_folder":    "/tmp/synthoseis_work",
    "run_id":         "api_test_01",
    "cube_shape":     [100, 100, 500],
    "incident_angles": [7, 15, 24],
    "digi": 4,
    "infill_factor": 10,
    "initial_layer_stdev": [7.0, 25.0],
    "thickness_min": 2,
    "thickness_max": 12,
    "seabed_min_depth": [20, 50],
    "signal_to_noise_ratio_db": [7.5, 12.5, 17.5],
    "bandwidth_low":  [3.0, 6.0],
    "bandwidth_high": [20.0, 35.0],
    "bandwidth_ord":  4,
    "dip_factor_max": 2,
    "min_number_faults": 1,
    "max_number_faults": 6,
    "pad_samples": 10,
    "max_column_height": [150.0, 150.0],
    "closure_types": ["simple", "faulted", "onlap"],
    "min_closure_voxels_simple":  500,
    "min_closure_voxels_faulted": 2500,
    "min_closure_voxels_onlap":   500,
    "sand_layer_thickness": 2,
    "sand_layer_fraction": {"min": 0.05, "max": 0.25},
}

r = httpx.post("http://localhost:8000/api/runs", json=config)
r.raise_for_status()
run_id = r.json()["run_id"]
print(run_id)
```

## Poll run status

```python
import time

while True:
    r = httpx.get(f"http://localhost:8000/api/runs/{run_id}")
    status = r.json()["status"]
    print(status)
    if status in ("COMPLETE", "FAILED"):
        break
    time.sleep(5)
```

## Stream logs via SSE

```python
with httpx.stream("GET", f"http://localhost:8000/api/runs/{run_id}/logs") as r:
    for line in r.iter_lines():
        if line.startswith("data:"):
            print(line[5:].strip())
```

## Cancel a run

```python
httpx.delete(f"http://localhost:8000/api/runs/{run_id}")
```

## Fetch the manifest for completed runs

```python
r = httpx.get(
    "http://localhost:8000/api/manifest",
    params={"project_folder": "/tmp/synthoseis_output"},
)
manifest = r.json()
for entry in manifest:
    print(entry["run_id"], entry["volumes"])
```

## List available rock-physics models

```python
r = httpx.get("http://localhost:8000/api/models")
print(r.json())   # ["example", ...]
```
