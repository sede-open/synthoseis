# How to run large batches

## CLI batch (simplest)

```bash
uv run python main.py \
  --config config/example.json \
  --num_runs 50 \
  --run_id batch_01
```

Runs are sequential in a single process. Each gets an independent random seed.

## Parallel batches via shell

Spawn multiple processes that write to the same `project_folder`:

```bash
for i in 1 2 3 4; do
  uv run python main.py \
    --config config/example.json \
    --num_runs 25 \
    --run_id "worker_${i}" &
done
wait
echo "All done"
```

Outputs are isolated by `run_id`; there are no shared write paths between
worker processes.

## Via the API (dashboard-compatible)

Submit runs in a loop and track them through the REST API. Runs are logged in
`runs.db` and visible in the dashboard.

```python
import httpx, time

BASE = "http://localhost:8000"
base_config = { ... }   # your config dict (see api-client.md)

run_ids = []
for i in range(10):
    cfg = {**base_config, "run_id": f"batch_api_{i:03d}"}
    r = httpx.post(f"{BASE}/api/runs", json=cfg)
    run_ids.append(r.json()["run_id"])

# Wait for all to finish
pending = set(run_ids)
while pending:
    for rid in list(pending):
        status = httpx.get(f"{BASE}/api/runs/{rid}").json()["status"]
        if status in ("COMPLETE", "FAILED"):
            print(rid, status)
            pending.discard(rid)
    time.sleep(10)
```

:::{note}
The API server runs with `workers=1`. Runs are launched as subprocesses, so
CPU parallelism is still available — the single-worker constraint only affects
the HTTP process registry.
:::

## Regenerate the manifest after a batch

```bash
uv run python scripts/generate_manifest.py ~/synthoseis_output
```

Or via the API:

```python
httpx.get(
    "http://localhost:8000/api/manifest",
    params={"project_folder": "~/synthoseis_output"},
)
```
