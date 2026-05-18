# Tutorial: Your first synthetic model

**What you will build:** a single 300 × 300 × 1 250-sample seismic cube saved
as a Zarr store, using the bundled example config.

**Time:** ~10 minutes (generation itself takes 2–5 min depending on hardware).

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.10 |
| [uv](https://docs.astral.sh/uv/) | any recent |

## 1. Clone and install

```bash
git clone https://github.com/sede-open/synthoseis.git
cd synthoseis
uv sync
```

`uv sync` reads `pyproject.toml` and installs all dependencies into a local
`.venv`. Nothing is installed globally.

## 2. Inspect the example config

Open `config/example.json`. The two paths you care about:

```json
{
  "project_folder": "~/synthoseis_output",
  "work_folder":    "~/synthoseis_work"
}
```

- **`project_folder`** — where the finished Zarr store lands.
- **`work_folder`** — scratch space, deleted after a successful run.

Both accept `~` and absolute paths. Leave them as-is for now.

## 3. Run the generator

```bash
uv run python main.py \
  --config config/example.json \
  --num_runs 1 \
  --run_id my_first_run
```

You should see structured log output. A successful run ends with a line
containing `COMPLETE`.

## 4. Inspect the output

```
~/synthoseis_output/
└── example/
    └── my_first_run/
        ├── seismic_angle_07.zarr/
        ├── seismic_angle_15.zarr/
        ├── seismic_angle_24.zarr/
        ├── geologic_age.zarr/
        ├── fault_segments.zarr/
        └── ...
```

Each `.zarr` directory is a self-describing, chunked array store. Open one
with [zarr-python](https://zarr.readthedocs.io/):

```python
import zarr
import numpy as np

store = zarr.open("~/synthoseis_output/example/my_first_run/seismic_angle_07.zarr")
print(store.shape)   # (300, 300, 1250)
print(store.dtype)   # float32
```

## 5. View in the dashboard

Once you have output on disk, start the interactive dashboard to browse slices:

```bash
./scripts/dev.sh
```

Open [http://localhost:5173](http://localhost:5173) and point it at
`~/synthoseis_output`.

---

**Next step →** {doc}`dashboard` walks through all the dashboard features.
