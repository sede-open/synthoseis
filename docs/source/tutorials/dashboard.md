# Tutorial: Interactive dashboard end-to-end

**What you will do:** launch the dashboard, submit a simulation run through the
UI, watch it complete, and explore the output volumes — without touching the
command line again.

**Time:** ~15 minutes (includes a short generation run).

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| Python | ≥ 3.10 |
| [uv](https://docs.astral.sh/uv/) | any recent |
| Node.js | ≥ 18 |

## 1. Start both services

```bash
./scripts/dev.sh
```

The script starts two processes and keeps them running until you press
`Ctrl-C`:

| Process | URL |
|---------|-----|
| FastAPI / uvicorn (REST API) | <http://localhost:8000> |
| Vite dev server (React webapp) | <http://localhost:5173> |

On the first run `dev.sh` runs `npm install` automatically.

## 2. Open the webapp

Navigate to [http://localhost:5173](http://localhost:5173).

You will see the **Runs** panel on the left and the **Launch** panel on the
right.

## 3. Submit a run

In the Launch panel:

1. **Project** — select `example` from the dropdown (auto-detected from
   `rockphysics/`).
2. **Project folder** — click **Browse** and choose a destination directory,
   or type a path directly.
3. **Run ID** — leave blank to auto-generate, or type `tutorial_run`.
4. Accept the default cube shape (`300 × 300 × 1250`) and click **Launch**.

The run appears in the Runs panel with status `RUNNING`.

## 4. Watch the logs

Click the run card. The **Log** tab streams stdout from the subprocess in real
time via Server-Sent Events. You will see horizon construction, fault
application, and seismic convolution phases scroll past.

Status changes to `COMPLETE` when the run finishes (typically 2–5 minutes).

## 5. Explore output volumes

Switch to the **Volumes** tab on the run card:

- Use the **Volume selector** to switch between `seismic_angle_07`,
  `seismic_angle_15`, `geologic_age`, `fault_segments`, etc.
- Drag the **inline / crossline / depth** sliders to move through the cube.
- The **Colormap selector** lets you choose a perceptually-uniform colormap
  (RdBu is the default for seismic).

## 6. Stop the services

Press `Ctrl-C` in the terminal where `dev.sh` is running. Both processes shut
down cleanly.

---

**Next step →** See {doc}`../how-to/custom-rpm` to add your own rock-physics
model.
