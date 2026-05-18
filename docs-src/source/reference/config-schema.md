# Config schema reference

All generation parameters are set in a JSON file passed to `--config`. The
schema is enforced by `api.config.SimulationConfig` (Pydantic v2).

## Required fields

| Key | Type | Description |
|-----|------|-------------|
| `project` | `string` | Rock-physics model to use. Must match a `rockphysics/rpm_<project>.py` file. |
| `project_folder` | `string` | Output directory. Created if it does not exist. `~` is expanded. |
| `work_folder` | `string` | Scratch directory. Deleted after a successful run. |
| `cube_shape` | `[int, int, int]` | Model dimensions `[X, Y, Z]` in samples. |
| `incident_angles` | `[int, ...]` | Centre angles for the output seismic angle-stacks (1–5 values). |
| `digi` | `int > 0` | Vertical sampling rate (ms). |
| `infill_factor` | `int > 0` | Over-sampling multiplier in Z for initial horizon construction. |
| `initial_layer_stdev` | `[float, float]` | `[low, high]` range for the standard deviation of the base horizon depth. |
| `thickness_min` | `int > 0` | Minimum layer thickness (samples). Must be < `thickness_max`. |
| `thickness_max` | `int > 0` | Maximum layer thickness (samples). |
| `seabed_min_depth` | `[int, int]` | `[low, high]` range for the minimum seabed depth (samples). |
| `signal_to_noise_ratio_db` | `[float, float, float]` | `[left, mode, right]` of a trimmed triangular distribution for SNR (dB). |
| `bandwidth_low` | `[float, float]` | `[low, high]` range for the bandpass low-cut (Hz). |
| `bandwidth_high` | `[float, float]` | `[low, high]` range for the bandpass high-cut (Hz). |
| `bandwidth_ord` | `int > 0` | Butterworth filter order. |
| `dip_factor_max` | `float ≥ 0` | Maximum dip scaling factor applied to layers. |
| `min_number_faults` | `int ≥ 0` | Minimum number of faults. |
| `max_number_faults` | `int ≥ 0` | Maximum number of faults. Must be ≥ `min_number_faults`. |
| `pad_samples` | `int ≥ 0` | Z padding to reduce edge effects (samples). |
| `max_column_height` | `[float, float]` | `[low, high]` range for the maximum hydrocarbon column height (samples). |
| `closure_types` | `["simple" \| "faulted" \| "onlap", ...]` | Closure styles to include (at least one). |
| `min_closure_voxels_simple` | `int > 0` | Closures with fewer voxels than this are filled with brine (simple closures). |
| `min_closure_voxels_faulted` | `int > 0` | Same for faulted closures. |
| `min_closure_voxels_onlap` | `int > 0` | Same for stratigraphic (onlap) closures. |
| `sand_layer_thickness` | `int > 0` | Mean stacked-sand layer thickness (number of layers). |
| `sand_layer_fraction` | `{"min": float, "max": float}` | A priori sand fraction range. Both values must be in `[0, 1]`; `min < max`. |

## Optional fields (with defaults)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `run_id` | `string \| null` | `null` (UUID generated) | Label appended to the output directory. |
| `extra_qc_plots` | `bool` | `true` | Write additional PNG QC images. |
| `verbose` | `bool` | `true` | Enable verbose logging. |
| `partial_voxels` | `bool` | `true` | Average properties in voxels that span multiple layers. |
| `variable_shale_ng` | `bool` | `false` | Allow shale net-to-gross to vary laterally. |
| `basin_floor_fans` | `bool` | `false` | Include basin floor fan bodies. |
| `include_channels` | `bool` | `false` | Include channel features (deprecated; always `false`). |
| `include_salt` | `bool` | `true` | Include salt bodies. |
| `write_to_hdf` | `bool` | `false` | Write QC volumes to an HDF5 file (legacy). |
| `broadband_qc_volume` | `bool` | `false` | Output a broadband (2–90 Hz) QC seismic volume. |
| `model_qc_volumes` | `bool` | `true` | Save QC volumes to disk. |
| `multiprocess_bp` | `bool` | `true` | Use multiprocessing for bandpass operations. |
| `model_store_in_memory` | `bool` | `false` | Build the Zarr store in memory (faster, no disk writes during generation). |
| `cleanup_intermediates` | `bool` | `true` | Delete `work_folder` contents after a successful run. |

## Validation rules

- `thickness_min` must be strictly less than `thickness_max`.
- `min_number_faults` must be ≤ `max_number_faults`.
- All `cube_shape` dimensions must be positive.
- `sand_layer_fraction.min` must be less than `sand_layer_fraction.max`.

## Example

```json
{
  "project": "example",
  "project_folder": "~/synthoseis_output",
  "work_folder":    "~/synthoseis_work",
  "cube_shape": [300, 300, 1250],
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
  "sand_layer_fraction": {"min": 0.05, "max": 0.25}
}
```
