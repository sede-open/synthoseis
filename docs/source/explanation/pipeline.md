# Generation pipeline

A single model passes through these stages in order. Each stage is a Python
class that reads from and writes to the `Parameters` object.

## 1. Parameter setup

`Parameters.__init__` loads the JSON config, generates a random seed
(or uses the one provided), and constructs the directory tree under
`project_folder`.

## 2. Horizon building

`build_unfaulted_depth_maps` deposits random-thickness sedimentary layers from
the base up until the seabed depth threshold is reached. Layer thicknesses are
drawn from a uniform distribution between `thickness_min` and `thickness_max`.

An initial oversampling in Z (`infill_factor`) improves the accuracy of the
horizon geometry before being resampled to the target cube shape.

Fan bodies and onlap surfaces are added at this stage if enabled.

## 3. Facies assignment

`create_facies_array` assigns a facies (shale, sand, or salt) to each voxel
using a Markov chain driven by `sand_layer_fraction` and
`sand_layer_thickness`.

## 4. Unfaulted geomodels

`Geomodel.build_unfaulted_geomodels` converts the facies array into elastic
property volumes (Vp, Vs, ρ) by calling the active rock-physics model at
each depth.

## 5. Fault construction and application

`Faults` generates fault planes using one of four styles — self-branching,
staircase, horst-graben, or relay ramp — and deforms all volumes by
applying the fault displacement field. The number of faults is drawn uniformly
from `[min_number_faults, max_number_faults]`.

## 6. Closure identification

`Closures` runs a flood-fill on the faulted geologic-age volume to identify
structural and stratigraphic traps. Closures above the minimum voxel threshold
are filled randomly with brine, oil, or gas. Smaller closures revert to brine.

## 7. Seismic synthesis

`SeismicVolume`:
1. Computes angle-dependent reflection coefficients at each interface using
   the Zoeppritz equations (Numba-accelerated).
2. Adds random noise at the specified SNR.
3. Convolves with a Butterworth bandpass wavelet to produce band-limited
   reflectivity.
4. Applies geophysical augmentations (lateral smoothing, trace integration,
   amplitude balancing, residual moveout).

Output is one seismic volume per entry in `incident_angles`.

## 8. Output writing

`write_volume_to_zarr` serialises each volume to a chunked Zarr store under
`project_folder`. The store includes array metadata (dtype, shape, chunk
layout) readable by any Zarr v3 client.

## Randomness and reproducibility

All stochastic operations use a single `numpy.random.Generator` seeded from a
`SeedSequence`. Passing `--seed <int>` to `main.py` (or `"seed"` in the API
config) reproduces the exact same model geometry and seismic response.
