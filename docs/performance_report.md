# Synthoseis Performance & Resource Analysis

**Generated:** 2026-05-05 23:37:16  
**Config:** `/tmp/synthoseis_test/small_test.json`  
**Mode:** test_mode=70  
**Cube shape (after override):** 70×70×600  
**Total voxels:** 2.94 M  
**Seed:** 99  
**Total wall-clock time:** 194.6 s (3.2 min)  
**Peak RSS:** 1454 MB  

## 1. Pipeline Phase Timing & Memory

| # | Phase | Elapsed (s) | Phase Δ (s) | RSS (MB) | RSS Δ (MB) |
|---|-------|-------------|-------------|----------|------------|
|  0 | START                                       |         0.7 |         0.0 |       36 | +0 |
|  1 | IMPORTS_DONE                                |         2.6 |         1.8 |      217 | +181 |
|  2 | PARAMETERS_SETUP                            |         2.7 |         0.1 |      217 | +0 |
|  3 | UNFAULTED_DEPTH_MAPS                        |       121.7 |       119.0 |      270 | +53 |
|  4 | FACIES_ARRAY                                |       121.7 |         0.0 |      270 | +0 |
|  5 | UNFAULTED_GEOMODELS                         |       122.6 |         0.9 |      810 | +540 |
|  6 | FAULTING_APPLIED                            |       137.2 |        14.6 |      366 | -443 |
|  7 | FAULTED_PROPERTY_GEOMODELS                  |       182.1 |        45.0 |      397 | +31 |
|  8 | GEOLOGY_ZARR_WRITTEN                        |       182.2 |         0.1 |      398 | +1 |
|  9 | CLOSURES_CREATED                            |       185.0 |         2.7 |      577 | +179 |
| 10 | CLOSURES_ZARR_WRITTEN                       |       185.1 |         0.1 |      577 | +0 |
| 11 | SEISMIC_INIT                                |       185.1 |         0.1 |      577 | +0 |
| 12 | ELASTIC_PROPERTIES                          |       191.4 |         6.3 |      587 | +10 |
| 13 | SEISMIC_VOLUMES_BUILT                       |       194.1 |         2.7 |     1336 | +749 |
| 14 | SEISMIC_WRITTEN                             |       194.1 |         0.0 |     1336 | +0 |
| 15 | CLOSURE_LOG_WRITTEN                         |       194.4 |         0.3 |     1064 | -271 |
| 16 | END                                         |       194.6 |         0.2 |     1064 | +0 |

## 2. Memory Profile Summary

| Metric | Value |
|--------|-------|
| Baseline RSS (at START) | 36 MB |
| After imports | 217 MB |
| Import overhead | 181 MB |
| Peak RSS (sampler, 0.5 s interval) | 1454 MB |
| Final RSS | 1064 MB |
| Working-set above baseline | 1418 MB |
| MB per million voxels | 482.4 |

### Top single-phase RSS jumps

| ΔRSS (MB) | Phase |
|-----------|-------|
| +749 | SEISMIC_VOLUMES_BUILT |
| +540 | UNFAULTED_GEOMODELS |
| +181 | IMPORTS_DONE |
| +179 | CLOSURES_CREATED |
| +53 | UNFAULTED_DEPTH_MAPS |
| +31 | FAULTED_PROPERTY_GEOMODELS |
| +10 | ELASTIC_PROPERTIES |
| +1 | GEOLOGY_ZARR_WRITTEN |

## 3. Phase Time Share

| Phase | Δ (s) | % of total |
|-------|-------|------------|
| UNFAULTED_DEPTH_MAPS                        |   119.0 |  61.1% ██████████████████████████████ |
| FAULTED_PROPERTY_GEOMODELS                  |    45.0 |  23.1% ███████████ |
| FAULTING_APPLIED                            |    14.6 |   7.5% ███ |
| ELASTIC_PROPERTIES                          |     6.3 |   3.2% █ |
| CLOSURES_CREATED                            |     2.7 |   1.4%  |
| SEISMIC_VOLUMES_BUILT                       |     2.7 |   1.4%  |
| IMPORTS_DONE                                |     1.8 |   0.9%  |
| UNFAULTED_GEOMODELS                         |     0.9 |   0.5%  |
| CLOSURE_LOG_WRITTEN                         |     0.3 |   0.1%  |
| END                                         |     0.2 |   0.1%  |
| CLOSURES_ZARR_WRITTEN                       |     0.1 |   0.1%  |
| PARAMETERS_SETUP                            |     0.1 |   0.1%  |
| GEOLOGY_ZARR_WRITTEN                        |     0.1 |   0.0%  |
| SEISMIC_INIT                                |     0.1 |   0.0%  |
| SEISMIC_WRITTEN                             |     0.0 |   0.0%  |
| FACIES_ARRAY                                |     0.0 |   0.0%  |

## 4. Disk Usage

| Item | MB |
|------|----|
| Work folder before run | 435.5 |
| Work folder after run  | 727.7 |
| This run's output folder | 292.3 |
| Net new data written | 292.3 |
| Disk per million voxels | 99.4 MB/Mvox |

### Per-zarr store sizes (this run)

| Store | MB |
|-------|----|
| `seismicCubes_RFC__7_degrees_normalized_0505_2334.zarr` | 21.88 |
| `seismicCubes_RFC__15_degrees_normalized_0505_2334.zarr` | 21.87 |
| `seismicCubes_cumsum__7_degrees_normalized_0505_2334.zarr` | 21.86 |
| `seismicCubes_cumsum__15_degrees_normalized_0505_2334.zarr` | 21.86 |
| `seismicCubes_cumsum__24_degrees_normalized_0505_2334.zarr` | 21.85 |
| `seismicCubes_RFC__24_degrees_normalized_0505_2334.zarr` | 21.85 |
| `seismicCubes_cumsum_7_degrees_normalized_augmented_0505_2334.zarr` | 20.21 |
| `seismicCubes_cumsum_15_degrees_normalized_augmented_0505_2334.zarr` | 20.21 |
| `seismicCubes_cumsum_24_degrees_normalized_augmented_0505_2334.zarr` | 20.21 |
| `seismicCubes_RFC__7_degrees_0505_2334.zarr` | 10.67 |
| `seismicCubes_RFC__15_degrees_0505_2334.zarr` | 10.67 |
| `seismicCubes_RFC_fullstack_0505_2334.zarr` | 10.67 |
| `seismicCubes_RFC__24_degrees_0505_2334.zarr` | 10.66 |
| `seismicCubes_cumsum__7_degrees_0505_2334.zarr` | 10.63 |
| `seismicCubes_cumsum__15_degrees_0505_2334.zarr` | 10.62 |
| `seismicCubes_cumsum_fullstack_0505_2334.zarr` | 10.62 |
| `seismicCubes_cumsum__24_degrees_0505_2334.zarr` | 10.62 |
| `geologic_age.zarr` | 9.93 |
| `depth_maps.zarr` | 1.43 |
| `depth_maps.zarr` | 1.43 |
| `depth_maps_gaps.zarr` | 1.42 |
| `faulted_lithology.zarr` | 0.46 |
| `depth_maps_onlaps.zarr` | 0.15 |
| `reservoir_label_0505_2334.zarr` | 0.10 |
| `sealed_label_0505_2334.zarr` | 0.09 |
| `all_closure_segments_0505_2334.zarr` | 0.02 |
| `trap_label_0505_2334.zarr` | 0.00 |
| `closure_segments_hc_voxelcount_0505_2334.zarr` | 0.00 |
| `hc_labels.zarr` | 0.00 |
| `gas.zarr` | 0.00 |
| `hc_closures_augmented_0505_2334.zarr` | 0.00 |
| `oil.zarr` | 0.00 |
| `brine.zarr` | 0.00 |

## 5. RSS Timeline (sampled every 0.5 s)

```
    t(s)    RSS(MB)  bar (max=1454 MB)
     0.0         34  █
     1.5        111  ███
     4.1        253  ████████
     7.7        253  ████████
    10.9        253  ████████
    15.3        253  ████████
    19.0        253  ████████
    22.7        253  ████████
    26.5        253  ████████
    29.5        253  ████████
    32.7        253  ████████
    36.5        255  ████████
    40.2        255  ████████
    42.7        255  ████████
    46.5        255  ████████
    50.2        256  ████████
    52.7        258  ████████
    56.5        258  ████████
    60.3        258  ████████
    62.7        258  ████████
    65.3        258  ████████
    69.0        259  ████████
    72.8        265  █████████
    76.6        272  █████████
    78.5        270  █████████
    82.7        268  █████████
    86.5        264  █████████
    90.3        272  █████████
    92.7        277  █████████
    95.9        270  █████████
    99.0        276  █████████
   100.9        273  █████████
   104.0        268  █████████
   107.8        278  █████████
   110.2        275  █████████
   112.7        272  █████████
   115.2        269  █████████
   118.9        270  █████████
   120.9        280  █████████
   122.4        781  ██████████████████████████
   123.9       1373  ███████████████████████████████████████████████
   125.4        925  ███████████████████████████████
   126.9        663  ██████████████████████
   128.4        386  █████████████
   129.9        386  █████████████
   131.4        386  █████████████
   133.0        340  ███████████
   134.5        340  ███████████
   136.0        340  ███████████
   137.8        368  ████████████
   139.8        368  ████████████
   143.5        368  ████████████
   147.6        368  ████████████
   150.9        368  ████████████
   153.9        368  ████████████
   156.5        368  ████████████
   160.2        368  ████████████
   162.8        368  ████████████
   166.6        368  ████████████
   170.4        368  ████████████
   172.9        368  ████████████
   175.4        368  ████████████
   179.4        368  ████████████
   182.3        398  █████████████
   183.9        548  ██████████████████
   185.4        637  █████████████████████
   186.9        571  ███████████████████
   188.4        493  ████████████████
   190.0        496  █████████████████
   191.5        618  █████████████████████
   193.0       1454  ██████████████████████████████████████████████████
   194.5       1064  ████████████████████████████████████
```

## 6. Observations & Recommendations

### Key findings

- **Slowest pipeline phase:** `UNFAULTED_DEPTH_MAPS` — 119.0 s (61% of total runtime)
- **Largest RSS jump:** `SEISMIC_VOLUMES_BUILT` — +749 MB
- **Peak working set:** 1418 MB above baseline ≈ **482.4 MB per million voxels**
- **Output disk:** 292.3 MB for 2.94 Mvox ≈ **99.4 MB/Mvox**
- **Import footprint:** 181 MB (model-independent overhead)

### Scaling projections

Assuming observed MB/Mvox and s/Mvox rates hold linearly:

| Cube | Voxels (M) | Est. Peak RAM (MB) | Est. Runtime (s) |
|------|-----------|---------------------|------------------|
| 50×50×600 | 1.50 | 760 | 99 |
| 70×70×600 | 2.94 | 1454 | 195 |
| 100×100×600 | 6.00 | 2931 | 397 |
| 150×150×600 | 13.50 | 6549 | 894 |
| 300×300×1250 | 112.50 | 54309 | 7447 |

### Recommendations

- ⚠️  **`UNFAULTED_DEPTH_MAPS`** dominates runtime. Consider profiling inner loops for vectorisation opportunities.
- The model-independent import cost (~181 MB RSS, ~2 s) suggests heavy module-level initialisation in `Closures` and `Faults`. Lazy imports could reduce cold-start time.
- Output is **99.4 MB/Mvox**. Zarr chunk tuning and lossless compression (e.g. Blosc-Zstd) could reduce this by 30–60%.
