# Rock-physics models

A rock-physics model (RPM) translates lithology (facies) and depth into elastic
properties: P-wave velocity (Vp), S-wave velocity (Vs), and density (ρ).

## Role in the pipeline

`Geomodels` calls the active RPM once per depth sample to populate the
elastic property volumes. `Seismic` then applies the Zoeppritz equations to
these volumes to compute reflection coefficients.

## How models are selected

The `"project"` key in the config is matched to `rockphysics/rpm_<project>.py`.
The class inside must be named `RPM<Project>` (strip `rpm_`, title-case). The
`Parameters` class handles the dynamic import at runtime.

## What an RPM defines

Each RPM provides static methods that return `numpy` arrays of values as a
function of depth `z` (metres):

```
shale_vp(z)       brine_sand_vp(z)   oil_sand_vp(z)   gas_sand_vp(z)
shale_vs(z)       brine_sand_vs(z)   oil_sand_vs(z)   gas_sand_vs(z)
shale_rho(z)      brine_sand_rho(z)  oil_sand_rho(z)  gas_sand_rho(z)
```

The trends are typically low-order polynomials fit to well-log data from the
target basin. See `rockphysics/rpm_example.py` for the reference
implementation.

## Zoeppritz equations

Reflection coefficients are computed per-interface using the exact Zoeppritz
equations (not the Aki-Richards approximation). The Numba-compiled kernel in
`datagenerator/zoeppritz_kernel.py` vectorises this over all interfaces and
angles simultaneously.

## Adding a new model

See {doc}`../how-to/custom-rpm`.
