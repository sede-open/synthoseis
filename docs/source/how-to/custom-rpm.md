# How to add a custom rock-physics model

Synthoseis selects a rock-physics model (RPM) by matching the `"project"` key
in the config file to a Python module under `rockphysics/`.

## 1. Copy the example template

```bash
cp rockphysics/rpm_example.py rockphysics/rpm_myproject.py
```

The file name must start with `rpm_` and end with `.py`. The part after `rpm_`
becomes the project name you use in the config.

## 2. Edit the depth trends

Open `rockphysics/rpm_myproject.py`. Each static method returns a `numpy`
array of values at depths `z` (in metres). Edit the polynomial coefficients to
match your regional trends:

```python
@staticmethod
def calc_shale_vp(z):
    # Replace with your own trend
    return -0.00013 * z**2 + 1.13 * z + 1580

@staticmethod
def calc_shale_rho(z):
    return 7.7e-12 * z**3 + -8.8e-08 * z**2 + 0.0004 * z + 1.957
```

Required methods for each fluid/facies combination:

| Facies | Methods |
|--------|---------|
| Shale | `calc_shale_vp`, `calc_shale_vs`, `calc_shale_rho` |
| Brine sand | `calc_brine_sand_vp`, `calc_brine_sand_vs`, `calc_brine_sand_rho` |
| Oil sand | `calc_oil_sand_vp`, `calc_oil_sand_vs`, `calc_oil_sand_rho` |
| Gas sand | `calc_gas_sand_vp`, `calc_gas_sand_vs`, `calc_gas_sand_rho` |

## 3. Name the class to match the file

The class name inside the file must follow the pattern: strip `rpm_`, title-case
each word. Example:

| File | Class name |
|------|-----------|
| `rpm_myproject.py` | `RPMMyproject` |
| `rpm_north_sea.py` | `RPMNorthSea` |

## 4. Create a config file

```json
{
  "project": "myproject",
  "project_folder": "~/synthoseis_output",
  ...
}
```

Set `"project"` to the name without the `rpm_` prefix.

## 5. Verify discovery

```bash
curl http://localhost:8000/api/models
# → ["example", "myproject"]
```

Or from Python:

```python
import glob, pathlib
stems = [p.stem for p in pathlib.Path("rockphysics").glob("rpm_*.py")]
print([s.removeprefix("rpm_") for s in stems])
```

## 6. Run with the new model

```bash
uv run python main.py --config config/myproject.json --num_runs 1 --run_id test
```
