# How to run the generator from the CLI

## Basic usage

```bash
uv run python main.py --config <path/to/config.json> --num_runs <N> --run_id <id>
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | yes | Path to a JSON config file |
| `--num_runs` | no (default `1`) | Number of independent models to generate |
| `--run_id` | no | String tag appended to the output directory name |

## Run multiple models sequentially

```bash
uv run python main.py --config config/example.json --num_runs 10 --run_id batch_01
```

Each model gets a unique random seed; outputs land in numbered subdirectories
under `project_folder`.

## Override the output directory at run time

Edit the relevant fields in a copy of the config before running:

```bash
cp config/example.json config/my_run.json
# edit project_folder in my_run.json
uv run python main.py --config config/my_run.json --num_runs 1 --run_id exp_01
```

## Run in test mode (faster, smaller cube)

Pass a positive integer to `--test_mode` to shrink the cube to that number of
samples in X and Y:

```bash
uv run python main.py --config config/example.json --num_runs 1 --test_mode 100
```

:::{warning}
Values below `50` may cause failures — the model needs space to place faults
and closures.
:::

## Reproduce an exact run

Every run writes its seed to `project_folder`. Pass it back via `--seed`:

```bash
uv run python main.py --config config/example.json --num_runs 1 --seed 3141592653
```
