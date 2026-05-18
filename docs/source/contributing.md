# Contributing

## Quick start

```bash
git clone https://github.com/sede-open/synthoseis.git
cd synthoseis
uv sync
uv run pytest        # all fast tests should pass
```

## Branch workflow

1. Fork the repo on GitHub.
2. Create a feature branch: `git checkout -b feat/my-feature`.
3. Make your changes with atomic commits.
4. Open a pull request against `sede-open/synthoseis:master`.

## Code style

- Python: formatted with **ruff** (`uv run ruff format .`).
- Imports: sorted by **ruff** (`uv run ruff check --fix .`).
- TypeScript/React: formatted with **Prettier** (`cd webapp && npm run format`).

## Tests

- Add a test for every bug fix and new feature.
- Place tests in `tests/`.
- Mark any test that generates a real model with `@pytest.mark.slow`.
- CI runs `pytest -m "not slow"`. Slow tests are opt-in.

## Adding a rock-physics model

See {doc}`how-to/custom-rpm`. New RPMs are welcome — include the source
reference (paper or well-log dataset) in the module docstring.

## Documentation

Docs live in `docs-src/`. Build locally:

```bash
cd docs-src
uv pip install -r requirements.txt
make html
open build/html/index.html
```

Edit the `.md` files under `docs-src/source/`. Follow the
[Diataxis](https://diataxis.fr) structure:

- New feature that teaches → `tutorials/`
- Specific task recipe → `how-to/`
- Factual specification → `reference/`
- Background knowledge → `explanation/`

## Reporting issues

Open a GitHub issue with:
- Python version and OS.
- The exact command that failed.
- The full traceback.
- The config JSON (with paths redacted if needed).
