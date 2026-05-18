# How to run the test suite

## Run all fast tests

```bash
uv run pytest
```

## Run tests and see verbose output

```bash
uv run pytest -v
```

## Skip slow integration tests

End-to-end generation tests are marked `slow` and are excluded by default via
`pyproject.toml`. To confirm:

```bash
uv run pytest -m "not slow"
```

## Run only slow tests

```bash
uv run pytest -m slow
```

## Run a single test file

```bash
uv run pytest tests/test_api_routes.py -v
```

## Run with coverage

```bash
uv run pytest --cov=datagenerator --cov=api --cov-report=term-missing
```

## Test the API while the server is running

The API tests in `tests/test_api_routes.py` use `httpx.AsyncClient` with the
FastAPI app directly — no live server required.

## Test markers

| Marker | Description |
|--------|-------------|
| `slow` | Full generation runs (minutes each) |

Add `@pytest.mark.slow` to any test that launches a subprocess or generates a
real model.
