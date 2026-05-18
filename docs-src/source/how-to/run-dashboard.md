# How to run the interactive dashboard

## Start both services

From the repo root:

```bash
./scripts/dev.sh
```

The script handles:
- Installing webapp Node dependencies on first run (`npm install`)
- Starting the API (`uv run python -m api.server --dev`) with auto-reload
- Starting the Vite dev server (`npm run dev`)
- Forwarding `Ctrl-C` to both processes

| Service | URL |
|---------|-----|
| API | <http://localhost:8000> |
| Webapp | <http://localhost:5173> |

## Start services individually

If you need to run them in separate terminals:

**API**

```bash
uv run python -m api.server --dev
# or: uv run python api/server.py --dev
```

**Webapp**

```bash
cd webapp
npm install   # first time only
npm run dev
```

## Build a production bundle

```bash
cd webapp && npm run build
```

The compiled output lands in `webapp/dist/`. When `webapp/dist/index.html`
exists, the API automatically serves the frontend at `/` — no separate Node
process needed in production.

```bash
uv run python -m api.server   # serves everything on port 8000
```

## Change the API port

```bash
uv run python -m api.server --port 9000
```

Update `webapp/src/config.ts` to match if running the dev server separately.

## Expose the API to other machines

```bash
uv run python -m api.server --host 0.0.0.0 --port 8000
```

:::{warning}
The API has no authentication. Only expose it on trusted networks.
:::
