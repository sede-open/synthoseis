#!/usr/bin/env bash
# dev.sh — Start the Synthoseis API (Python/uvicorn) and webapp (Vite) together.
# Must be run from the repo root:  ./scripts/dev.sh
# Press Ctrl-C once to stop both processes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Colours ──────────────────────────────────────────────────────────────────
BOLD="\033[1m"; CYAN="\033[36m"; GREEN="\033[32m"; RED="\033[31m"; RESET="\033[0m"

log()  { echo -e "${BOLD}${CYAN}[dev]${RESET} $*"; }
ok()   { echo -e "${BOLD}${GREEN}[dev]${RESET} $*"; }
err()  { echo -e "${BOLD}${RED}[dev]${RESET} $*" >&2; }

# ── Cleanup on exit ───────────────────────────────────────────────────────────
API_PID=""
WEBAPP_PID=""

cleanup() {
  echo ""
  log "Shutting down…"
  [[ -n "$API_PID" ]]   && kill "$API_PID"   2>/dev/null && log "API stopped   (pid $API_PID)"
  [[ -n "$WEBAPP_PID" ]] && kill "$WEBAPP_PID" 2>/dev/null && log "Webapp stopped (pid $WEBAPP_PID)"
  exit 0
}
trap cleanup SIGINT SIGTERM

# ── Preflight checks ──────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  err "'uv' not found. Install it: https://docs.astral.sh/uv/"
  exit 1
fi

if ! command -v npm &>/dev/null; then
  err "'npm' not found. Install Node.js: https://nodejs.org"
  exit 1
fi

if [[ ! -d webapp/node_modules ]]; then
  log "Installing webapp dependencies (npm install)…"
  (cd webapp && npm install)
fi

# ── Start API ─────────────────────────────────────────────────────────────────
log "Starting API  →  http://localhost:8000"
uv run python -m api.server --dev &
API_PID=$!

# ── Start Webapp ──────────────────────────────────────────────────────────────
log "Starting Webapp  →  http://localhost:5173"
(cd webapp && npm run dev) &
WEBAPP_PID=$!

ok "Both services running. Press Ctrl-C to stop."
echo ""

# ── Wait ──────────────────────────────────────────────────────────────────────
# Exit if either process dies unexpectedly
wait -n "$API_PID" "$WEBAPP_PID" 2>/dev/null || true

# One process died — check which one and report
if ! kill -0 "$API_PID" 2>/dev/null; then
  err "API process exited unexpectedly."
fi
if ! kill -0 "$WEBAPP_PID" 2>/dev/null; then
  err "Webapp process exited unexpectedly."
fi

cleanup
