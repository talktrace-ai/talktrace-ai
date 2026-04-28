#!/usr/bin/env bash
# TalkTrace AI -- developer launcher (hot-reload, browser).
#
# Run from the project folder:
#   ./dev.sh
#
# Edits to any .py file under talktrace_ai/ -> server auto-restarts.
# Press Ctrl+C to stop.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Delegate setup (Python detection, venv creation, deps install, distro
# package install if needed) to start.sh so the logic lives in one place.
"$PROJECT_ROOT/start.sh" --setup-only

VENV_PY="$PROJECT_ROOT/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "[TalkTrace dev] Setup did not produce a venv at $VENV_PY — aborting."
    exit 1
fi

echo "[TalkTrace dev] Project root: $PROJECT_ROOT"
echo "[TalkTrace dev] Hot-reload enabled. Edit any .py file under talktrace_ai/ to trigger a restart."
echo "[TalkTrace dev] Press Ctrl+C to stop."
echo

exec "$VENV_PY" -m shiny run \
    --reload \
    --launch-browser \
    --reload-dir "$PROJECT_ROOT/talktrace_ai" \
    --port 8000 \
    talktrace_ai.app:app
