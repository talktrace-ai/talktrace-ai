#!/usr/bin/env bash
# TalkTrace AI -- install + launch helper for Linux/macOS.
#
# Usage (from this folder):
#   ./start.sh               install (if needed) and start the app
#   ./start.sh --reinstall   force-recreate the virtual environment
#   ./start.sh --nowindow    start the app headless (no desktop window)
#   ./start.sh --setup-only  ensure venv + deps, but do not launch (used by dev.sh)

set -euo pipefail

REINSTALL=""
NOWINDOW=""
SETUP_ONLY=""

while [ $# -gt 0 ]; do
    case "$1" in
        --reinstall|-reinstall|/reinstall)   REINSTALL=1 ;;
        --nowindow|-nowindow|/nowindow)      NOWINDOW=1 ;;
        --setup-only|-setup-only)            SETUP_ONLY=1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
echo "[TalkTrace] Project root: $PROJECT_ROOT"

# --- 1. Locate Python ---------------------------------------------------
PY_CMD=""
for candidate in python3 python python3.13 python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
        # Require Python 3.10+
        ver="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
        major="${ver%%.*}"; minor="${ver##*.}"
        if [ "$major" = "3" ] && [ "${minor:-0}" -ge 10 ] 2>/dev/null; then
            PY_CMD="$candidate"
            break
        fi
    fi
done

if [ -z "$PY_CMD" ]; then
    echo "[TalkTrace] Python 3.10+ not found. Install from https://www.python.org/downloads/ and re-run."
    exit 1
fi
echo "[TalkTrace] Using Python: $PY_CMD ($($PY_CMD --version))"

# --- 2. Virtual environment ---------------------------------------------
VENV_DIR="$PROJECT_ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"

if [ -n "$REINSTALL" ] && [ -d "$VENV_DIR" ]; then
    echo "[TalkTrace] Removing existing venv (--reinstall)..."
    rm -rf "$VENV_DIR"
fi

# On Debian/Ubuntu/Mint, python3 ships without venv/ensurepip by default.
# Detect this up front and offer to install the missing apt packages so the
# next step (`python3 -m venv`) doesn't fail with a cryptic ensurepip error.
if [ ! -x "$VENV_PY" ]; then
    if ! "$PY_CMD" -c 'import ensurepip, venv' >/dev/null 2>&1; then
        echo "[TalkTrace] Python is missing the venv/ensurepip modules."
        if command -v apt >/dev/null 2>&1; then
            echo "[TalkTrace] Installing python3-venv and python3-pip via apt (sudo password required)..."
            sudo apt update
            sudo apt install -y python3-venv python3-pip
        elif command -v dnf >/dev/null 2>&1; then
            echo "[TalkTrace] Installing python3-virtualenv and python3-pip via dnf (sudo password required)..."
            sudo dnf install -y python3-virtualenv python3-pip
        elif command -v pacman >/dev/null 2>&1; then
            echo "[TalkTrace] Installing python-pip via pacman (sudo password required)..."
            sudo pacman -S --noconfirm python-pip
        else
            echo "[TalkTrace] Unknown package manager. Please install the equivalent of"
            echo "[TalkTrace]   python3-venv  python3-pip"
            echo "[TalkTrace] for your distribution and re-run ./start.sh."
            exit 1
        fi
    fi
    echo "[TalkTrace] Creating virtual environment in .venv ..."
    "$PY_CMD" -m venv "$VENV_DIR"
fi

# --- 3. Install dependencies --------------------------------------------
REQ_FILE="$PROJECT_ROOT/requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
    echo "[TalkTrace] requirements.txt not found at $REQ_FILE"
    exit 1
fi

STAMP_FILE="$VENV_DIR/.requirements.sha256"

# sha256: prefer GNU sha256sum (Linux), fall back to shasum -a 256 (macOS).
if command -v sha256sum >/dev/null 2>&1; then
    CURRENT_HASH="$(sha256sum "$REQ_FILE" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
    CURRENT_HASH="$(shasum -a 256 "$REQ_FILE" | awk '{print $1}')"
else
    CURRENT_HASH="no-hash-tool"
fi

STORED_HASH=""
[ -f "$STAMP_FILE" ] && STORED_HASH="$(cat "$STAMP_FILE")"

if [ "$STORED_HASH" != "$CURRENT_HASH" ]; then
    echo "[TalkTrace] Installing/upgrading dependencies ..."
    "$VENV_PY" -m pip install --upgrade pip
    "$VENV_PY" -m pip install -r "$REQ_FILE"
    echo "$CURRENT_HASH" > "$STAMP_FILE"
fi

# --- 4. Launch the app --------------------------------------------------
if [ -n "$SETUP_ONLY" ]; then
    echo "[TalkTrace] Setup complete (--setup-only)."
    exit 0
fi

echo "[TalkTrace] Starting Shiny app ... press Ctrl+C to stop."

if [ -n "$NOWINDOW" ]; then
    exec "$VENV_PY" -c "from talktrace_ai.app import main; main(open_window=False)"
else
    exec "$VENV_PY" -c "from talktrace_ai.app import main; main()"
fi
