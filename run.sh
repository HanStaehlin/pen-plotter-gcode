#!/usr/bin/env bash
# Launch the web UI. PenPlotter.app on the Desktop runs this.
cd "$(dirname "$0")"
PY="${PLOTTER_PYTHON:-$HOME/CV/.venv/bin/python3}"
[ -x "$PY" ] || PY="$(command -v python3)"
exec "$PY" app.py
