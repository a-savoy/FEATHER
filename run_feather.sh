#!/bin/bash
# Launch the FEATHER app on macOS/Linux (see run_feather.bat for Windows).
cd "$(dirname "$0")"
exec "$(dirname "$0")/.venv/bin/python" playback_gui.py "$@"
