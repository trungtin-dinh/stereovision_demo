#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="/usr/bin/python3"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: /usr/bin/python3 was not found."
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import gi
gi.require_version("Gtk", "3.0")
gi.require_version("WebKit2", "4.1")
print("OK: GTK/WebKit bindings found.")
PY

if [ ! -d ".venv_webview" ]; then
  "$PYTHON_BIN" -m venv --system-site-packages ".venv_webview"
fi

source ".venv_webview/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements_desktop.txt

python desktop_launcher.py
