#!/usr/bin/env bash
set -euo pipefail

APP_ID="stereovision-demo"
APP_NAME="Stereovision Demo"
APP_EXEC="StereovisionDemo"
APPIMAGE_NAME="StereovisionDemo-Linux-WebView-x86_64.AppImage"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build_webview_appimage"
APPDIR="$BUILD_DIR/AppDir"
APP_ROOT="$APPDIR/opt/$APP_ID"
APPIMAGE_TOOL="$BUILD_DIR/appimagetool-x86_64.AppImage"
DIST_DIR="$ROOT_DIR/dist"

PYTHON_BIN="/usr/bin/python3"

cd "$ROOT_DIR"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: /usr/bin/python3 was not found."
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
try:
    import gi
    gi.require_version("Gtk", "3.0")
    gi.require_version("WebKit2", "4.1")
except Exception as exc:
    raise SystemExit(
        "ERROR: Python GTK/WebKit bindings are missing.\n"
        "Install them first on Ubuntu 24.04 with:\n\n"
        "sudo apt update\n"
        "sudo apt install -y \\\n"
        "  python3-venv python3-gi python3-gi-cairo \\\n"
        "  gir1.2-gtk-3.0 gir1.2-webkit2-4.1 \\\n"
        "  libwebkit2gtk-4.1-0\n\n"
        f"Original error: {exc}"
    )
print("OK: GTK/WebKit bindings found.")
PY

for required_file in app_sl.py app_local.py desktop_launcher.py requirements_desktop.txt; do
  if [ ! -f "$required_file" ]; then
    echo "ERROR: $required_file not found. Run this script from the repository root."
    exit 1
  fi
done

echo "=== Cleaning previous build ==="
rm -rf "$BUILD_DIR"
mkdir -p "$APP_ROOT" "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/scalable/apps" "$DIST_DIR"

echo "=== Copying application files ==="
copy_if_exists() {
  local path="$1"
  if [ -e "$path" ]; then
    cp -a "$path" "$APP_ROOT/"
  fi
}

copy_if_exists app_sl.py
copy_if_exists app_local.py
copy_if_exists desktop_launcher.py
copy_if_exists requirements_desktop.txt
copy_if_exists documentation_en.md
copy_if_exists documentation_fr.md
copy_if_exists README.md
copy_if_exists LICENSE.txt
copy_if_exists .gitattributes

for optional_dir in assets docs examples default_images; do
  copy_if_exists "$optional_dir"
done

echo "=== Creating embedded Python environment ==="
"$PYTHON_BIN" -m venv --system-site-packages "$APP_ROOT/.venv_webview"
source "$APP_ROOT/.venv_webview/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "$APP_ROOT/requirements_desktop.txt"

echo "=== Writing AppRun ==="
cat > "$APPDIR/AppRun" <<EOF_APPRUN
#!/usr/bin/env bash
set -euo pipefail

HERE="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="\$HERE/opt/$APP_ID"

cd "\$APP_ROOT"

export PYTHONUTF8=1
export STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

exec "\$APP_ROOT/.venv_webview/bin/python" "\$APP_ROOT/desktop_launcher.py"
EOF_APPRUN
chmod +x "$APPDIR/AppRun"

cat > "$APPDIR/usr/bin/$APP_EXEC" <<EOF_EXEC
#!/usr/bin/env bash
exec "\$(dirname "\$0")/../../AppRun"
EOF_EXEC
chmod +x "$APPDIR/usr/bin/$APP_EXEC"

echo "=== Writing desktop file ==="
cat > "$APPDIR/$APP_ID.desktop" <<EOF_DESKTOP
[Desktop Entry]
Type=Application
Name=$APP_NAME
Exec=AppRun
Icon=$APP_ID
Categories=Education;Science;Graphics;
Terminal=false
EOF_DESKTOP

cp "$APPDIR/$APP_ID.desktop" "$APPDIR/usr/share/applications/$APP_ID.desktop"

echo "=== Writing icon ==="
cat > "$APPDIR/$APP_ID.svg" <<'EOF_ICON'
<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
  <rect width="256" height="256" rx="48" fill="#4b3f72"/>
  <circle cx="86" cy="104" r="38" fill="#ffffff" opacity="0.92"/>
  <circle cx="170" cy="104" r="38" fill="#ffffff" opacity="0.92"/>
  <circle cx="86" cy="104" r="16" fill="#4b3f72"/>
  <circle cx="170" cy="104" r="16" fill="#4b3f72"/>
  <path d="M64 174 L128 132 L192 174 Z" fill="#a9def9" opacity="0.95"/>
  <path d="M64 174 H192" stroke="#ffffff" stroke-width="10" stroke-linecap="round"/>
</svg>
EOF_ICON
cp "$APPDIR/$APP_ID.svg" "$APPDIR/.DirIcon"
cp "$APPDIR/$APP_ID.svg" "$APPDIR/usr/share/icons/hicolor/scalable/apps/$APP_ID.svg"

if [ ! -x "$APPIMAGE_TOOL" ]; then
  echo "=== Downloading appimagetool ==="
  if command -v curl >/dev/null 2>&1; then
    curl -L \
      "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage" \
      -o "$APPIMAGE_TOOL"
  elif command -v wget >/dev/null 2>&1; then
    wget \
      "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage" \
      -O "$APPIMAGE_TOOL"
  else
    echo "ERROR: curl or wget is required to download appimagetool."
    exit 1
  fi
  chmod +x "$APPIMAGE_TOOL"
fi

echo "=== Building AppImage ==="
rm -f "$DIST_DIR/$APPIMAGE_NAME"
ARCH=x86_64 "$APPIMAGE_TOOL" "$APPDIR" "$DIST_DIR/$APPIMAGE_NAME"

chmod +x "$DIST_DIR/$APPIMAGE_NAME"

echo ""
echo "Build completed:"
echo "$DIST_DIR/$APPIMAGE_NAME"
echo ""
echo "Run it with:"
echo "chmod +x \"$DIST_DIR/$APPIMAGE_NAME\""
echo "\"$DIST_DIR/$APPIMAGE_NAME\""
