from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def _resource_path(relative_path: str) -> Path:
    return Path(__file__).resolve().parent / relative_path


def _find_free_port(start: int = 8501, end: int = 8599) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free local port found between 8501 and 8599.")


def _wait_for_server(port: int, timeout: float = 20.0) -> None:
    start = time.time()

    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)

    raise RuntimeError(f"Streamlit server did not start on port {port}.")


def _start_streamlit_subprocess(app_path: Path, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode=false",
        "--server.headless=true",
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]

    return subprocess.Popen(
        cmd,
        cwd=str(app_path.parent),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except Exception:
        try:
            process.terminate()
        except Exception:
            pass

    try:
        process.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def main() -> int:
    app_path = _resource_path("app_local.py")
    if not app_path.exists():
        raise FileNotFoundError(f"Cannot find app_local.py at: {app_path}")

    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"

    process = _start_streamlit_subprocess(app_path, port)

    try:
        _wait_for_server(port)

        import webview

        webview.create_window(
            title="Stereovision Demo",
            url=url,
            width=1280,
            height=850,
            min_size=(1000, 700),
            text_select=True,
        )

        webview.start(gui="gtk", debug=False)

    finally:
        _stop_process(process)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
