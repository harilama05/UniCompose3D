"""Run locally or from Kaggle. No provider API calls at startup."""
from __future__ import annotations

import argparse
import atexit
import os
import secrets
import socket
import threading
import time
from pathlib import Path

import requests
import uvicorn

from dotenv import load_dotenv

from unicompose3d.api import create_app

HERE = Path(__file__).resolve().parent
load_dotenv(Path.cwd() / ".env", override=False)
_running = None


class RunningStudio:
    def __init__(self, app, server, thread, url, token):
        self.app, self.server, self.thread = app, server, thread
        self.url = url + "/#token=" + token

    def stop(self):
        self.app.state.engine.stopping.set()
        self.app.state.engine.segments.close()
        self.server.should_exit = True
        self.thread.join(timeout=5)
        self.app.state.engine.pool.shutdown(wait=False, cancel_futures=True)


def start(share=True, port=7860, data_dir=None):
    global _running
    if _running is not None and _running.thread.is_alive():
        print("Studio is already running. Reuse this link:")
        print(_running.url)
        return _running
    if not (HERE / "unicompose3d/static/vendor/three.module.js").exists():
        raise RuntimeError("Missing Three.js assets. Run: python tools/prepare_assets.py")
    data_dir = Path(data_dir or os.environ.get("TRYON_DATA_DIR", HERE / "data"))
    token = os.environ.get("TRYON_ACCESS_TOKEN") or secrets.token_urlsafe(32)
    with socket.socket() as check:
        try:
            check.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is already in use. Stop the previous Studio or choose another port.") from exc
    app = create_app(data_root=data_dir, access_token=token)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True, name="unicompose3d-web")
    thread.start()
    local_url = f"http://127.0.0.1:{port}"
    for _ in range(80):
        try:
            if requests.get(local_url + "/health", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(0.1)
    else:
        server.should_exit = True
        raise RuntimeError("Server startup failed. Inspect the messages above.")
    url = local_url
    if share:
        try:
            # Pinned Gradio 5.49.1; this internal function is isolated here so
            # the entire FastAPI/Three.js application shares one notebook tunnel.
            from gradio.networking import setup_tunnel
            url = setup_tunnel(local_host="127.0.0.1", local_port=port, share_token=secrets.token_urlsafe(32), share_server_address=None, share_server_tls_certificate=None)
        except Exception as exc:
            print("Could not create a public link. Check Kaggle Internet access and try studio_share_again().")
            print(f"Error type: {type(exc).__name__}. The local server is still running.")
    _running = RunningStudio(app, server, thread, url, token)
    atexit.register(_running.stop)
    print("UniCompose3D started.")
    print("Open this complete link in a new tab (it grants access to this session):")
    print(_running.url)
    print("No provider calls made. Open the free demo to try the editor.")
    return _running


def studio_share_again():
    if _running is None:
        raise RuntimeError("Run start() first.")
    from gradio.networking import setup_tunnel
    port = _running.server.config.port
    url = setup_tunnel(local_host="127.0.0.1", local_port=port, share_token=secrets.token_urlsafe(32), share_server_address=None, share_server_tls_certificate=None)
    _running.url = url + "/#token=" + _running.app.state.token
    print(_running.url)
    return _running.url


def main():
    parser = argparse.ArgumentParser()
    sharing = parser.add_mutually_exclusive_group()
    sharing.add_argument("--share", action="store_true", help="Create a public notebook tunnel")
    sharing.add_argument("--no-share", action="store_true", help="Keep the server local (default)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    studio = start(share=args.share, port=args.port)
    try:
        while studio.thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        studio.stop()



if __name__ == "__main__":
    main()
