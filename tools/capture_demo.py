"""Capture the actual procedural Studio demo without paid APIs or model inference."""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from launch import start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/studio.png"))
    parser.add_argument("--port", type=int, default=7869)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        with contextlib.redirect_stdout(io.StringIO()):
            studio = start(share=False, port=args.port, data_dir=directory)
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True, args=[
                    "--enable-webgl", "--use-gl=angle", "--use-angle=swiftshader",
                    "--enable-unsafe-swiftshader"])
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(studio.url)
                page.locator("#demoBtn").click()
                page.wait_for_function("!document.querySelector('#exportBtn').disabled")
                page.locator("#instruction").fill("Place the vase on the table.")
                page.locator("#sceneTitle").fill("Table and vase")
                page.locator("#frameBtn").click()
                # Let textures, lighting, and the demo notification settle.
                page.locator("#toast").wait_for(state="hidden", timeout=15000)
                page.screenshot(path=str(args.output), full_page=True)
                browser.close()
                if errors:
                    raise RuntimeError(str(errors))
        finally:
            studio.stop()
    print("Saved procedural demo:", args.output)


if __name__ == "__main__":
    main()
