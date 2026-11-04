"""Recover Paint in the running Studio; keep saved shape and original log."""
from __future__ import annotations

import json
from pathlib import Path


def retry_texture(engine, job_id=None):
    from unicompose3d.clients import safe_error
    store = engine.store
    jobs = [job for job in store.list("jobs") if job.get("provider") == "hunyuan"
            and job.get("texture") and job.get("status") == "error"
            and (store.path("jobs", job["id"]) / "shape.glb").is_file()]
    if job_id:
        jobs = [job for job in jobs if job["id"] == job_id]
    if not jobs:
        print("No failed texture job with a saved shape.glb was found.")
        return None
    job = jobs[0]; ident = job["id"]; directory = store.path("jobs", ident)
    original = json.loads((directory / "reconstruct-request.json").read_text(encoding="utf-8"))
    front = Path(original["views"]["front"]).resolve()
    if not front.is_relative_to(store.root / "sources") or not front.is_file():
        raise ValueError("The saved front image is no longer valid; retry stopped.")
    output = directory / "texture-result.glb"
    request = {"shape": str(directory / "shape.glb"), "views": {"front": str(front)}, "output": str(output)}
    with engine.lock:
        if engine.active or engine.stopping.is_set():
            raise RuntimeError("Wait for other jobs and keep Studio running before retrying texture.")
        engine.active.add(ident)
    print("Retrying texture only:", job.get("name", ident), "Keeping the existing geometry.", flush=True)
    try:
        store.update("jobs", ident, status="running", progress=70, message="Restoring texture from the saved shape; geometry is not regenerated.")
        engine.quality.run("texture", request, directory, engine.stopping)
        if not output.is_file():
            raise RuntimeError("Paint finished without a result GLB.")
        asset = engine.asset(output, job["name"], source_id=job["source_id"], origin="hunyuan", credits_consumed=0)
        store.update("jobs", ident, status="success", progress=100, asset_id=asset["id"], message="Texture restored with Hunyuan3D. Open the library to inspect it.")
        print("Textured model created. Refresh Studio and open the library:", job["name"], flush=True)
        return asset
    except Exception as error:
        store.update("jobs", ident, status="error", message=safe_error(error) + " The previous shape is retained.")
        log = directory / "texture.log"
        if log.is_file():
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]:
                print(safe_error(line))
        raise
    finally:
        with engine.lock:
            engine.active.discard(ident)
