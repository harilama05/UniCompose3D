"""Isolated, sequential GPU operations; no model imports in the web server."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from .clients import ProviderError, safe_error

from .resident import GPU_LOCK, ResidentWorker, release_residents
CODE_REVISION = "f8db63096c8282cb27354314d896feba5ba6ff8a"
SHAPE_REVISION = "9cd649ba6913f7a852e3286bad86bfa9a2d83dcf"
MULTIVIEW_REVISION = "3a761b539b29fe4ff64714813aa9560fd66f5de0"
SAM_REVISION = "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f"
QUALITY_SETTINGS = f"hunyuan2|{CODE_REVISION}|{SHAPE_REVISION}|{MULTIVIEW_REVISION}|turbo5|hierarchical384-finite-f32|paint1024-no-delight|seed42|v2"
SEGMENT_SETTINGS = f"sam-vit-base|{SAM_REVISION}|box-points|v1"


def gpu_environment():
    env = dict(os.environ)
    for name in ("TRIPO_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "HF_TOKEN", "TRYON_ACCESS_TOKEN", "PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env.update(USE_TF="0", USE_FLAX="0", PYTHONUNBUFFERED="1")
    return env


class QualityBackend:
    def __init__(self):
        self.python = Path(os.environ.get("TRYON_HUNYUAN_PYTHON", "/kaggle/working/tryon-quality-env/bin/python"))
        self.repository = Path(os.environ.get("TRYON_HUNYUAN_DIR", "/kaggle/working/Hunyuan3D-2"))
        self.weights = Path(os.environ.get("TRYON_QUALITY_WEIGHTS", "/kaggle/working/tryon-quality-weights"))
        self.segment_python = Path(os.getenv("TRYON_SEGMENT_PYTHON", str(self.python)))
        self.segment_weights = Path(os.getenv("TRYON_SEGMENT_WEIGHTS", str(self.weights)))
        self.sam_worker = ResidentWorker(self.segment_python, "sam", self.repository, self.segment_weights)
        self.runner = Path(__file__).resolve().parents[1] / "tools/quality_worker.py"
        self.hunyuan_worker=ResidentWorker(self.python,"hunyuan",self.repository,self.weights)
        self.hunyuan_worker.runner=self.runner

    def _marker(self, name, expected):
        marker = self.weights / name
        return self.python.is_file() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == expected

    @property
    def segment_ready(self):
        marker = self.segment_weights / ".segment-ready"
        return self.segment_python.is_file() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == SEGMENT_SETTINGS and (self.segment_weights / "sam/model.safetensors").is_file()

    @property
    def ready(self):
        return self._marker(".quality-ready", QUALITY_SETTINGS) and (self.repository / "hy3dgen").is_dir()

    def run(self, task, request, directory, stopping, timeout=2400, progress=None):
        if not (self.segment_ready if task == "segment" else self.ready):
            raise ProviderError("SAM/Hunyuan3D is not installed. Enable a GPU and run the notebook setup.")
        if task in {"reconstruct","texture"}:
            return self.hunyuan_worker.call({**request,"_task":task},directory,stopping,progress,timeout=timeout)
        request_path = directory / f"{task}-request.json"
        # Different immutable cutout directories prevent simultaneous edits
        # from overwriting a request or the images used by an active job.
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        command = [str(self.python), "-u", str(self.runner), task, str(request_path), "--repository", str(self.repository), "--weights", str(self.weights)]
        log = directory / f"{task}.log"
        seen_progress = None
        with GPU_LOCK:
            release_residents()
            if stopping.is_set():
                raise ProviderError("Studio stopped before the model ran.")
            with log.open("w", encoding="utf-8") as handle:
                process = subprocess.Popen(command, env=gpu_environment(), stdout=handle, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + timeout
                try:
                    while process.poll() is None:
                        if progress and (directory / "hunyuan-progress.json").is_file():
                            try:
                                value = json.loads((directory / "hunyuan-progress.json").read_text(encoding="utf-8"))
                                if value.get("serial") != seen_progress:
                                    seen_progress = value.get("serial")
                                    progress(value["stage"], value.get("timings", {}))
                            except (OSError, ValueError, KeyError):
                                pass
                        if stopping.wait(0.5) or time.monotonic() > deadline:
                            raise ProviderError("The GPU job stopped or timed out. Previous results are retained.")
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait(timeout=10)
            if process.returncode:
                tail = log.read_text(encoding="utf-8", errors="replace")[-2400:]
                if "out of memory" in tail.lower():
                    raise ProviderError("GPU out of memory. Stop other GPU tasks; disable texture to inspect shape first. No paid API fallback.")
                # Select the END before safe_error truncates the message. The
                # final exception is more useful than the middle of a traceback.
                raise ProviderError("GPU model failed: " + safe_error(tail[-850:]))

    def segment(self, image, box, points, labels, directory, stopping, progress=None):
        if not self.segment_ready:
            raise ProviderError("SAM is not installed. Run setup before segmentation.")
        result = self.sam_worker.call({"image": str(image), "box": box, "points": points, "labels": labels,
            "output": str(directory / "mask.png")}, directory, stopping, progress, timeout=600)
        self.last_segment_timings = result
        return directory / "mask.png"

    def generate(self, views, directory, stopping, texture=True, progress=None):
        self.run("reconstruct", {"views": {k: str(v) for k, v in views.items()}, "output": str(directory / "result.glb"), "texture": texture}, directory, stopping, progress=progress)
        result = directory / "result.glb"
        if not result.is_file():
            raise ProviderError("The model finished without exporting a GLB.")
        return result
