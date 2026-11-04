"""Cancellable segmentation jobs; LLM network calls never hold the GPU/commit lock."""
from __future__ import annotations
import copy
import hashlib
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from .clients import ProviderError, safe_error
from .quality import QualityBackend, SEGMENT_SETTINGS
from . import input_processing

STAGES = {"queued": "Waiting for processing", "locate": "LLM is locating the object",
          "gpu_queue": "Waiting for GPU", "sam_load": "Loading or reusing SAM",
          "image_encode": "SAM is encoding or reusing image features", "sam_mask": "SAM is segmenting the object",
          "commit": "Preparing the preview"}


class SegmentCancelled(ProviderError): pass


class StopSignal:
    def __init__(self, *events): self.events = events
    def is_set(self): return any(event.is_set() for event in self.events if event is not None)
    def wait(self, timeout=0):
        end = time.monotonic() + timeout
        while not self.is_set() and time.monotonic() < end: time.sleep(min(.05, max(0, end - time.monotonic())))
        return self.is_set()


class SegmentCoordinator:
    def __init__(self, engine):
        self.engine = engine; self.lock = threading.RLock()
        self.jobs = {}; self.versions = {}
        # Manual work cannot queue behind slow, already cancelled LLM requests.
        self.auto_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="locate")
        self.manual_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="manual-mask")

    def close(self):
        with self.lock:
            for job in self.jobs.values(): job["cancel"].set()
        for pool in (self.auto_pool, self.manual_pool): pool.shutdown(wait=False, cancel_futures=True)
        self.engine.quality.sam_worker.close() if isinstance(self.engine.quality, QualityBackend) else None
        self.engine.quality.hunyuan_worker.close() if isinstance(self.engine.quality, QualityBackend) else None

    def revision(self, ident, filename):
        with self.lock:
            key = (ident, filename); self.versions[key] = self.versions.get(key, 0) + 1
            return self.versions[key]

    def public(self, job):
        return {key: copy.deepcopy(value) for key, value in job.items() if key not in {"cancel", "started"} } | {"elapsed_s": round(time.monotonic() - job["started"], 1)}

    def get(self, ident):
        with self.lock:
            if ident not in self.jobs: raise ValueError("Segmentation job not found.")
            return self.public(self.jobs[ident])

    def cancel(self, ident):
        with self.lock:
            if ident not in self.jobs: raise ValueError("Segmentation job not found.")
            job = self.jobs[ident]
            job["cancel"].set()
            if job["status"] in {"queued", "running"}: job.update(status="cancelled", message="Cancelled. You can draw a box immediately.")
            return self.public(job)

    def start(self, ident, request):
        source = self.engine.store.read("sources", ident)
        if request.filename not in {item["file"] for item in source.get("images", [])}: raise ValueError("The image does not belong to this source.")
        version = self.revision(ident, request.filename)
        with self.lock:
            for job in self.jobs.values():
                if job["source_id"] == ident and job["filename"] == request.filename and job["status"] in {"queued", "running"}:
                    job["cancel"].set(); job.update(status="cancelled", message="Replaced by a newer request.")
            # Bound metadata. Active jobs are never removed.
            for key in list(self.jobs):
                if len(self.jobs) < 128: break
                if self.jobs[key]["status"] not in {"queued", "running"}: self.jobs.pop(key)
            if len(self.jobs) >= 128: raise ProviderError("Too many pending requests; cancel or wait for earlier jobs.")
            job_id = uuid.uuid4().hex
            job = {"id": job_id, "source_id": ident, "filename": request.filename, "status": "queued",
                   "stage": "queued", "message": STAGES["queued"], "timings": {}, "cancel": threading.Event(), "started": time.monotonic()}
            self.jobs[job_id] = job
        def run():
            def progress(stage, timings=None):
                with self.lock:
                    if job["status"] != "cancelled":
                        job.update(status="running", stage=stage, message=STAGES.get(stage, stage))
                        job["timings"].update(timings or {})
            try:
                result = self.execute(ident, request, job["cancel"], progress, version)
                with self.lock:
                    if not job["cancel"].is_set(): job.update(status="success", result=result, message="Segmentation complete. Review the image on the right.", stage="done")
            except SegmentCancelled:
                with self.lock: job.update(status="cancelled", message="Cancelled; the stale result was not applied.")
            except Exception as error:
                with self.lock:
                    if not job["cancel"].is_set(): job.update(status="error", message=safe_error(error))
        pool = self.manual_pool if request.box is not None or request.points else self.auto_pool
        pool.submit(run)
        return self.get(job_id)

    def edit_mask(self, ident, request):
        # Same lock/revision as SAM commits: late model results cannot overwrite a brush edit.
        with self.lock:
            source = self.engine.store.read("sources", ident)
            if request.filename not in {image["file"] for image in source.get("images", [])}:
                raise ValueError("The image does not belong to this source.")
            cutouts = dict(source.get("cutouts", {}))
            current = cutouts.get(request.filename)
            if not current or current["file"] != request.cutout_file:
                raise ValueError("The mask changed. Reopen the current image before editing.")
            history = dict(source.get("cutout_history", {}))
            previous = list(history.get(request.filename, []))
            if request.undo:
                if not previous:
                    raise ValueError("No mask edit is available to undo.")
                item = previous.pop()
            else:
                directory = self.engine.store.path("sources", ident)
                filename = "cut_edit_" + uuid.uuid4().hex + ".png"
                diagnostics = input_processing.brush_mask(directory / request.filename,
                    directory / current["file"], request.strokes, directory / filename)
                item = {**current, **diagnostics, "file": filename,
                        "explanation": "Mask edited on CPU using a region or stroke. Undo is available.", "timings": {}}
                previous = (previous + [current])[-8:]
            self.revision(ident, request.filename)
            for job in self.jobs.values():
                if job["source_id"] == ident and job["filename"] == request.filename and job["status"] in {"queued", "running"}:
                    job["cancel"].set()
                    job.update(status="cancelled", message="Mask edited manually.")
            cutouts[request.filename] = item
            history[request.filename] = previous
            self.engine.store.update("sources", ident, cutouts=cutouts, cutout_history=history)
            return item

    def execute(self, ident, request, cancel=None, progress=None, version=None):
        engine = self.engine
        if not engine.quality.segment_ready: raise ValueError("SAM is not installed. Run the notebook setup cell.")
        version = version or self.revision(ident, request.filename)
        stop = StopSignal(engine.stopping, cancel)
        def abort():
            if stop.is_set() or self.versions.get((ident, request.filename)) != version:
                raise SegmentCancelled("Cancelled or selection changed; stale result discarded.")
        def notify(stage, values=None):
            abort()
            if progress: progress(stage, values or {})
        source = engine.store.read("sources", ident)
        if request.filename not in {item["file"] for item in source.get("images", [])}: raise ValueError("The image does not belong to the selected source.")
        directory = engine.store.path("sources", ident); image = directory / request.filename
        target = request.target.strip()
        if not target: raise ValueError("Describe the object to keep.")
        original_bytes = image.read_bytes()
        signature = getattr(engine.gemini, "locate_signature", None)
        if not isinstance(signature, str): signature = engine.gemini.model + "|locate-v2"
        location_key = hashlib.sha256(original_bytes + (signature + "|" + target).encode()).hexdigest()
        box = request.box; timings = {}; started = time.monotonic()
        explanation = "Using your box without an LLM call."
        if box is None:
            notify("locate"); cached = engine.store.cache_get(location_key)
            if cached: location = cached["location"]; timings["locate_cached"] = True
            else:
                location = engine.gemini.locate(image, target).model_dump(); abort()
                engine.store.cache_put(location_key, {"location": location})
            if not location["found"] or location["confidence"] < .55:
                raise ValueError("The AI could not identify the object. Draw a box: " + location["explanation"])
            box, explanation = location["box"], location["explanation"]
        timings["locate_s"] = round(time.monotonic() - started, 3)
        key = hashlib.sha256(original_bytes + json.dumps([SEGMENT_SETTINGS, "prompt-consistent-v2", target, box, request.points, request.labels], sort_keys=True).encode()).hexdigest()
        filename = "cut_" + key[:24] + ".png"; cached = engine.store.cache_get(key)
        if cached and (directory / filename).is_file():
            item = cached["cutout"]; timings["mask_cached"] = True
        else:
            notify("gpu_queue", timings)
            working = directory / ("segment_" + key[:24]); working.mkdir(exist_ok=True)
            if isinstance(engine.quality, QualityBackend):
                mask = engine.quality.segment(image, box, request.points, request.labels, working, stop, progress=notify)
                timings.update(engine.quality.last_segment_timings.get("timings", {}))
            else: mask = engine.quality.segment(image, box, request.points, request.labels, working, stop)
            abort(); notify("commit", timings)
            diagnostics = input_processing.make_cutout(image, mask, directory / filename)
            item = {"file": filename, "target": target, "box": box, "points": request.points,
                    "labels": request.labels, "explanation": explanation, **diagnostics}
        with self.lock:
            abort()
            engine.store.cache_put(key, {"cutout": item})
            current = engine.store.read("sources", ident)
            cutouts = dict(current.get("cutouts", {})); cutouts[request.filename] = item
            history = dict(current.get("cutout_history", {})); history.pop(request.filename, None)
            engine.store.update("sources", ident, cutouts=cutouts, cutout_history=history)
        timings["total_s"] = round(time.monotonic() - started, 3)
        return {**item, "timings": timings}

