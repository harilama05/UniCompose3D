from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import io
import json
import logging
import os
import secrets
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image

from . import __version__, composition, geometry, media, aperture, input_processing
from .clients import Gemini, ProviderError, safe_error
from .openai_client import make_llm
from .segmentation import SegmentCoordinator
from .mesh_analysis import geometry_profile, PROFILE_VERSION
from .schemas import AnchorRequest, Capture, ComposeRequest, DeformRequest, EvaluateRequest, GenerationRequest, PlacementPlan, ApertureFitRequest, SceneRequest, SegmentRequest, MaskEditRequest, CutoutExportRequest
from .storage import Store
from .quality import QualityBackend, QUALITY_SETTINGS, SEGMENT_SETTINGS

STATIC = Path(__file__).parent / "static"
logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, store, gemini=None):
        self.store = store
        self.gemini = gemini or make_llm()
        self.quality = QualityBackend()
        self.segments = SegmentCoordinator(self)
        self.segmentation_lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="unicompose3d")
        self.active = set()
        self.lock = threading.RLock()
        self.composition_lock = threading.RLock()
        self.stopping = threading.Event()
        for task in store.list("composition_jobs"):
            if task.get("status") in {"queued", "running"}:
                store.update("composition_jobs", task["id"], status="paused",
                    message="The previous session stopped. Reselect A/B and compose again. LLM calls are not retried automatically.")
        # A kernel restart must never resubmit a paid task automatically.
        for job in store.list("jobs"):
            if job.get("status") in {"uploading", "submitting", "queued", "running"}:
                store.update("jobs", job["id"], status="paused", message="The previous session stopped. Reuse the saved model or restart Hunyuan reconstruction after setup.")

    def asset(self, path, name, **metadata):
        ident, directory = self.store.create("assets", {"name": name, "status": "processing", **metadata})
        try:
            info = geometry.normalize(path, directory / "model.glb")
        except Exception:
            self.store.update("assets", ident, status="error")
            raise
        return self.store.update("assets", ident, status="ready", **info)

    def model_path(self, ident):
        meta = self.store.read("assets", ident)
        if meta.get("status") != "ready":
            raise ValueError("The model is not ready.")
        path = self.store.path("assets", ident) / "model.glb"
        if not path.is_file():
            raise ValueError("The model GLB is missing from this session. Reimport it or restore the project.")
        return path

    def inspect_asset(self, ident):
        meta = self.store.read("assets", ident)
        if (meta.get("geometry_profile") or {}).get("version") == PROFILE_VERSION:
            return meta
        profile = geometry_profile(geometry.load_scene(self.model_path(ident)))
        return self.store.update("assets", ident, geometry_profile=profile)

    def generation_views(self, request):
        source = self.store.read("sources", request.source_id)
        available = {item["file"] for item in source.get("images", [])}
        selected = request.views or ({"front": source["images"][0]["file"]} if source.get("kind") == "image" else {})
        self.validate_selected_images(request, selected, available)
        if "front" not in selected or not 1 <= len(selected) <= 4:
            raise ValueError("Assign a front image. Multi-view/video input requires at least two confirmed views.")
        if len(set(selected.values())) != len(selected):
            raise ValueError("Do not assign the same image to multiple views.")
        for view, filename in selected.items():
            if view not in {"front", "left", "back", "right"} or filename not in available:
                raise ValueError("Invalid view or image filename.")
        directory = self.store.path("sources", request.source_id)
        return self.prepared_views(request, {view: directory / filename for view, filename in selected.items()})

    def validate_selected_images(self, request, views, available):
        selected = request.selected_images or list(views.values())
        if len(set(selected)) != len(selected) or not set(selected) <= available:
            raise ValueError("Invalid selected-image list.")
        if set(selected) != set(views.values()):
            raise ValueError("Not all selected images are assigned. This backend requires up to four explicit views; no images are dropped silently.")

    def prepared_views(self, request, views):
        if not request.use_cutouts:
            raise ValueError("This backend requires correctly segmented objects. Review every selected view first.")
        if not request.use_cutouts:
            return views
        if not request.confirm_cutouts:
            raise ValueError("Confirm that you reviewed the segmentation in every selected view.")
        source = self.store.read("sources", request.source_id)
        cutouts = source.get("cutouts", {})
        result = {}
        for view, path in views.items():
            item = cutouts.get(path.name)
            if not item or item["target"] != request.target.strip():
                raise ValueError("Missing target segmentation in view " + view + ". Segment again and review the preview.")
            if request.cutout_files.get(path.name) != item["file"]:
                raise ValueError("Segmentation changed after review. Inspect and confirm it again.")
            result[view] = path.parent / item["file"]
        return result

    def segment_source(self, ident, request):
        return self.segments.execute(ident, request)





    def generate_quality(self, request):
        if not self.quality.ready:
            raise ValueError("Hunyuan3D is not installed. Enable a GPU and run setup, or import an existing GLB.")
        views = self.generation_views(request)
        digest = hashlib.sha256((QUALITY_SETTINGS + "|texture=" + str(request.texture)).encode())
        for view, path in sorted(views.items()):
            digest.update(view.encode()); digest.update(path.read_bytes())
        key = digest.hexdigest()
        with self.lock:
            if not request.force_new:
                cached = self.store.cache_get(key)
                if cached and cached.get("job_id"):
                    job = self.store.read("jobs", cached["job_id"])
                    if job["status"] in {"queued", "running", "success"}:
                        return {**job, "reused": True}
            ident, _ = self.store.create("jobs", {"provider": "hunyuan", "model": "Hunyuan3D-2mv-Turbo" if len(views) > 1 else "Hunyuan3D-2-Turbo",
                "status": "queued", "progress": 0, "name": request.name, "source_id": request.source_id,
                "cache_key": key, "estimated_credits": 0, "credits_consumed": 0, "texture": request.texture,
                "input_views": list(views), "input_count": len(views), "used_images": len(views),
                "selected_images": request.selected_images or list(request.views.values()),
                "input_images": [path.name for path in views.values()], "quality_verified": False,
                "generation_request": request.model_dump(exclude={"force_new"}),
                "target": request.target, "message": "Waiting for GPU: using the segmented images."})
            self.store.cache_put(key, {"job_id": ident})
            self.active.add(ident)
            self.pool.submit(self._quality_worker, ident, views, request.texture)
            return self.store.read("jobs", ident)

    def _quality_worker(self, ident, views, texture):
        directory = self.store.path("jobs", ident)
        try:
            self.store.update("jobs", ident, status="running", progress=10, message="Hunyuan3D Turbo: five shape steps followed by texture. Multi-view mode may need its checkpoint; previous results are retained.")
            def progress(stage, timings):
                labels = {"weights":(5,"Checking weights"),"shape_load":(10,"Loading shape model"),
                    "shape":(20,"Generating shape in five steps"),"decode":(40,"Decoding mesh"),
                    "memory_retry":(20,"Low VRAM: switching to memory-saving mode at the same resolution"),
                    "paint_load":(55,"Loading texture model"),"uv_unwrap":(60,"Unwrapping UVs on CPU"),
                    "paint_normals":(65,"Rendering normals on GPU"),"paint_positions":(68,"Rendering positions"),
                    "paint_diffusion":(72,"Multi-view texture generation"),"paint_bake":(82,"Baking color onto the mesh"),
                    "paint_inpaint":(88,"Finishing texture regions"),"export":(92,"Exporting GLB"),"done":(94,"Preparing the library")}
                percent, text = labels.get(stage,(10,stage))
                self.store.update("jobs", ident, progress=percent, message="Hunyuan3D · "+text+(" · "+str(timings["uv_patches_done"])+"/"+str(timings["uv_patches_total"])+" patches" if stage=="uv_unwrap" and "uv_patches_done" in timings else ""), timings=timings)
            kwargs = {"texture":texture}
            if isinstance(self.quality, QualityBackend):
                kwargs["progress"] = progress
            path = self.quality.generate(views, directory, self.stopping, **kwargs)
            progress("done", self.store.read("jobs", ident).get("timings", {}))
            normalized_at = time.monotonic()
            job = self.store.read("jobs", ident)
            asset = self.asset(path, job["name"], source_id=job["source_id"], origin="hunyuan", credits_consumed=0)
            diagnostic_path = directory / "hunyuan-diagnostics.json"
            diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8")) if diagnostic_path.is_file() else {}
            timings = diagnostics.get("timings", {})
            timings["studio_normalize_s"] = round(time.monotonic() - normalized_at, 3)
            self.store.update("jobs", ident, status="success", progress=100, asset_id=asset["id"], timings=timings,
                message="Hunyuan3D GLB created. Inspect geometry and details before composing.")
        except Exception as exc:
            extra = {}
            message = safe_error(exc)
            if (directory / "shape.glb").is_file():
                try:
                    job = self.store.read("jobs", ident)
                    asset = self.asset(directory / "shape.glb", job["name"] + " · no texture", origin="hunyuan-shape", source_id=job["source_id"])
                    extra["asset_id"] = asset["id"]
                    message += " The shape was saved in the library without texture."
                except Exception:
                    logger.exception("Could not preserve partial shape")
            self.store.update("jobs", ident, status="error", message=message, **extra)
        finally:
            with self.lock:
                self.active.discard(ident)

    def generate(self, request):
        return self.generate_quality(request)

    def placement_key(self, request):
        return hashlib.sha256(json.dumps(["placement-spatial-program-9-budgeted-hypotheses",
            self.gemini.__class__.__name__,request.base_id,request.product_id,request.base_transform.model_dump(),
            request.instruction,request.hints.model_dump(),self.gemini.model,request.refinement_of,request.clarification_of,request.clarification_answer,request.search_budget.model_dump()],
            ensure_ascii=False).encode()).hexdigest()

    def placement_plan(self, request):
        from .agent_loop import parent_state
        parent = parent_state(self,request)
        a = self.inspect_asset(request.base_id)
        b = self.inspect_asset(request.product_id)
        key = self.placement_key(request)
        cached = self.store.cache_get(key) if request.use_cache else None
        if cached:
            plan = PlacementPlan.model_validate(cached["plan"])
        else:
            if len(request.captures) < 2 or {c.slot for c in request.captures} != {"A", "B"}:
                raise ValueError("Render both models before calling the LLM.")
            # Only geometry descriptors and names are sent, never disk paths or keys.
            def descriptor(meta, slot):
                value = {k: meta.get(k) for k in ("name", "bounds", "extents", "faces", "geometry_profile")}
                value["user_hints"] = {"point_local": getattr(request.hints, slot + "_point"), "axis_end_local": getattr(request.hints, slot + "_end")}
                from .clarification import answers
                value["user_answers"]=answers(self,request)[0]
                from .spatial_context import catalog
                value["landmarks"]=catalog(self.model_path(request.base_id if slot=="A" else request.product_id))
                value["search_budget"]=request.search_budget.model_dump()
                value["scene_transform"]=request.base_transform.model_dump() if slot=="A" else "to be solved"
                value["coordinate_frame"]={"units":"independently normalized mesh; point/end are local bbox fractions","Y_up":True,"landmark_points":"actual local mesh XYZ","view_directions":"camera-defined, not semantic object front"}
                return value
            from .spatial_context import label_captures
            llm_captures=label_captures({"A":self.model_path(request.base_id),"B":self.model_path(request.product_id)},request.captures)
            if parent:
                from .agent_loop import feedback
                previous, rendered = feedback(self,parent,request.captures)
                plan = self.gemini.revise(descriptor(a,"A"),descriptor(b,"B"),
                    request.instruction,llm_captures,previous,rendered)
            else:
                plan = self.gemini.plan(descriptor(a, "A"), descriptor(b, "B"), request.instruction, llm_captures)
            # Captures must be cached with the plan: UVs depend on these cameras.
            self.store.cache_put(key, {"plan": plan.model_dump(), "cameras": [c.model_dump(exclude={"image"}) for c in request.captures]})
        captures = request.captures
        if cached:
            captures = [Capture(image="", **c) for c in cached.get("cameras", [])]
        return plan, captures, bool(cached)

    def compose(self, request):
        # Explicit single-pass baseline, retained for comparison.
        plan, captures, cached = self.placement_plan(request)
        if plan.clarification is not None:return self.prepare_composition(request,planned=(plan,captures,cached))
        candidates, diagnostics = composition.build_candidates(self.model_path(request.base_id),
            self.model_path(request.product_id),plan,captures,request.base_transform,1,request.hints)
        candidate=candidates[0]
        return {"plan":plan.model_dump(),"transform":candidate["transform"],"diagnostics":diagnostics,
                "eligible":candidate["eligible"],"failures":candidate["failures"],"cached":bool(cached)}

    def prepare_composition(self, request, planned=None):
        from .agent_loop import parent_state, history_entry
        parent = parent_state(self,request)
        canonical = request.model_dump(exclude={"captures", "use_cache"})
        key = hashlib.sha256(json.dumps([composition.ALGORITHM_VERSION, self.gemini.model, canonical], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        with self.composition_lock:
            if request.use_cache:
                cached = self.store.cache_get(key)
                if cached and cached.get("composition_id"):
                    try:
                        saved = self.store.read("compositions", cached["composition_id"])
                    except FileNotFoundError:
                        self.store.cache_put(key, {})
                    else:
                        if not any(saved.get("diagnostics", {}).get(kind) for kind in ("feature_error","scale_error","goal_error")):
                            return {**saved, "reused": True}
            base_path, product_path = self.model_path(request.base_id), self.model_path(request.product_id)
            plan, captures, plan_cached = planned or self.placement_plan(request)
            from .clarification import answers
            user_answers,question_calls=answers(self,request)
            question_ids=list(parent.get("question_call_ids",[])) if parent else []
            if request.clarification_of in question_ids:question_calls=0
            elif request.clarification_of:question_ids.append(request.clarification_of)
            def save_question(asked,spent,contract=None):
                if len(user_answers)>=3:raise ValueError("The agent still lacks information after three questions. Clarify the intended composition.")
                ident,_=self.store.create("compositions",{
                    "status":"needs_input","question":asked.clarification.model_dump(),
                    "plan":asked.model_dump(),"base_id":request.base_id,"product_id":request.product_id,
                    "base_transform":request.base_transform.model_dump(),"instruction":request.instruction,
                    "hints":request.hints.model_dump(),"clarification_answers":user_answers,
                    "goal_contract":contract,"question_call_ids":question_ids,
                    "api_calls":{"plan":question_calls+spent,"evaluation":0},
                    "candidates":[],"agent_round":int(parent.get("agent_round",1))+1 if parent else 1,
                    "parent_id":parent["id"] if parent else None,
                    "agent_history":parent.get("agent_history",[])+[history_entry(parent)] if parent else [],
                    "summary":asked.clarification.question,"warnings":[],"auto_apply":False,
                    "needs_review":True,"evaluation_status":"awaiting_user"})
                self.store.cache_put(key,{"composition_id":ident})
                return self.store.read("compositions",ident)
            if plan.clarification is not None:
                return save_question(plan,0 if plan_cached else getattr(plan,"_provider_calls",1))
            from .goal_contract import inherited,derive
            question_parent=self.store.read("compositions",request.clarification_of) if request.clarification_of else None
            contract=inherited(self,parent) or (question_parent or {}).get("goal_contract") or derive(plan)
            contract.setdefault("original_instruction",request.instruction)
            candidates, diagnostics = composition.build_candidates(base_path, product_path, plan, captures, request.base_transform, request.candidate_count, request.hints,goal_contract=contract,budget=request.search_budget.model_dump())
            calls = 0 if plan_cached else getattr(plan, "_provider_calls", 1)
            # Retry a completed but ungrounded plan once using measured geometry
            # feedback, not a blind repeat of the same cached UV coordinates.
            extraction_error = diagnostics.get("feature_error") or diagnostics.get("scale_error") or diagnostics.get("goal_error")
            if extraction_error and isinstance(self.gemini, Gemini) and calls < 2:
                feedback = ("Geometry engine could not resolve your previous plan: " +
                    extraction_error + ". Correct the landmarks/view indices using the same renders. "
                    "An aperture centre lies in empty space and is verified geometrically. "
                    "User landmarks override predicted UVs. Do not invent missing topology. Previous plan: " +
                    plan.model_dump_json() + "\nOriginal user instruction: " + request.instruction + "\nGeometry tool observations (authoritative measurements): " + json.dumps(diagnostics,ensure_ascii=False))
                a, b = self.inspect_asset(request.base_id), self.inspect_asset(request.product_id)
                try:
                    calls += 1
                    repaired = self.gemini.plan(
                        {k:a.get(k) for k in ("name","bounds","extents","geometry_profile")} |
                            {"user_hints":{"point_local":request.hints.A_point,"axis_end_local":request.hints.A_end},"user_answers":user_answers},
                        {k:b.get(k) for k in ("name","bounds","extents","geometry_profile")} |
                            {"user_hints":{"point_local":request.hints.B_point,"axis_end_local":request.hints.B_end},"user_answers":user_answers},
                        feedback, request.captures)
                    calls += getattr(repaired, "_provider_calls", 1) - 1
                    if repaired.clarification is not None:
                        return save_question(repaired,calls,contract)
                    previous_attempt={"plan":plan.model_dump(),"diagnostics":copy.deepcopy(diagnostics)}
                    plan, captures = repaired, request.captures
                    candidates, diagnostics = composition.build_candidates(base_path, product_path, plan, captures,
                        request.base_transform, request.candidate_count, request.hints,goal_contract=contract,budget=request.search_budget.model_dump())
                    diagnostics["previous_attempts"]=[previous_attempt]
                    diagnostics["grounding_repair_attempted"] = True
                    if not (diagnostics.get("feature_error") or diagnostics.get("scale_error") or diagnostics.get("goal_error")):
                        self.store.cache_put(self.placement_key(request), {"plan":plan.model_dump(),
                            "cameras":[capture.model_dump(exclude={"image"}) for capture in captures]})
                except ProviderError as error:
                    diagnostics["grounding_repair_error"] = safe_error(error)
            initial = composition.rank_candidates(candidates, plan_confidence=plan.confidence)
            no_change = False
            if parent:
                import numpy as np
                old = next((candidate for candidate in parent["candidates"] if candidate["id"]==(
                    parent.get("selected_candidate_id") or parent.get("suggested_candidate_id"))),None)
                new = next((candidate for candidate in initial["candidates"] if candidate["id"]==initial["suggested_candidate_id"]),None)
                if old and new and old.get("eligible") and new.get("eligible"):
                    no_change=all(np.allclose(old["transform"][key],new["transform"][key],rtol=0,atol=1e-5)
                                  for key in ("position","rotation","scale"))
                if no_change:
                    initial["summary"]="The agent proposed no concrete pose or scale change from the previous round."
                    initial["warnings"].append(initial["summary"])
                    initial["needs_review"],initial["auto_apply"]=True,False
            ident, _ = self.store.create("compositions", {
                "algorithm": composition.ALGORITHM_VERSION, "status": "prepared", "model": self.gemini.model,
                "agent_round":int(parent.get("agent_round",1))+1 if parent else 1,
                "agent_no_change":no_change,
                "goal_contract":contract,
                "parent_id":parent["id"] if parent else None,
                "agent_history":parent.get("agent_history",[])+[history_entry(parent)] if parent else [],
                "hints":request.hints.model_dump(),
                "base_id": request.base_id, "product_id": request.product_id, "base_transform": request.base_transform.model_dump(), "instruction": request.instruction,
                "objects": {"A": self.store.read("assets", request.base_id)["name"], "B": self.store.read("assets", request.product_id)["name"]},
                "plan": plan.model_dump(), "diagnostics": diagnostics, "plan_cached": plan_cached,
                "api_calls": {"plan": calls+question_calls, "evaluation": 0},
                "clarification_answers":user_answers,"question_call_ids":question_ids,"renders": [], **initial, "evaluation_status": "pending",
            })
            self.store.cache_put(key, {"composition_id": ident})
            return self.store.read("compositions", ident)

    def start_composition(self, request):
        # Resolve assets before creating a task so stale selections fail clearly.
        self.model_path(request.base_id); self.model_path(request.product_id)
        ident, _ = self.store.create("composition_jobs", {"status":"queued",
            "message":"Waiting for spatial analysis and composition.", "base_id":request.base_id,
            "product_id":request.product_id})
        def run():
            self.store.update("composition_jobs", ident, status="running",
                message="AI is grounding the target and solving geometric constraints…")
            try:
                result = self.prepare_composition(request)
                self.store.update("composition_jobs", ident, status="success",
                    composition_id=result["id"], message="Composition result is available.")
            except Exception as error:
                self.store.update("composition_jobs", ident, status="error", message=safe_error(error))
        self.pool.submit(run)
        return self.store.read("composition_jobs", ident)

    def start_evaluation(self, ident, request):
        composition_record=self.store.read("compositions",ident)
        task_id,_=self.store.create("composition_jobs",{"status":"queued","action":"evaluate",
            "message":"Agent is reviewing rendered candidates.","composition_id":ident})
        def run():
            self.store.update("composition_jobs",task_id,status="running",
                message="Agent is checking the original instruction, placement, orientation, and scale across views…")
            try:
                self.evaluate_composition(ident,request)
                self.store.update("composition_jobs",task_id,status="success",message="Visual evaluation complete.")
            except Exception as error:
                self.store.update("composition_jobs",task_id,status="error",message=safe_error(error))
        self.pool.submit(run)
        return self.store.read("composition_jobs",task_id)

    def evaluate_composition(self, ident, request):
        with self.composition_lock:
            current = self.store.read("compositions", ident)
            if current.get("evaluation_status") == "completed":
                return {**current, "reused": True}
            if current.get("status")=="needs_input":
                raise ValueError("Answer the clarification before visual evaluation.")
            expected = {c["id"] for c in current["candidates"]}
            received = [r.candidate_id for r in request.renders]
            if len(received) != len(expected) or set(received) != expected:
                raise ValueError("Provide exactly one multi-view sheet for each candidate in this run.")
            directory = self.store.path("compositions", ident)
            images = []
            for render in request.renders:
                try:
                    raw = base64.b64decode(render.image.split(",", 1)[-1], validate=True)
                    if len(raw) > 2_000_000:
                        raise ValueError("Rendered image exceeds 2 MB.")
                    with Image.open(io.BytesIO(raw)) as image:
                        width, height = image.size
                        if image.format != "PNG" or not (256 <= width <= 1800 and 256 <= height <= 1800):
                            raise ValueError("Evaluation images must be multi-view PNG sheets, 256–1800 pixels per side.")
                        image.verify()
                except (OSError, ValueError) as exc:
                    raise ValueError("Invalid evaluation image: " + safe_error(exc)) from exc
                (directory / f"{render.candidate_id}.png").write_bytes(raw)
                images.append({"candidate_id": render.candidate_id, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "width": width, "height": height, "layout": render.layout})
            calls = dict(current["api_calls"])
            calls["evaluation"] += 1
            self.store.update("compositions", ident, renders=images, api_calls=calls)
            try:
                evaluation = self.gemini.evaluate({**current["plan"],"original_goal_contract":current.get("goal_contract",{})}, current["candidates"], request.renders, current["instruction"])
                result = composition.rank_candidates(current["candidates"], evaluation=evaluation, plan_confidence=current["plan"]["confidence"],semantic_required=bool(current.get("goal_contract",{}).get("intent")))
            except (ProviderError, ValueError) as exc:
                result = composition.rank_candidates(current["candidates"], plan_confidence=current["plan"]["confidence"], error="Visual evaluation unavailable: " + safe_error(exc))
            return self.store.update("compositions", ident, status="complete" if result["evaluation_status"] == "completed" else "evaluation_unavailable", **result)


def create_app(data_root=None, access_token=None, gemini=None):
    store = Store(Path(data_root or os.environ.get("TRYON_DATA_DIR", Path.cwd() / "data")))
    engine = Engine(store, gemini=gemini)
    token = access_token or os.environ.get("TRYON_ACCESS_TOKEN") or secrets.token_urlsafe(32)
    app = FastAPI(title="UniCompose3D", version=__version__)
    app.state.engine = engine
    app.state.token = token

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            received = request.headers.get("Authorization", "").removeprefix("Bearer ")
            if not hmac.compare_digest(received.encode(), token.encode()):
                return JSONResponse({"detail": "The link is missing its access token. Open the full URL printed by the notebook."}, status_code=401)
        response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def bad_value(request, exc):
        return JSONResponse({"detail": safe_error(exc)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Data not found. Reselect the model from the library."}, status_code=404)

    @app.exception_handler(ProviderError)
    async def provider(request, exc):
        return JSONResponse({"detail": safe_error(exc)}, status_code=502)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/config")
    def config():
        llm_name = getattr(engine.gemini, "name", "LLM")
        if not isinstance(llm_name, str): llm_name = "LLM"
        from .spatial_search import configured_budget
        return {"composition_search_budget": configured_budget(), "version": __version__,
            "default_provider": "hunyuan", "hunyuan_ready": engine.quality.ready,
            "segment_ready": engine.quality.segment_ready, "llm_ready": bool(engine.gemini.key),
            "llm_name": llm_name, "llm_model": engine.gemini.model}

    @app.get("/api/assets")
    def assets():
        return [a for a in store.list("assets") if a.get("status") == "ready"]

    @app.get("/api/assets/{ident}/inspect")
    def inspect_asset(ident: str):
        return engine.inspect_asset(ident)

    @app.get("/api/assets/{ident}/model.glb")
    def model(ident: str):
        return FileResponse(engine.model_path(ident), media_type="model/gltf-binary", filename=f"{ident}.glb")

    @app.post("/api/sources")
    async def source(kind: str = Form(...), name: str = Form("Object"), files: list[UploadFile] = File(...)):
        if kind not in {"image", "multiview", "video", "model"} or not 1 <= len(files) <= 16:
            raise ValueError("Invalid input type or file count.")
        if kind in {"video", "model", "image"} and len(files) != 1:
            raise ValueError("This input mode accepts exactly one file.")
        allowed = {"image": {".png", ".jpg", ".jpeg", ".webp"}, "multiview": {".png", ".jpg", ".jpeg", ".webp"}, "video": {".mp4", ".mov", ".webm", ".avi"}, "model": {".glb", ".gltf", ".obj", ".stl", ".ply"}}[kind]
        ident, directory = store.create("sources", {"kind": kind, "name": name[:100]})
        images = []
        for index, upload in enumerate(files):
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in allowed:
                raise ValueError("Unsupported file format for this input mode.")
            target = directory / f"input_{index}{suffix}"
            limit = (20 if kind in {"image", "multiview"} else 150) * 1024 * 1024
            size = 0
            with target.open("wb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise ValueError(f"File exceeds the limit of {limit // 1024 // 1024} MB.")
                    handle.write(chunk)
            await upload.close()
            if kind in {"image", "multiview"}:
                filename = f"image_{index:02d}.png"
                digest = await run_in_threadpool(media.normalize_image, target, directory / filename)
                images.append({"file": filename, "original_name": Path(upload.filename or "image").name[:100], "sha256": digest})
        if kind == "video":
            images = await run_in_threadpool(media.video_frames, target, directory)
        for item in images:
            item["quality"] = await run_in_threadpool(input_processing.image_quality, directory / item["file"])
        if len(images) > 1:
            hashes = [hashlib.sha256((directory / item["file"]).read_bytes()).hexdigest() for item in images]
            for i, item in enumerate(images):
                if hashes[i] in hashes[:i]:
                    item["quality"]["warnings"].append("This image duplicates an earlier image; choose another view.")
        source = store.update("sources", ident, images=images)
        if kind == "model":
            source["asset"] = await run_in_threadpool(engine.asset, target, name[:100], source_id=ident, origin="upload")
        return source

    @app.get("/api/sources/{ident}/{filename}")
    def source_image(ident: str, filename: str):
        value = store.read("sources", ident)
        available = {image["file"] for image in value.get("images", [])} | {image["file"] for image in value.get("cutouts", {}).values()}
        if filename not in available:
            raise HTTPException(404, "Image not found.")
        return FileResponse(store.path("sources", ident) / filename, media_type="image/png")

    @app.post("/api/sources/{ident}/segment-jobs")
    def start_segment(ident: str, request: SegmentRequest):
        return engine.segments.start(ident, request)

    @app.get("/api/segment-jobs/{ident}")
    def segment_status(ident: str):
        return engine.segments.get(ident)

    @app.delete("/api/segment-jobs/{ident}")
    def cancel_segment(ident: str):
        return engine.segments.cancel(ident)

    @app.post("/api/sources/{ident}/segment")
    def segment(ident: str, request: SegmentRequest):
        return engine.segment_source(ident, request)

    @app.post("/api/sources/{ident}/mask-edit")
    def edit_mask(ident: str, request: MaskEditRequest):
        return engine.segments.edit_mask(ident, request)

    @app.post("/api/sources/{ident}/export-cutouts")
    def export_cutouts(ident: str, request: CutoutExportRequest):
        if not request.confirmed:
            raise ValueError("Review and confirm masks for all selected images before exporting.")
        with engine.segments.lock:
            source = store.read("sources", ident)
            available = {item["file"] for item in source.get("images", [])}
            if len(set(request.files)) != len(request.files) or not set(request.files) <= available:
                raise ValueError("Invalid image list.")
            stream = io.BytesIO()
            manifest = {"source_id": ident, "target": request.target.strip(), "images": []}
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                for index, filename in enumerate(request.files):
                    item = source.get("cutouts", {}).get(filename)
                    if not item or item["target"] != request.target.strip() or request.cutout_files.get(filename) != item["file"]:
                        raise ValueError("Masks are missing or changed. Review and confirm the image set again.")
                    name = f"view_{index:02}.png"
                    archive.write(store.path("sources", ident) / item["file"], f"object_{ident}/{name}")
                    manifest["images"].append({"file": name, "original": filename, "cutout": item["file"]})
                archive.writestr(f"object_{ident}/manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        return __import__("fastapi").Response(stream.getvalue(), media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="unicompose3d-cutouts-{ident}.zip"'})

    @app.post("/api/generate")
    def generate(request: GenerationRequest):
        return engine.generate(request)

    @app.get("/api/jobs")
    def jobs():
        return store.list("jobs")


    @app.get("/api/jobs/{ident}/reconstruction.zip")
    def reconstruction_diagnostics(ident: str):
        job = store.read("jobs", ident)
        if job.get("provider") != "hunyuan":
            raise ValueError("This diagnostic archive is for Hunyuan jobs.")
        if job.get("status") in {"queued", "running"}:
            raise ValueError("Wait for the job to finish before downloading diagnostics.")
        source = store.read("sources", job["source_id"])
        directory = store.path("sources", job["source_id"])
        originals = {item["file"] for item in source.get("images", [])}
        selected = job.get("selected_images") or list(job.get("generation_request", {}).get("views", {}).values())
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            fields = ("id", "provider", "model", "source_id", "status", "message", "input_count",
                      "selected_images", "input_images",
                      "quality_verified", "timings", "asset_id")
            manifest = {key: job[key] for key in fields if key in job}
            manifest["note"] = "Original images, masks, and meshes for reconstruction diagnostics; not quality certification."
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for index, name in enumerate(selected):
                if name in originals and Path(name).name == name and (directory / name).is_file():
                    archive.write(directory / name, f"originals/{index:02}_{name}")
            for index, name in enumerate(job.get("input_images", [])):
                if Path(name).name != name:
                    raise ValueError("Invalid job-image filename.")
                path = directory / name
                if path.is_file():
                    archive.write(path, f"cutouts/{index:02}_{name}")
            for name in ("hunyuan-diagnostics.json", "hunyuan-progress.json", "shape.glb", "result.glb"):
                path = store.path("jobs", ident) / name
                if path.is_file():
                    archive.write(path, "native/" + name)
            if job.get("asset_id"):
                archive.write(engine.model_path(job["asset_id"]), "studio/model.glb")
        return __import__("fastapi").Response(stream.getvalue(), media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="reconstruction-{ident}.zip"'})

    @app.get("/api/jobs/{ident}")
    def job(ident: str):
        return store.read("jobs", ident)


    @app.post("/api/compose")
    def compose(request: ComposeRequest):
        return engine.compose(request)

    @app.post("/api/composition-jobs")
    def start_composition_job(request: ComposeRequest):
        return engine.start_composition(request)

    @app.get("/api/composition-jobs/{ident}")
    def composition_job(ident: str):
        task = store.read("composition_jobs", ident)
        if task.get("status") == "success":
            task["result"] = store.read("compositions", task["composition_id"])
        return task

    @app.post("/api/compositions")
    def prepare_composition(request: ComposeRequest):
        return engine.prepare_composition(request)

    @app.post("/api/compositions/{ident}/evaluation-jobs")
    def start_evaluation_job(ident: str, request: EvaluateRequest):
        return engine.start_evaluation(ident,request)

    @app.post("/api/compositions/{ident}/evaluate")
    def evaluate_composition(ident: str, request: EvaluateRequest):
        return engine.evaluate_composition(ident, request)

    @app.get("/api/compositions/{ident}/agent-result")
    def composition_agent_result(ident: str):
        from .agent_loop import best_result
        return JSONResponse(best_result(engine,ident),headers={
            "Content-Disposition":'attachment; filename="unicompose3d-composition-report.json"'})

    @app.get("/api/compositions/{ident}/diagnostics.zip")
    def composition_diagnostics(ident: str):
        current=store.read("compositions",ident)
        records=[current]
        for item in current.get("agent_history",[]):
            records.append(store.read("compositions",item["id"]))
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,"w",zipfile.ZIP_DEFLATED) as archive:
            for slot,asset_id in (("A",current["base_id"]),("B",current["product_id"])):
                archive.write(engine.model_path(asset_id),slot+".glb")
            for record in records:
                prefix="round_"+str(record.get("agent_round",1))
                archive.writestr(prefix+"/report.json",json.dumps(record,ensure_ascii=False,indent=2))
                for render in record.get("renders",[]):
                    path=store.path("compositions",record["id"])/(render["candidate_id"]+".png")
                    if path.is_file():archive.write(path,prefix+"/"+render["candidate_id"]+".png")
        return __import__("fastapi").Response(stream.getvalue(),media_type="application/zip",
            headers={"Content-Disposition":'attachment; filename="unicompose3d-composition-diagnostics.zip"'})

    @app.get("/api/compositions/{ident}/report")
    def composition_report(ident: str):
        result = store.read("compositions", ident)
        return JSONResponse(result, headers={"Content-Disposition": 'attachment; filename="unicompose3d-composition-report.json"'})

    @app.get("/api/compositions/{ident}/renders/{candidate_id}.png")
    def composition_render(ident: str, candidate_id: str):
        value = store.read("compositions", ident)
        if candidate_id not in {r["candidate_id"] for r in value.get("renders", [])}:
            raise HTTPException(404, "No render is available for this candidate.")
        return FileResponse(store.path("compositions", ident) / f"{candidate_id}.png", media_type="image/png")

    @app.post("/api/align")
    def align(request: AnchorRequest):
        return {"transform": geometry.align_anchors(request.base_point, request.product_point, request.base_transform, request.product_transform).model_dump()}

    @app.post("/api/fit-aperture")
    @app.post("/api/fit-ring", include_in_schema=False)
    def fit_ring(request: ApertureFitRequest):
        base = geometry.flat_mesh(engine.model_path(request.base_id))
        product = geometry.flat_mesh(engine.model_path(request.product_id))
        transform, diagnostics = aperture.fit(base, product, request.base_transform, request.base_point, request.axis_end, request.product_transform.rotation)
        return {"transform": transform.model_dump(), "diagnostics": diagnostics, "warning": "Aesthetic fit based on mesh sections; not physical sizing. Rotate to inspect all sides."}

    @app.post("/api/deform")
    def deform(request: DeformRequest):
        path = engine.model_path(request.asset_id)
        ident, directory = store.create("assets", {"name": request.name, "origin": "deformation", "parent_id": request.asset_id, "status": "processing"})
        info = geometry.deform_mesh(path, directory / "model.glb", request.point, request.delta, request.radius)
        return store.update("assets", ident, status="ready", **info)

    @app.post("/api/demo")
    def demo():
        existing = {a.get("demo_slot"): a for a in store.list("assets") if a.get("origin") == "demo" and a.get("status") == "ready"}
        if "A" in existing and "B" in existing:
            return {"A": existing["A"], "B": existing["B"]}
        source_id, directory = store.create("sources", {"kind": "demo"})
        a, b = geometry.create_demo(directory)
        return {"A": engine.asset(a, "Procedural table", origin="demo", demo_slot="A"), "B": engine.asset(b, "Procedural vase", origin="demo", demo_slot="B")}

    @app.post("/api/scenes")
    def save_scene(request: SceneRequest):
        engine.model_path(request.base_id)
        engine.model_path(request.product_id)
        ident, _ = store.create("scenes", request.model_dump())
        return store.read("scenes", ident)

    @app.get("/api/scenes")
    def scenes():
        return store.list("scenes")

    @app.post("/api/export")
    def export(request: SceneRequest):
        payload = geometry.export_scene(engine.model_path(request.base_id), engine.model_path(request.product_id), request.base_transform, request.product_transform)
        return __import__("fastapi").Response(payload, media_type="model/gltf-binary", headers={"Content-Disposition": 'attachment; filename="unicompose3d-scene.glb"'})

    @app.post("/api/project/export")
    def export_project(request: SceneRequest):
        paths = {"A": engine.model_path(request.base_id), "B": engine.model_path(request.product_id)}
        stream = io.BytesIO()
        manifest = {"format": "universal3dtryon", "version": 1, "scene": request.model_dump(), "assets": {slot: store.read("assets", ident) for slot, ident in (("A", request.base_id), ("B", request.product_id))}}
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("project.json", json.dumps(manifest, ensure_ascii=False))
            for slot, path in paths.items():
                archive.write(path, f"{slot}.glb")
        return __import__("fastapi").Response(stream.getvalue(), media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="unicompose3d-project.zip"'})

    @app.post("/api/project/import")
    async def import_project(file: UploadFile = File(...)):
        data = await file.read(150 * 1024 * 1024 + 1)
        await file.close()
        if len(data) > 150 * 1024 * 1024:
            raise ValueError("Project ZIP exceeds 150 MB.")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                required = {"project.json", "A.glb", "B.glb"}
                if set(archive.namelist()) != required or sum(i.file_size for i in archive.infolist()) > 300 * 1024 * 1024:
                    raise ValueError("Invalid or oversized project ZIP.")
                if archive.getinfo("project.json").file_size > 100_000:
                    raise ValueError("Manifest is too large.")
                manifest = json.loads(archive.read("project.json"))
                if manifest.get("format") != "universal3dtryon" or manifest.get("version") != 1:
                    raise ValueError("Unsupported project version.")
                scene = SceneRequest.model_validate(manifest["scene"])
                source_id, directory = store.create("sources", {"kind": "project"})
                ids = {}
                for slot in ("A", "B"):
                    target = directory / f"{slot}.glb"
                    target.write_bytes(archive.read(f"{slot}.glb"))
                    meta = manifest.get("assets", {}).get(slot, {})
                    # Preserve original coordinates, including sculpted meshes.
                    geometry.load_scene(target)
                    ident, dest = store.create("assets", {"name": str(meta.get("name", slot))[:100], "origin": "project", "status": "ready"})
                    (dest / "model.glb").write_bytes(target.read_bytes())
                    store.update("assets", ident, **geometry.describe(geometry.load_scene(target)))
                    ids[slot] = ident
                scene.base_id, scene.product_id = ids["A"], ids["B"]
                ident, _ = store.create("scenes", scene.model_dump())
                return store.read("scenes", ident)
        except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
            raise ValueError("Project ZIP is corrupt or incomplete.") from exc

    app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="studio")
    return app





