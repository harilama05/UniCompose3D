"""Resident GPU worker with bounded warm CPU caches and phase placement."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")


def sam_prompts(size, request):
    width, height = size
    box = [v * (width if i % 2 == 0 else height) for i, v in enumerate(request["box"])]
    prompts = {"input_boxes": [[box]]}
    if request["points"]:
        prompts["input_points"] = [[[[x * width, y * height] for x, y in request["points"]]]]
        prompts["input_labels"] = [[request["labels"]]]
    return prompts


def segment(request, weights):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import SamModel, SamProcessor

    assert torch.cuda.is_available(), "CUDA GPU unavailable"
    image = Image.open(request["image"]).convert("RGB")
    model = SamModel.from_pretrained(str(weights / "sam"), local_files_only=True).to("cuda")
    processor = SamProcessor.from_pretrained(str(weights / "sam"), local_files_only=True)
    inputs = processor(image, return_tensors="pt", **sam_prompts(image.size, request)).to("cuda")
    with torch.inference_mode():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0][0]
    from sam_mask import select_mask
    mask = select_mask(masks.numpy(), outputs.iou_scores[0, 0].cpu().numpy(),
                       request["points"], request["labels"])
    Image.fromarray(mask, "L").save(request["output"])


def reconstruct(request, repository, weights, runtime=None, notify=None):
    import torch
    from PIL import Image

    assert torch.cuda.is_available(), "CUDA GPU unavailable"
    sys.path.insert(0, str(repository))
    from hunyuan_compat import patch_repository
    patch_repository(repository)
    from hunyuan_compat import patch_renderer
    patch_renderer(repository)
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    from quality_progress import Progress
    progress = Progress(Path(request["output"]).parent,notify=notify)
    views = {view: Image.open(path).convert("RGBA") for view, path in request["views"].items()}
    multiview = len(views) > 1
    from download_quality_weights import ensure_shape
    progress.call("weights", ensure_shape, weights, multiview)
    from quality_runtime import WarmModels,prepare_shape,finish_shape
    runtime=runtime or WarmModels()
    def load_shape():
        return progress.call("shape_load",Hunyuan3DDiTFlowMatchingPipeline.from_pretrained,
            str(weights / ("multiview" if multiview else "single")),
            subfolder="hunyuan3d-dit-v2-mv-turbo" if multiview else "hunyuan3d-dit-v2-0-turbo",
            use_safetensors=True,device="cpu",dtype=torch.float16)
    pipeline,warm=runtime.get(("shape",bool(multiview)),load_shape,
        lambda value:[value.conditioner,value.model,value.vae])
    if warm:progress.timings["shape_load_s"]=0.
    # Hierarchical decoding uses the checkpoint's own VAE; no hidden downloads.
    # Keep adaptive KV selection off: it requires a different distilled VAE.
    pipeline.enable_flashvdm(adaptive_kv_selection=False, replace_vae=False, mc_algo="mc")
    from hunyuan_compat import install_surface_extractor
    install_surface_extractor(pipeline.vae)
    # This upstream revision exposes an incomplete offload convenience method
    # (expects DiffusionPipeline attributes it does not initialize). Use the
    # public accelerate API with explicit components and execution order.
    hooks,policy=prepare_shape(pipeline)
    pipeline.device=torch.device("cuda")
    original_decode=pipeline.vae.latents2mesh
    pipeline.vae.latents2mesh=progress.wrap("decode",original_decode)
    try:
        retry=False
        with torch.inference_mode():
            try:
                mesh=progress.call("shape",pipeline,image=views if multiview else views["front"],
                    num_inference_steps=5,octree_resolution=384,num_chunks=32768 if not hooks else 8000,
                    generator=torch.Generator(device="cuda").manual_seed(42),output_type="trimesh")[0]
            except torch.cuda.OutOfMemoryError:retry=True
            if retry:
                # Leave the exception scope before retrying, releasing tensors
                # retained by the failed forward's traceback.
                finish_shape(pipeline,hooks);hooks=[]
                gc.collect();torch.cuda.empty_cache();progress.emit("memory_retry")
                hooks,policy=prepare_shape(pipeline,force_offload=True)
                mesh=progress.call("shape",pipeline,image=views if multiview else views["front"],
                    num_inference_steps=5,octree_resolution=384,num_chunks=8000,
                    generator=torch.Generator(device="cuda").manual_seed(42),output_type="trimesh")[0]
    finally:
        pipeline.vae.latents2mesh=original_decode
        finish_shape(pipeline,hooks)
    if mesh is None or len(mesh.faces) == 0:
        raise RuntimeError("No surface was extracted; try a clearer image or mask.")
    import numpy as np
    if not np.isfinite(mesh.vertices).all():
        raise RuntimeError("Decoded mesh contains non-finite vertices; refusing to export.")
    # Preserve topology; avoid largest-component deletion and hole filling.
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    output = Path(request["output"])
    diagnostics = {"model": "Hunyuan3D-2mv-Turbo" if multiview else "Hunyuan3D-2-Turbo",
        "steps": 5, "seed": 42, "used_images": len(views), "views": list(views),
        "shape_s": round(time.monotonic() - began, 3), "texture": bool(request["texture"]),
        "quality_verified": False, "decoder": "hierarchical float coordinates, native VAE, finite MC384",
        "paint": "Turbo, no delight; original lighting may remain"}
    (output.parent / "hunyuan-diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    mesh.export(output.with_name("shape.glb"))
    diagnostics.update(shape_model_reused=warm,shape_gpu_policy=policy)
    del pipeline,original_decode,hooks
    runtime.prune();gc.collect();torch.cuda.empty_cache()
    if request["texture"]:
        try:
            mesh = paint_mesh(mesh, views["front"], repository, weights, progress=progress,runtime=runtime)
        finally:
            diagnostics["timings"] = dict(progress.timings)
            (output.parent / "hunyuan-diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    progress.call("export", mesh.export, output)
    progress.emit("done")
    diagnostics["timings"] = dict(progress.timings)
    diagnostics["total_s"] = round(time.monotonic() - began, 3)
    diagnostics["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    (output.parent / "hunyuan-diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")


def paint_input(value):
    """Respect alpha, including holes, without loading another neural network."""
    from PIL import Image
    return Image.alpha_composite(Image.new("RGBA", value.size, (255, 255, 255, 255)),
                                 value.convert("RGBA")).convert("RGB")


def paint_mesh(mesh, image, repository, weights, progress=None,runtime=None):
    from hunyuan_compat import patch_renderer
    patch_renderer(repository)
    from paint_compat import patch_paint_weights
    patch_paint_weights(weights)
    sys.path.insert(0, str(repository))
    from hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline, Hunyuan3DTexGenConfig
    config = Hunyuan3DTexGenConfig(str(weights / "single/hunyuan3d-delight-v2-0"),
                                str(weights / "single/hunyuan3d-paint-v2-0-turbo"), "hunyuan3d-paint-v2-0-turbo")
    config.device = "cpu"
    config.render_size = 1024
    config.texture_size = 1024
    # Compact paint omits the separate 4 GiB de-light model. White-composite
    # alpha explicitly so RGB stored under transparent pixels never leaks in.
    from hy3dgen.texgen.utils.multiview_utils import Multiview_Diffusion_Net
    class CompactPaint(Hunyuan3DPaintPipeline):
        def load_models(self):
            self.models["delight_model"] = paint_input
            self.models["multiview_model"] = Multiview_Diffusion_Net(self.config)

        def enable_model_cpu_offload(self):
            self.models["multiview_model"].pipeline.enable_model_cpu_offload()
    from quality_progress import Progress
    progress = progress or Progress(Path(repository).parent)
    from quality_runtime import WarmModels,paint_parts,prepare_paint
    runtime=runtime or WarmModels()
    paint,warm=runtime.get(("paint",),lambda:progress.call("paint_load",CompactPaint,config),paint_parts)
    if warm:progress.timings["paint_load_s"]=0.
    policy=prepare_paint(paint)
    progress.timings["paint_model_reused"]=warm
    progress.timings["paint_gpu_policy"]=policy
    # Instrument the real operations, keeping resolution, steps and geometry.
    import hy3dgen.texgen.pipelines as paint_module
    original_uv = paint_module.mesh_uv_wrap
    def fast_uv(current):
        if len(current.faces)<=100000:return original_uv(current)
        from uv_atlas import unwrap
        def update(done,total):
            progress.timings.update(uv_patches_done=done,uv_patches_total=total)
            progress.emit("uv_unwrap")
        result,report=unwrap(current,progress.path.parent,resolution=config.texture_size,progress=update)
        progress.timings["uv_report"]=report
        return result
    paint_module.mesh_uv_wrap = progress.wrap("uv_unwrap",fast_uv)
    originals=(paint.render_normal_multiview,paint.render_position_multiview,
        paint.models["multiview_model"],paint.bake_from_multiview,paint.texture_inpaint)
    paint_parts_holder=paint_parts(paint)
    paint.render_normal_multiview = progress.wrap("paint_normals", paint.render_normal_multiview)
    paint.render_position_multiview = progress.wrap("paint_positions", paint.render_position_multiview)
    paint.models["multiview_model"] = progress.wrap("paint_diffusion", paint.models["multiview_model"])
    paint.bake_from_multiview = progress.wrap("paint_bake", paint.bake_from_multiview)
    paint.texture_inpaint = progress.wrap("paint_inpaint", paint.texture_inpaint)
    try:
        with __import__("torch").inference_mode():
            try:return progress.call("paint_total",paint,mesh,image=image)
            except __import__("torch").cuda.OutOfMemoryError:pass
            for part in paint_parts_holder:part.to("cpu")
            gc.collect();__import__("torch").cuda.empty_cache()
            originals[2].pipeline.enable_model_cpu_offload()
            progress.emit("memory_retry")
            progress.timings["paint_gpu_policy"]="component-offload-after-oom"
            return progress.call("paint_total",paint,mesh,image=image)
    finally:
        paint_module.mesh_uv_wrap=original_uv
        paint.render_normal_multiview,paint.render_position_multiview,paint.models["multiview_model"],paint.bake_from_multiview,paint.texture_inpaint=originals
        for part in paint_parts(paint):part.to("cpu")
        runtime.prune()



def texture(request, repository, weights,runtime=None):
    """Retry only Paint using the exact saved geometry and immutable front image."""
    import torch
    import trimesh
    from PIL import Image
    assert torch.cuda.is_available(), "CUDA GPU unavailable"
    mesh = trimesh.load(request["shape"], force="mesh", process=False)
    if len(mesh.faces) == 0:
        raise RuntimeError("The saved shape mesh has no surface.")
    image = Image.open(request["views"]["front"]).convert("RGBA")
    result = paint_mesh(mesh,image,repository,weights,runtime=runtime)
    result.export(request["output"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task",choices=["segment","reconstruct","texture","hunyuan"])
    parser.add_argument("request",type=Path,nargs="?")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    if args.task=="hunyuan":
        import contextlib,sys,traceback
        from quality_runtime import WarmModels
        runtime=WarmModels()
        def emit(event,**payload):
            sys.__stdout__.write(json.dumps({"event":event,**payload})+"\n");sys.__stdout__.flush()
        for line in sys.stdin:
            try:
                request=json.loads(line)
                with contextlib.redirect_stdout(sys.stderr):
                    if request.get("_control")=="park":
                        runtime.park();result={"parked":True}
                    else:
                        if request.get("_task")=="texture":
                            texture(request,args.repository,args.weights,runtime=runtime)
                        else:
                            reconstruct(request,args.repository,args.weights,runtime,
                                notify=lambda value:emit("progress",**value))
                        runtime.park();result={"output":request["output"]}
                emit("result",result=result)
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                emit("error",error=type(error).__name__+": "+str(error)[-1200:])
        return
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if args.task == "segment":
        segment(request, args.weights)
    elif args.task == "texture":
        texture(request, args.repository, args.weights)
    else:
        reconstruct(request, args.repository, args.weights)


if __name__ == "__main__":
    main()
