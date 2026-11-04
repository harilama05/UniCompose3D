"""Bounded model cache; never touches photos, exported meshes or projects."""
from __future__ import annotations

import shutil
from pathlib import Path

GIB = 1024 ** 3
RESERVE = GIB
REDUNDANT = "single/hunyuan3d-paint-v2-0-turbo/unet/diffusion_pytorch_model.bin"
REDUNDANT_SHA = "24e7f1aea8a7c94cee627eb06f5265f19eeff4e19568636c5eaef050cc19ba3d"
IGNORE_PATTERNS = ["hunyuan3d-paint-v2-0-turbo/unet/diffusion_pytorch_model.bin"]
# Sizes and LFS etags verified from the official pinned Hugging Face revisions.
HEAVY_FILES = {
    "sam/model.safetensors": (374979480, "892c410e496344e527255ccdcb2cb7244a609acb5389c7c4fdba1288f861c579"),
    "single/hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors": (4930777530, "5ee5a81e4df08a1c65b79910bf5b145a90376e526794f4607a4d5d068d62f269"),
    "multiview/hunyuan3d-dit-v2-mv-turbo/model.fp16.safetensors": (4930777530, "172d7a989b99f66af760da0080d8c6f6350c72609536cf0c55b87a4afa574e45"),
    "single/hunyuan3d-delight-v2-0/text_encoder/model.safetensors": (680820392, "bc1827c465450322616f06dea41596eac7d493f4e95904dcb51f0fc745c4e13f"),
    "single/hunyuan3d-delight-v2-0/unet/diffusion_pytorch_model.safetensors": (3463772592, "0ce61d15a43d11ba19079ab8f24dfce78b876d3f5291470079ef64b17e08ca58"),
    "single/hunyuan3d-delight-v2-0/vae/diffusion_pytorch_model.safetensors": (167335342, "3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a"),
    "single/hunyuan3d-paint-v2-0-turbo/image_encoder/model.safetensors": (1264217240, "ae616c24393dd1854372b0639e5541666f7521cbe219669255e865cb7f89466a"),
    "single/hunyuan3d-paint-v2-0-turbo/text_encoder/pytorch_model.bin": (1361671895, "c3e254d7b61353497ea0be2c4013df4ea8f739ee88cffa0ba58cd085459ed565"),
    "single/hunyuan3d-paint-v2-0-turbo/unet/diffusion_pytorch_model.safetensors": (3722161032, "d6acffa4a22f4da61d87f446bfa83e7ac245481c1535fbf25b200fe4462d0b22"),
    "single/hunyuan3d-paint-v2-0-turbo/vae/diffusion_pytorch_model.bin": (334707217, "1b4889b6b1d4ce7ae320a02dedaeff1780ad77d415ea0d744b476155c6377ddc"),
}


def shape_file(multiview):
    return "multiview/hunyuan3d-dit-v2-mv-turbo/model.fp16.safetensors" if multiview else "single/hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors"


def selected_files(multiview=False):
    return [name for name in HEAVY_FILES if not name.startswith("single/hunyuan3d-delight-") and name not in (shape_file(True), shape_file(False))] + [shape_file(multiview)]


def verify_weights(root, multiview=False):
    root = Path(root)
    for relative in selected_files(multiview):
        path = root / relative
        expected = HEAVY_FILES[relative][0]
        if not path.is_file() or path.stat().st_size != expected:
            raise RuntimeError("Missing weights or insufficient disk space: " + relative + ". Run the resume download cell in the notebook.")


def incomplete_path(root, relative, etag):
    from huggingface_hub._local_folder import get_local_download_paths
    slot, filename = relative.split("/", 1)
    return get_local_download_paths(root / slot, filename).incomplete_path(etag)


def needed(root, relative):
    expected, etag = HEAVY_FILES[relative]
    path = root / relative
    if path.is_file() and path.stat().st_size == expected:
        return 0
    # Account for bytes already reserved by this exact revision's partial file.
    partial = incomplete_path(root, relative, etag)
    existing = partial.stat().st_size if partial.is_file() else 0
    return max(0, expected - existing) if existing <= expected else expected


def check_space(root, multiview=False, shape_only=False):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    selected = shape_file(multiview)
    files = [selected] if shape_only else selected_files(multiview)
    remaining = sum(needed(root, name) for name in files)
    free = shutil.disk_usage(root).free
    print(f"Free disk space: {free / GIB:.2f} GiB; remaining weights require {remaining / GIB:.2f} GiB; keep {RESERVE / GIB:.1f} GiB for results.", flush=True)
    if free < remaining + RESERVE:
        raise RuntimeError(f"Insufficient disk space: free another {(remaining + RESERVE - free) / GIB:.2f} GiB. Images, GLBs, and projects are preserved; remove pip cache and redundant UNet .bin files. Do not download onto a full disk.")


def unlink_cache(root, paths):
    root = Path(root).resolve()
    total = 0
    # Verify EVERY target before deleting any file; reject symlink escapes.
    verified = []
    for path in paths:
        path = Path(path)
        if not path.resolve().is_relative_to(root):
            raise ValueError("Cache path escapes the weights directory; refusing deletion.")
        if path.is_file():
            verified.append(path)
    for path in verified:
        total += path.stat().st_size
        path.unlink()
    return total


def remove_redundant(root):
    root = Path(root).resolve()
    if not root.exists():
        return 0
    paths = [root / REDUNDANT, incomplete_path(root, REDUNDANT, REDUNDANT_SHA)]
    freed = unlink_cache(root, paths)
    if freed:
        print(f"Removed redundant UNet .bin: {freed / GIB:.2f} GiB. Safetensors, images, and GLBs are preserved.", flush=True)
    return freed


def evict_other_shape(root, multiview):
    # Called only before loading a model, inside the shared GPU lock. These are
    # reproducible pretrained caches; never remove the 'single' directory that
    # also holds Paint/Delight, or any user reconstruction.
    root = Path(root).resolve()
    other = shape_file(not multiview)
    paths = [root / other, incomplete_path(root, other, HEAVY_FILES[other][1])]
    freed = unlink_cache(root, paths)
    if freed:
        print(f"Switching mode: remove the unused shape checkpoint cache ({freed / GIB:.2f} GiB); GLBs and projects are preserved.", flush=True)
    return freed
