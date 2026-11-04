"""Resume pinned public weights after a successful CUDA build; GET retries only."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from unicompose3d.quality import QUALITY_SETTINGS, SEGMENT_SETTINGS
from unicompose3d.weight_storage import IGNORE_PATTERNS, check_space, remove_redundant, evict_other_shape, shape_file, needed, verify_weights


def transient(error):
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if status is not None:
            return status in {408, 429, 500, 502, 503, 504}
        if isinstance(current, (requests.Timeout, requests.ConnectionError)):
            return True
        current = current.__cause__ or current.__context__
    return False


def download_snapshots(specs, downloader=None, sleep=time.sleep, attempts=5):
    if downloader is None:
        from huggingface_hub import snapshot_download
        downloader = snapshot_download
    for repo, revision, destination, patterns in specs:
        print("Checking/resuming download:", repo, flush=True)
        for attempt in range(attempts):
            try:
                downloader(repo, revision=revision, local_dir=destination,
                           allow_patterns=patterns, ignore_patterns=IGNORE_PATTERNS, max_workers=1, force_download=False)
                break
            except requests.RequestException as exc:
                if not transient(exc):
                    raise
                if attempt + 1 == attempts:
                    raise RuntimeError("Hugging Face still failed after retries. Downloaded files are retained. Run --resume-downloads later without reinstalling.") from None
                wait = min(30, 2 ** (attempt + 1))
                print(f"Temporary weight-download failure. Retry {attempt + 2}/{attempts} in {wait}s; downloaded files are retained.", flush=True)
                sleep(wait)


def ensure_shape(weights, multiview):
    from setup_quality import download_spec
    weights = Path(weights)
    remove_redundant(weights)
    try:
        check_space(weights, multiview, shape_only=True)
    except RuntimeError:
        evict_other_shape(weights, multiview)
        check_space(weights, multiview, shape_only=True)
    original = download_spec(weights, multiview)[-1 if multiview else 1]
    folder = "hunyuan3d-dit-v2-mv-turbo" if multiview else "hunyuan3d-dit-v2-0-turbo"
    if needed(weights, shape_file(multiview)) == 0 and (Path(original[2]) / folder / "config.yaml").is_file():
        return
    spec = [*original[:3], [folder + "/config.yaml", folder + "/model.fp16.safetensors"]]
    download_snapshots([spec])


def finish(repository, weights):
    from setup_quality import download_spec
    # Validate compiled runtime BEFORE downloading or creating readiness markers.
    import torch, torchvision, custom_rasterizer
    from transformers import SamModel, SamProcessor, CLIPVisionModel
    sys.path.insert(0, str(repository))
    from hunyuan_compat import patch_repository
    patch_repository(repository)
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; enable a GPU before resuming.")
    torch.zeros(1, device="cuda").add_(1)
    from hunyuan_compat import probe_decoder
    probe_decoder()
    print("SAM + Hunyuan shape/paint imports + CUDA OK", flush=True)
    weights.mkdir(parents=True, exist_ok=True)
    for marker in (".segment-ready", ".quality-ready"):
        (weights / marker).unlink(missing_ok=True)
    remove_redundant(weights)
    try:
        check_space(weights)
    except RuntimeError:
        evict_other_shape(weights, False)
        check_space(weights)
    download_snapshots(download_spec(weights))
    verify_weights(weights)
    # The checkpoint supplies a CUSTOM UNet loader that ignores Diffusers'
    # automatic safetensors preference. Adapt it before declaring Paint ready.
    from paint_compat import patch_paint_weights
    patch_paint_weights(weights)
    SamProcessor.from_pretrained(str(weights / "sam"), local_files_only=True)
    SamModel.from_pretrained(str(weights / "sam"), local_files_only=True)
    (weights / ".segment-ready").write_text(SEGMENT_SETTINGS, encoding="utf-8")
    (weights / ".quality-ready").write_text(QUALITY_SETTINGS, encoding="utf-8")
    print("Weights downloaded and verified. Continue with steps 3 and 4.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    finish(args.repository, args.weights)


if __name__ == "__main__":
    main()
