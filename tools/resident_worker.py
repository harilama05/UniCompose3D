"""JSON-line warm SAM worker. No LLM/API credentials are passed here."""
from __future__ import annotations
import argparse
from collections import OrderedDict
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Use stdout only for the protocol, even while importing noisy third-party code.
CHANNEL = sys.stdout
def emit(event, **values):
    CHANNEL.write(json.dumps({"event": event, **values}) + "\n"); CHANNEL.flush()


class SamSession:
    def __init__(self, weights, device="cuda"):
        self.device = device
        self.weights = weights; self.model = None; self.processor = None; self.cache = OrderedDict()

    def run(self, request):
        import torch
        from PIL import Image
        from transformers import SamModel, SamProcessor
        from quality_worker import sam_prompts
        if self.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("SAM requires a CUDA GPU.")
        timings = {}; begin = time.monotonic()
        emit("progress", stage="sam_load", timings=timings)
        if self.model is None:
            self.model = SamModel.from_pretrained(str(self.weights / "sam"), local_files_only=True).eval().to(self.device)
            self.processor = SamProcessor.from_pretrained(str(self.weights / "sam"), local_files_only=True)
        self.model.to(self.device)
        timings["sam_load_s"] = round(time.monotonic() - begin, 3)
        image_path = Path(request["image"]); digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt", **sam_prompts(image.size, request)).to(self.device)
        emit("progress", stage="image_encode", timings=timings)
        begin = time.monotonic(); cached = digest in self.cache
        with torch.inference_mode():
            if cached:
                features = self.cache.pop(digest).to(self.device); self.cache[digest] = features
            else:
                features = self.model.get_image_embeddings(inputs["pixel_values"])
                self.cache[digest] = features
                while len(self.cache) > 2: self.cache.popitem(last=False)
            if self.device == "cuda": torch.cuda.synchronize()
            timings["image_encode_s"] = round(time.monotonic() - begin, 3)
            timings["embedding_cached"] = cached
            emit("progress", stage="sam_mask", timings=timings)
            begin = time.monotonic()
            prompts = {k: inputs[k] for k in ("input_boxes", "input_points", "input_labels") if k in inputs}
            outputs = self.model(image_embeddings=features, **prompts)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0][0]
        from sam_mask import select_mask
        mask = select_mask(masks.numpy(), outputs.iou_scores[0, 0].cpu().numpy(),
                           request["points"], request["labels"])
        Image.fromarray(mask).save(request["output"])
        timings["sam_mask_s"] = round(time.monotonic() - begin, 3)
        return {"output": request["output"], "timings": timings}




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["sam"])
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    session = SamSession(args.weights)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request.get("_control") == "park":
                    import torch
                    if session.model is not None: session.model.to("cpu")
                    if isinstance(session, SamSession):
                        session.cache = OrderedDict((key, value.cpu()) for key, value in session.cache.items())
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    result = {"parked": True}
                else:
                    result = session.run(request)
            emit("result", result=result)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit("error", error=type(error).__name__ + ": " + str(error)[-1200:])


if __name__ == "__main__": main()


