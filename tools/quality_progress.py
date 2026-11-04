"""Atomic progress and timings readable while the GPU subprocess is running."""
import json
import os
import time
from pathlib import Path

class Progress:
    def __init__(self, directory,notify=None):
        self.notify=notify
        self.path = Path(directory) / "hunyuan-progress.json"
        self.timings = {}
        self.started = time.monotonic()
        self.serial = 0

    def emit(self, stage):
        self.serial += 1
        payload = {"serial":self.serial, "stage":stage, "timings":dict(self.timings),
                   "elapsed_s":round(time.monotonic()-self.started,3)}
        try:
            import torch,psutil
            free,total=torch.cuda.mem_get_info()
            payload["resources"]={"gpu_free_gib":round(free/1024**3,2),
                "gpu_total_gib":round(total/1024**3,2),
                "gpu_allocated_gib":round(torch.cuda.memory_allocated()/1024**3,2),
                "process_ram_gib":round(psutil.Process().memory_info().rss/1024**3,2)}
        except (ImportError,RuntimeError,AssertionError):pass
        if "resources" in payload:
            self.timings["resources"]=payload["resources"]
            payload["timings"]["resources"]=payload["resources"]
        if self.notify:self.notify(payload)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload),encoding="utf-8")
        os.replace(temporary,self.path)
        print("HUNYUAN_STAGE " + json.dumps(payload),flush=True)

    def call(self, stage, function, *args, **kwargs):
        self.emit(stage)
        began = time.monotonic()
        try:
            return function(*args, **kwargs)
        finally:
            self.timings[stage+"_s"] = round(time.monotonic()-began,3)

    def wrap(self, stage, function):
        def run(*args,**kwargs):
            return self.call(stage,function,*args,**kwargs)
        return run

def gpu_vertex_normals(positions, indices, face_normals):
    """Equivalent unweighted mean normals; avoid a full GPU->CPU->GPU trip."""
    import torch
    result = torch.zeros_like(positions[:, :3])
    result.index_add_(0, indices[:,:3].reshape(-1).long(),
                      face_normals.repeat_interleave(3,dim=0).to(result.dtype))
    length = result.norm(dim=1,keepdim=True)
    return torch.where(length > 1e-13, result / length.clamp_min(1e-13), torch.zeros_like(result))
