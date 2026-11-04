"""Bounded warm Hunyuan cache and measured GPU placement; no credentials."""
from collections import OrderedDict
import gc,time

def free_ram():
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:return 0

def model_bytes(modules):
    seen=set();total=0
    for module in modules:
        if not hasattr(module,"parameters"):continue
        for tensor in list(module.parameters())+list(module.buffers()):
            key=tensor.data_ptr()
            if key not in seen:total+=tensor.numel()*tensor.element_size();seen.add(key)
    return total

class WarmModels:
    def __init__(self,max_bytes=10*1024**3):
        self.items=OrderedDict();self.max_bytes=max_bytes
    def get(self,key,loader,modules):
        reused=key in self.items
        if reused:
            value,parts,size=self.items.pop(key);self.items[key]=(value,parts,size)
            return value,True
        self.prune(reserve=5*1024**3)
        value=loader();parts=modules(value);size=model_bytes(parts)
        self.items[key]=(value,parts,size)
        self.prune(keep=key)
        return value,False
    def prune(self,keep=None,reserve=4*1024**3):
        for key in list(self.items):
            if key==keep:continue
            if sum(x[2] for x in self.items.values())<=self.max_bytes and free_ram()>=reserve:break
            self.items.pop(key)
            gc.collect()
    def park(self):
        import torch
        for value,parts,size in list(self.items.values()):
            for part in parts:
                if hasattr(part,"to"):part.to("cpu")
        self.prune()
        gc.collect();torch.cuda.empty_cache()

def prepare_shape(pipeline,force_offload=False):
    import torch
    parts=[pipeline.conditioner,pipeline.model,pipeline.vae]
    required=model_bytes(parts)+4*1024**3
    available,_=torch.cuda.mem_get_info()
    if not force_offload and available>required:
        try:
            for part in parts:part.to("cuda")
            return [],"gpu-phase-resident"
        except torch.cuda.OutOfMemoryError:
            for part in parts:part.to("cpu")
            gc.collect();torch.cuda.empty_cache()
    from accelerate import cpu_offload_with_hook
    hooks=[];previous=None
    for part in parts:
        _,previous=cpu_offload_with_hook(part,"cuda:0",prev_module_hook=previous)
        hooks.append(previous)
    return hooks,"component-offload"

def finish_shape(pipeline,hooks):
    for hook in hooks:hook.offload();hook.remove()
    for part in (pipeline.conditioner,pipeline.model,pipeline.vae):part.to("cpu")

def paint_parts(paint):
    pipeline=paint.models["multiview_model"].pipeline
    return [value for value in pipeline.components.values() if hasattr(value,"parameters")]

def prepare_paint(paint):
    import torch
    pipeline=paint.models["multiview_model"].pipeline
    try:
        from accelerate.hooks import remove_hook_from_module
        for part in paint_parts(paint):remove_hook_from_module(part,recurse=True)
    except ImportError:pass
    if hasattr(pipeline,"_all_hooks"):pipeline._all_hooks=[]
    available,_=torch.cuda.mem_get_info()
    if available>model_bytes(paint_parts(paint))+5*1024**3:
        try:
            pipeline.to("cuda");return "gpu-phase-resident"
        except torch.cuda.OutOfMemoryError:
            for part in paint_parts(paint):part.to("cpu")
            gc.collect();torch.cuda.empty_cache()
    paint.enable_model_cpu_offload();return "component-offload"
