"""Real CPU parameter sizes, fake GPU transfers: no inference/VRAM claims."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from quality_runtime import WarmModels,prepare_shape,finish_shape

class RuntimeTests(unittest.TestCase):
    def test_cache_reuses_weights_and_evicts_to_fit_its_budget(self):
        import torch
        cache=WarmModels(max_bytes=40);loads=[]
        def load():
            loads.append(1);return torch.nn.Linear(2,2,bias=False)
        with patch("quality_runtime.free_ram",return_value=20*1024**3):
            one,reused=cache.get("one",load,lambda m:[m]);self.assertFalse(reused)
            self.assertIs(cache.get("one",load,lambda m:[m])[0],one)
            self.assertEqual(len(loads),1)
            cache.get("two",load,lambda m:[m]);cache.get("three",load,lambda m:[m])
            self.assertNotIn("one",cache.items)
    def test_low_host_memory_evicts_inactive_cached_parameters(self):
        import torch
        cache=WarmModels()
        with patch("quality_runtime.free_ram",return_value=20*1024**3):
            cache.get("old",lambda:torch.nn.Linear(2,2),lambda m:[m])
        with patch("quality_runtime.free_ram",return_value=1024):
            cache.prune()
        self.assertEqual(len(cache.items),0)
    def parts(self,oom=False):
        import torch
        class Part:
            def __init__(self):self.moves=[];self.fail=oom
            def parameters(self):return [torch.zeros(1)]
            def buffers(self):return []
            def to(self,where):
                self.moves.append(where)
                if where=="cuda" and self.fail:
                    self.fail=False;raise torch.cuda.OutOfMemoryError("fixture")
                return self
        return [Part(),Part(),Part()]
    def test_gpu_residency_is_selected_from_free_memory_not_a_fixed_three_gib_cap(self):
        import torch
        parts=self.parts();pipeline=SimpleNamespace(conditioner=parts[0],model=parts[1],vae=parts[2])
        with patch("torch.cuda.mem_get_info",return_value=(12*1024**3,15*1024**3)):
            hooks,policy=prepare_shape(pipeline)
        self.assertEqual(policy,"gpu-phase-resident");self.assertEqual(hooks,[])
        self.assertTrue(all(p.moves==["cuda"] for p in parts))
        finish_shape(pipeline,hooks)
        self.assertTrue(all(p.moves[-1]=="cpu" for p in parts))
    def test_oom_during_placement_automatically_uses_component_offload(self):
        parts=self.parts(oom=True);pipeline=SimpleNamespace(conditioner=parts[0],model=parts[1],vae=parts[2])
        hook=Mock()
        fake=SimpleNamespace(cpu_offload_with_hook=Mock(side_effect=lambda part,*a,**k:(part,hook)))
        with patch.dict(sys.modules,{"accelerate":fake}),patch("torch.cuda.mem_get_info",return_value=(12*1024**3,15*1024**3)),patch("torch.cuda.empty_cache"):
            hooks,policy=prepare_shape(pipeline)
        self.assertEqual(policy,"component-offload")
        self.assertEqual(fake.cpu_offload_with_hook.call_count,3)
        self.assertEqual(len(hooks),3)
if __name__=="__main__":unittest.main()
