"""Resource-bounded multi-hypothesis search, independent of object categories."""
from dataclasses import dataclass,field
import time
import numpy as np
from . import geometry
class SearchDeadline(ValueError):pass

@dataclass
class SearchContext:
 max_starts:int=16
 max_seconds:float=60.
 max_programs:int=4
 began:float=field(default_factory=time.monotonic)
 starts:int=0
 geometry_cache:dict=field(default_factory=dict)
 measurements:dict=field(default_factory=dict)
 started_programs:int=0
 quota:int=16
 used_in_program:int=0
 exhausted:bool=False
 def check(self):
  if time.monotonic()-self.began>=self.max_seconds:
   self.exhausted=True;raise SearchDeadline("Spatial search time budget reached; retain measured partial results.")
 def start(self):
  self.check()
  if self.starts>=self.max_starts or self.used_in_program>=self.quota:return False
  self.starts+=1;self.used_in_program+=1;return True
 def program(self,remaining):
  self.check();self.started_programs+=1;self.used_in_program=0
  self.quota=max(1,(self.max_starts-self.starts)//max(1,remaining))
 def report(self):
  return {"max_starts":self.max_starts,"starts_used":self.starts,"max_seconds":self.max_seconds,
   "elapsed_s":round(time.monotonic()-self.began,3),"time_budget_exhausted":self.exhausted,
   "programs":self.started_programs,"start_budget_exhausted":self.starts>=self.max_starts,"program_budget_exhausted":self.started_programs>=self.max_programs,"scope":"cooperative deadline between geometric operations; an atomic mesh operation can overrun"}

def local_programs(plan,base_path,product_path,resolved=None,base_transform=None):
 """Explore nearby semantic-compatible anchors, and measured frame alternatives."""
 output=[]
 # Different interpretations of facing vs tangent must be evaluated by the VLM.
 if resolved:
  for feature in plan.program.features:
   measured=resolved.get(feature.id,{})
   if feature.source=="user" or feature.kind!="axis" or not measured.get("alternative_surface_frame"):continue
   from .orientation import conflicts
   if not conflicts(resolved,plan.program.constraints)[0]:continue
   probe={k:dict(v) for k,v in resolved.items()}
   probe[feature.id]={**probe[feature.id],"axis":np.asarray(measured["alternative_surface_frame"]["normal"]),"kind":"surface"}
   if len(conflicts(probe,plan.program.constraints)[0])>=len(conflicts(resolved,plan.program.constraints)[0]):continue
   proposal=plan.model_copy(deep=True);f=next(x for x in proposal.program.features if x.id==feature.id)
   f.kind="surface";f.end=f.end_ref=f.end_uv=None;f.end_view=-1
   primary=next((c for c in proposal.program.constraints if c.kind=="clearance"),None)
   if primary:
    for c in proposal.program.constraints:
     if c.kind=="align" and feature.id in (c.a,c.b) and (c.a,c.b)!=(primary.a,primary.b):
      if any(k.source=="user" and k.id in (c.a,c.b) for k in proposal.program.features):continue
      c.frame_a,c.frame_b=primary.a,primary.b
   output.append(("measured-frame-"+feature.id,"measured surface normal and planar heading around the interface as an alternative interpretation; original intent requires visual verification",proposal))
 # Nearby seating planes remain within the original selected support region.
 candidates=[f for f in plan.program.features if f.source!="user" and f.kind in {"section","surface"}]
 for feature in candidates[:1]:
  if (feature.uv is not None and feature.point_ref is None) or (feature.end_uv is not None and feature.end_ref is None):continue
  mesh=geometry.flat_mesh(base_path if feature.slot=="A" else product_path)
  center=geometry.anchor_point(mesh,feature.point)
  if feature.point_ref:
   from .spatial_context import reference
   center=reference(mesh,feature.point_ref)
  if feature.kind=="section" and (feature.end is not None or feature.end_ref is not None):
   if feature.end_ref:
    from .spatial_context import reference
    endpoint=reference(mesh,feature.end_ref)
   else:endpoint=geometry.anchor_point(mesh,feature.end)
   axis=endpoint-center
   if resolved and resolved.get(feature.id,{}).get("axis") is not None:
    axis=np.asarray(resolved[feature.id]["axis"])
    if feature.slot=="A" and base_transform is not None:axis=np.linalg.solve(geometry.transform_matrix(base_transform)[:3,:3],axis)
  else:
   _,axis=geometry.surface_frame(mesh,center)
   axis=np.cross(axis,[0,1,0])
   if np.linalg.norm(axis)<.1:axis=np.cross(geometry.surface_frame(mesh,center)[1],[1,0,0])
  if np.linalg.norm(axis)<1e-8:continue
  axis=axis/np.linalg.norm(axis)
  # Local budget, not a category distance. User-picked anchors are never moved.
  span=min(float(max(mesh.extents*np.asarray(feature.extent)))*.25,float(max(mesh.extents))*.13)
  for sign in (-1,1):
   proposal=plan.model_copy(deep=True);f=next(x for x in proposal.program.features if x.id==feature.id)
   delta=axis*span*sign;point=(center+delta-mesh.bounds[0])/mesh.extents
   if np.any(point<0) or np.any(point>1):continue
   if f.end is not None or f.end_ref is not None:
    shifted_end=(endpoint+delta-mesh.bounds[0])/mesh.extents
    if np.any(shifted_end<0) or np.any(shifted_end>1):continue
    f.end=shifted_end.tolist()
   f.point=point.tolist();f.point_ref=f.uv=f.end_uv=f.end_ref=None;f.view=f.end_view=-1
   output.append(("local-anchor-"+feature.id+"-"+str(sign),"nearby anchor/section plane within the proposed support region; semantic verification required",proposal))
 return output

def configured_budget():
 import os
 from .schemas import SearchBudget
 return SearchBudget(max_starts=int(os.getenv("TRYON_SEARCH_MAX_STARTS","16")),
  max_seconds=float(os.getenv("TRYON_SEARCH_MAX_SECONDS","60")),
  max_programs=int(os.getenv("TRYON_SEARCH_MAX_PROGRAMS","4"))).model_dump()

def within_original_support(original,proposal,base_path,product_path):
 if not original.program:return False
 initial={f.id:f for f in original.program.features}
 for moved in proposal.program.features:
  start=initial.get(moved.id)
  if start is None:continue
  if moved.point==start.point and moved.point_ref==start.point_ref:continue
  if start.source=="user":return False
  mesh=geometry.flat_mesh(base_path if start.slot=="A" else product_path)
  def fraction(feature):
   if feature.point_ref:
    from .spatial_context import reference
    return (reference(mesh,feature.point_ref)-mesh.bounds[0])/mesh.extents
   return np.asarray(feature.point)
  if np.any(np.abs(fraction(moved)-fraction(start))>np.asarray(start.extent)*.30+1e-8):return False
 return True
