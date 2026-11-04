"""Replay recorded mesh-grounded axes; never invent corrected semantic landmarks."""
import sys,json,argparse,time
from pathlib import Path
import numpy as np,trimesh
parser=argparse.ArgumentParser();parser.add_argument("report",type=Path)
parser.add_argument("--a",type=Path,required=True);parser.add_argument("--b",type=Path,required=True)
parser.add_argument("--source",type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();sys.path.insert(0,str(args.source))
from unicompose3d.schemas import PlacementPlan,Transform
from unicompose3d import geometry
from unicompose3d.constraint_solver import build_candidates
report=json.loads(args.report.read_text(encoding="utf-8"));plan=PlacementPlan.model_validate(report["plan"])
base=Transform.model_validate(report["base_transform"]);inverse=np.linalg.inv(geometry.transform_matrix(base))
meshes={"A":geometry.flat_mesh(args.a),"B":geometry.flat_mesh(args.b)}
for feature in plan.program.features:
 measured=report["diagnostics"]["resolved_features"].get(feature.id)
 if not measured:continue
 # The exact recorded UV raycast outputs are the replay input. No hand-picked repairs.
 if feature.kind in {"axis","surface","point"}:
  start=np.asarray(measured["center"]);end=start+np.asarray(measured["axis"])*measured["extent"] if measured.get("axis") is not None else None
  if feature.slot=="A":
   start=trimesh.transform_points([start],inverse)[0]
   if end is not None:end=trimesh.transform_points([end],inverse)[0]
  mesh=meshes[feature.slot];fraction=(start-mesh.bounds[0])/mesh.extents
  feature.point=np.clip(fraction,0,1).tolist()
  if end is not None:feature.end=np.clip((end-mesh.bounds[0])/mesh.extents,0,1).tolist()
 feature.uv=feature.end_uv=None;feature.view=feature.end_view=-1
plan.base_uv=plan.product_uv=None;plan.base_view=plan.product_view=-1
began=time.monotonic()
candidates,diagnostics=build_candidates(args.a,args.b,plan,[],base,4)
args.output.mkdir(parents=True,exist_ok=True)
summary={"mode":"recorded geometric grounding replay; no new LLM call","source_report":report["id"],"seconds":time.monotonic()-began,
 "plan":plan.model_dump(),"base_transform":base.model_dump(),"candidates":candidates,"diagnostics":diagnostics}
(args.output/"replay.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
for c in candidates:
 (args.output/(c["id"]+".glb")).write_bytes(geometry.export_scene(args.a,args.b,base,Transform.model_validate(c["transform"])))
print("Replay:",len(candidates),"candidates;",sum(c["eligible"] for c in candidates),"eligible;",len(diagnostics.get("orientation_issues",[])),"axis contradictions",flush=True)
