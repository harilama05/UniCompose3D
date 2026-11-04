"""Compare a frozen source with current geometry on the SAME case definitions."""
import sys,os,json,time,tempfile,argparse
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument("--source",type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args();sys.path.insert(0,str(args.source))
from composition_cases import cases
from unicompose3d.constraint_solver import build_candidates
results=[]
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)
 for name,a,b,p,base,expected in cases():
  pa,pb=root/"a.glb",root/"b.glb";a.export(pa);b.export(pb)
  began=time.monotonic()
  candidates,diagnostics=build_candidates(pa,pb,p,[],base,4)
  actual=any(c["eligible"] for c in candidates)
  region=getattr(p,"_benchmark_region_check",None);region_correct=True
  if region:
   feature=diagnostics.get("resolved_features",{}).get(region["feature"],{})
   region_correct=bool(feature and feature["center"][region["axis"]]*region["sign"]>=region["minimum"])
  results.append(dict(case=name,expected_eligible=expected,actual_eligible=actual,passed=actual==expected and region_correct,target_region_correct=region_correct,
   candidates=len(candidates),eligible=sum(c["eligible"] for c in candidates),seconds=round(time.monotonic()-began,3),
   feature_error=diagnostics.get("feature_error"),best_score=max(c["geometry_score"] for c in candidates)))
  print(name,actual==expected,flush=True)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(dict(mode="deterministic geometry; not a live LLM benchmark",source=str(args.source),
 passed=sum(r["passed"] for r in results),total=len(results),results=results),indent=2),encoding="utf-8")
