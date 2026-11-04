"""Pose-independent tests of directional constraint compatibility."""
import itertools
import numpy as np
def angle(a,b):
 return float(np.degrees(np.arccos(np.clip(np.dot(a,b),-1,1))))
def conflicts(features,constraints):
 directional=[c for c in constraints if c.kind=="align" and c.frame_a is None]
 primary={(c.a,c.b) for c in constraints if c.kind=="clearance"}
 issues=[];suspended=set()
 for first,second in itertools.combinations(directional,2):
  aa,ab=features[first.a],features[second.a]
  ba,bb=features[first.b],features[second.b]
  if any(v["axis"] is None for v in (aa,ab,ba,bb)):continue
  a_angle=angle(aa["axis"],ab["axis"])
  b_angle=angle(ba["axis"],bb["axis"])
  signs=lambda c:[-1,1] if c.direction=="either" else [-1] if c.direction=="opposed" else [1]
  possible=[angle(np.asarray(aa["axis"])*s,np.asarray(ab["axis"])*t) for s in signs(first) for t in signs(second)]
  mismatch=min(abs(value-b_angle) for value in possible)
  # 1.35 axis stretch can change a right angle by about 17 degrees.
  # A generous 30 degree bound detects gross inconsistencies, not noise.
  if mismatch<=30:continue
  ids=[first.a,first.b,second.a,second.b]
  issues.append({"code":"inconsistent_axis_pairs","features":ids,
    "a_angle_degrees":a_angle,"b_angle_degrees":b_angle,"minimum_mismatch_degrees":mismatch,
    "repair":"Re-ground the directional features. Two vertical surface points do not define an outward facing direction; use a measured surface normal where appropriate. Scaling cannot resolve this angular contradiction."})
  if (first.a,first.b) in primary and (second.a,second.b) not in primary:suspended.add((second.a,second.b))
  elif (second.a,second.b) in primary and (first.a,first.b) not in primary:suspended.add((first.a,first.b))
 return issues,suspended


def repair_feedback(checked):
 """Causal, measured execution failures; never change the user's goal."""
 feedback=[]
 for issue in checked["metrics"].get("orientation_issues",[]):
  feedback.append(issue)
 for fit in checked["metrics"].get("interface_fit",[]):
  pair=[fit["a"],fit["b"]]
  def add(code,value,action):feedback.append({"code":code,"features":pair,"measured":value,"repair":action})
  roof=fit.get("roof_gap_ratio")
  if roof is not None and roof<-.03:
   add("insufficient_insertion_depth",roof,"The inserted portion crosses the closed end. Reconsider insertion side/axis or the seating plane; do not just enlarge clearance.")
  if roof is not None and roof>.5:
   add("excess_closed_end_headroom",roof,"The interface may be too near the object's tip. Check the intended seating plane in the renders; preserve the intended part and re-ground if needed.")
  if fit["gap_median_ratio"]>fit["limits"]["median_gap_ratio"]:
   add("loose_interface",fit["gap_median_ratio"],"Check the target region and axis first, then reduce measured relative opening size; increasing clearance makes this gap larger.")
  if fit["intrusion_fraction"]>fit["limits"]["intrusion_fraction"]:
   add("local_surface_intrusion",fit["intrusion_fraction"],"Correct centering/orientation or bounded per-axis fit; do not waive collision checks.")
  if fit["supported_fraction"]<fit["limits"]["supported_fraction"]:
   add("incomplete_profile_support",fit["supported_fraction"],"The transformed opening extends beyond measured target profiles. Check insertion side, axis and seating plane before changing scale.")
 return feedback
