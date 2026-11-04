"""Independent spatial and semantic regression checks; no paid API calls."""
import sys,tempfile,unittest,itertools
from pathlib import Path
import numpy as np,trimesh
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from composition_cases import cases
from unicompose3d.constraint_solver import ProgramSolver,build_candidates
from unicompose3d.schemas import Transform,SpatialConstraint,SpatialFeature,SemanticIntent,CandidateReview,VisualEvaluation
from unicompose3d.orientation import conflicts
from unicompose3d.spatial_context import landmarks,reference
from unicompose3d import goal_contract,composition

class UniversalRegressionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def test_geometry_matrix_has_expected_outcomes_without_category_rules(self):
  for name,a,b,p,base,expected in cases():
   with self.subTest(case=name):
    pa,pb=self.root/"a.glb",self.root/"b.glb";a.export(pa);b.export(pb)
    candidates,diag=build_candidates(pa,pb,p,[],base,4)
    self.assertEqual(any(c["eligible"] for c in candidates),expected,(name,candidates))
    region=getattr(p,"_benchmark_region_check",None)
    if region:self.assertGreaterEqual(diag["resolved_features"][region["feature"]]["center"][region["axis"]]*region["sign"],region["minimum"])
    self.assertLessEqual(len(candidates),4)
 def test_report_axis_contradiction_is_detected_before_trying_to_scale_it_away(self):
  features={k:{"axis":np.array(axis),"kind":kind} for k,axis,kind in [
   ("s",[0,-1,0],"section"),("c",[0,1,0],"cavity"),
   ("f",[0,.9805931133995166,-.1960539363375371],"axis"),
   ("b",[-.950627043028908,-.07257417459358464,-.30173036679156884],"axis")]}
  constraints=[SpatialConstraint(kind="align",a="s",b="c"),SpatialConstraint(kind="clearance",a="s",b="c",target=1.03),
   SpatialConstraint(kind="align",a="f",b="b",direction="parallel")]
  issues,suspended=conflicts(features,constraints)
  self.assertGreater(issues[0]["minimum_mismatch_degrees"],70)
  self.assertIn(("f","b"),suspended)
 def test_direction_check_is_rotation_invariant_and_accepts_opposed_normals(self):
  features={k:{"axis":np.array(v),"kind":"axis"} for k,v in {"a":[0,1,0],"b":[0,1,0],"x":[1,0,0],"y":[-1,0,0]}.items()}
  constraints=[SpatialConstraint(kind="align",a="a",b="b",direction="parallel"),
   SpatialConstraint(kind="align",a="x",b="y",direction="opposed")]
  for rotation in (Rotation.identity(),Rotation.from_euler("XYZ",[.4,1.1,-.3])):
   moved={k:{**v,"axis":rotation.apply(v["axis"])} for k,v in features.items()}
   self.assertFalse(conflicts(moved,constraints)[0])
 def test_landmark_ids_resolve_actual_vertices_and_unknown_ids_fail(self):
  mesh=trimesh.creation.icosphere(subdivisions=1)
  measured=landmarks(mesh)
  self.assertEqual(len({x["id"] for x in measured}),24)
  for item in measured:np.testing.assert_allclose(reference(mesh,item["id"]),item["point"])
  with self.assertRaises(ValueError):reference(mesh,"L99")
 def test_semantic_contract_survives_coordinate_revision_and_rejects_unconfirmed_review(self):
  _,_,_,p,_,_=next(cases())
  p.intent=SemanticIntent(target_region="requested upper support",moving_region="bottom",spatial_relationship="supported",orientation="outward")
  contract=goal_contract.derive(p)
  changed=p.model_copy(deep=True);changed.intent.target_region="wrong side"
  self.assertEqual(contract["intent"]["target_region"],"requested upper support")
  candidate={"id":"c0","baseline":False,"eligible":True,"geometry_score":1.,"warnings":[],"failures":[]}
  review=CandidateReview(candidate_id="c0",acceptable=True,contact_score=1,placement_score=1,proportion_score=1,appearance_score=1,explanation="unverified")
  result=composition.rank_candidates([candidate],VisualEvaluation(reviews=[review],summary="prior only"),semantic_required=True)
  self.assertIsNone(result["selected_candidate_id"])
  review.intent_satisfied=True
  result=composition.rank_candidates([candidate],VisualEvaluation(reviews=[review],summary="visible"),semantic_required=True)
  self.assertEqual(result["selected_candidate_id"],"c0")
 def test_failed_geometry_is_never_accepted_even_when_visual_model_says_yes(self):
  candidate={"id":"c0","baseline":False,"eligible":False,"geometry_score":0.,"warnings":[],"failures":["solid overlap"]}
  review=CandidateReview(candidate_id="c0",acceptable=True,intent_satisfied=True,contact_score=1,placement_score=1,proportion_score=1,appearance_score=1,explanation="incorrect claim")
  result=composition.rank_candidates([candidate],VisualEvaluation(reviews=[review],summary="claimed pass"),semantic_required=True)
  self.assertIsNone(result["selected_candidate_id"]);self.assertFalse(result["auto_apply"])
 def test_search_contains_real_direction_branches_and_distinct_poses(self):
  name,a,b,p,base,_=next(x for x in cases() if x[0]=="cavity_negative")
  pa,pb=self.root/"a.glb",self.root/"b.glb";a.export(pa);b.export(pb)
  candidates,diag=build_candidates(pa,pb,p,[],base,4)
  self.assertEqual(diag["search"]["direction_branches"],2)
  self.assertGreaterEqual(diag["search"]["sampled_starts"],2)
  self.assertLessEqual(diag["hypothesis_search"]["starts_used"],diag["hypothesis_search"]["max_starts"])
  eligible=[c for c in candidates if c["eligible"]]
  self.assertTrue(eligible)
  self.assertTrue(any(c["metrics"]["direction_branch"].get("section")==-1 for c in eligible))
  for first,second in itertools.combinations([c for c in candidates if not c["baseline"]],2):
   a,b=first["transform"],second["transform"]
   separation=np.linalg.norm(np.array(a["position"])-b["position"])
   angle=(Rotation.from_euler("XYZ",a["rotation"])*Rotation.from_euler("XYZ",b["rotation"]).inv()).magnitude()
   scale=np.max(np.abs(np.log(np.array(a["scale"])/b["scale"])))
   self.assertTrue(separation>=.015 or angle>=np.deg2rad(8) or scale>=.035)
if __name__=="__main__":unittest.main()

import test_agent_scale as fixture
class IntentFlowTests(unittest.TestCase):
 setUp=fixture.AgentLoopTests.setUp
 tearDown=fixture.AgentLoopTests.tearDown
 prepare=fixture.AgentLoopTests.prepare
 evaluate=fixture.AgentLoopTests.evaluate
 def test_planner_receives_measured_landmarks_and_explicit_coordinate_frame(self):
  self.prepare()
  descriptor=self.llm.plan.call_args.args[0]
  self.assertTrue(descriptor["landmarks"])
  self.assertEqual(len(descriptor["landmarks"][0]["point"]),3)
  self.assertTrue(descriptor["coordinate_frame"]["Y_up"])
 def test_visual_judge_receives_immutable_original_semantic_goal(self):
  from test_agent_scale import contact
  first=contact();first.intent=SemanticIntent(target_region="original target",moving_region="bottom",spatial_relationship="support",orientation="outward")
  changed=first.model_copy(deep=True);changed.intent.target_region="wrong new target"
  self.llm.plan.return_value=first;self.llm.revise.return_value=changed
  one=self.evaluate(self.prepare(),False)
  two=self.prepare(refinement_of=one["id"])
  self.evaluate(two,False)
  judged=self.llm.evaluate.call_args.args[0]
  self.assertEqual(judged["original_goal_contract"]["intent"]["target_region"],"original target")
  self.assertEqual(judged["original_goal_contract"]["original_instruction"],self.body["instruction"])

class BudgetedHypothesisTests(unittest.TestCase):
 def test_luna_program_alternative_is_executed_under_a_shared_budget(self):
  from unicompose3d.schemas import SpatialHypothesis
  name,a,b,good,base,_=next(cases())
  primary=good.model_copy(deep=True)
  primary.program.constraints=[c for c in primary.program.constraints if c.kind!="proportion"]
  primary.hypotheses=[SpatialHypothesis(id="alternate",reason="same support goal with a measured size relation",program=good.program)]
  with tempfile.TemporaryDirectory() as tmp:
   pa,pb=Path(tmp)/"a.glb",Path(tmp)/"b.glb";a.export(pa);b.export(pb)
   found,diag=build_candidates(pa,pb,primary,[],base,4,budget={"max_starts":6,"max_programs":2,"max_seconds":10})
  self.assertTrue(any(c["eligible"] and c.get("hypothesis",{}).get("id")=="alternate" for c in found))
  self.assertLessEqual(diag["hypothesis_search"]["starts_used"],6)
  self.assertLessEqual(diag["hypothesis_search"]["programs"],2)
  self.assertIn("semantic region",diag["hypothesis_search"]["dimensions"])
 def test_user_selected_anchors_are_not_perturbed_by_local_search(self):
  from unicompose3d.spatial_search import local_programs
  _,a,b,p,base,_=next(cases());p.program.features[0].source="user";p.program.features[1].source="user"
  with tempfile.TemporaryDirectory() as tmp:
   pa,pb=Path(tmp)/"a.glb",Path(tmp)/"b.glb";a.export(pa);b.export(pb)
   self.assertEqual(local_programs(p,pa,pb),[])

class GeometryObservationTests(unittest.TestCase):
 def test_local_normal_averages_high_frequency_triangle_noise(self):
  from unicompose3d.geometry import surface_patch_frame
  rng=np.random.default_rng(42);n=31
  x,y=np.meshgrid(np.linspace(-.5,.5,n),np.linspace(-.5,.5,n))
  vertices=np.column_stack([x.ravel(),y.ravel(),rng.normal(0,.003,n*n)])
  faces=[]
  for j in range(n-1):
   for i in range(n-1):
    a=j*n+i;faces.extend([[a,a+1,a+n],[a+1,a+n+1,a+n]])
  mesh=trimesh.Trimesh(vertices,faces,process=False)
  center,normal=surface_patch_frame(mesh,[0,0,0],.12)
  self.assertGreater(normal[2],.98)
  self.assertLess(abs(center[2]),.02)
 def test_local_search_cannot_walk_outside_the_original_region(self):
  from unicompose3d.spatial_search import within_original_support
  _,a,b,p,_,_=next(cases());moved=p.model_copy(deep=True)
  p.program.features[0].extent=[.2,.2,.2]
  moved.program.features[0].point=[.9,1,.5]
  with tempfile.TemporaryDirectory() as tmp:
   pa,pb=Path(tmp)/"a.glb",Path(tmp)/"b.glb";a.export(pa);b.export(pb)
   self.assertFalse(within_original_support(p,moved,pa,pb))
 def test_direction_checker_accepts_json_lists_as_well_as_numpy_vectors(self):
  f={"a":{"axis":[0,1,0]},"b":{"axis":[0,1,0]},"x":{"axis":[0,1,0]},"y":{"axis":[1,0,0]}}
  c=[SpatialConstraint(kind="align",a="a",b="b"),SpatialConstraint(kind="align",a="x",b="y",direction="parallel")]
  self.assertTrue(conflicts(f,c)[0])

class ProjectedHeadingTests(unittest.TestCase):
 def test_heading_compares_azimuth_without_forcing_a_sloping_normal_to_tilt_the_interface(self):
  name,a,b,p,base,_=next(x for x in cases() if x[0]=="cavity_positive")
  p.program.features.extend([SpatialFeature(id="facing",slot="A",kind="surface",point=[1,.5,.3]),
   SpatialFeature(id="heading",slot="B",kind="axis",point=[.1,.5,.2],end=[.9,.5,.8])])
  p.program.constraints.append(SpatialConstraint(kind="align",a="facing",b="heading",direction="parallel",frame_a="section",frame_b="opening"))
  with tempfile.TemporaryDirectory() as tmp:
   pa,pb=Path(tmp)/"a.glb",Path(tmp)/"b.glb";a.export(pa);b.export(pb)
   found,diag=build_candidates(pa,pb,p,[],base,4,budget={"max_starts":4,"max_programs":1,"max_seconds":20})
  self.assertTrue(any(c["eligible"] for c in found),found)
 def test_planar_heading_frame_cannot_reference_a_point_without_an_axis(self):
  from unicompose3d.schemas import SpatialProgram
  with self.assertRaises(ValueError):SpatialProgram(features=[SpatialFeature(id="a",slot="A",kind="point"),SpatialFeature(id="b",slot="B",kind="axis")],constraints=[SpatialConstraint(kind="align",a="a",b="b",frame_a="a",frame_b="b")])

class ProfileRecoveryTests(unittest.TestCase):
 def test_wrong_closed_fragment_does_not_hide_a_verified_surrounding_surface(self):
  from unittest.mock import patch
  from unicompose3d import aperture
  _,a,b,p,base,_=next(x for x in cases() if x[0]=="cavity_positive")
  with tempfile.TemporaryDirectory() as tmp:
   pa,pb=Path(tmp)/"a.glb",Path(tmp)/"b.glb";a.export(pa);b.export(pb)
   solver=ProgramSolver(pa,pb,base,p,[])
   solver=next(branch for branch in solver.branches() if not branch.insertion_feasibility())
   original=aperture.target_section
   def misleading(mesh,point,axis):
    result=original(mesh,point,axis)
    if mesh is solver.base and .12<point[2]<.16:
     result=dict(result);result["center"]=result["center"]+np.array([1.,0,0])
    return result
   with patch("unicompose3d.aperture.target_section",side_effect=misleading):seed=solver.seed()
   interface=next(iter(solver.fit_interfaces.values()))
   measured=interface.measure(seed)
   self.assertGreaterEqual(measured["supported_fraction"],.95)
   self.assertGreater(measured["target_radial_recoveries"],0)
   self.assertTrue(measured["eligible"],measured)
