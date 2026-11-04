"""Generic relative-scale and interface-fit regressions, without product categories."""
import hashlib, tempfile, unittest
from pathlib import Path
import numpy as np,trimesh
from unicompose3d.constraint_solver import ProgramSolver
from unicompose3d.schemas import Transform,PlacementPlan,SpatialProgram

def proposal(reverse=False):
    section={"id":"section","slot":"B" if reverse else "A","kind":"section",
             "point":[1,.5,.5],"end":[1,.5,.85]}
    opening={"id":"opening","slot":"A" if reverse else "B","kind":"aperture"}
    a,b=("opening","section") if reverse else ("section","opening")
    return PlacementPlan(relation="arbitrary spatial relation",base_anchor=[.5,.5,.5],
        product_anchor=[.5,.5,.5],rotation_degrees=[0,0,0],relative_scale=.7,confidence=.9,
        explanation="Local geometric compatibility",
        program=SpatialProgram(features=[section,opening],
            constraints=[{"kind":"clearance","a":a,"b":b,"target":1.03}]))

class SurfaceFitTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.a,self.b=self.root/"a.glb",self.root/"b.glb"
    def tearDown(self):self.tmp.cleanup()
    def meshes(self,aspect=1,reverse=False):
        section=trimesh.creation.cylinder(radius=.07,height=.8,sections=64)
        section.apply_scale([1,aspect,1])
        opening=trimesh.creation.torus(major_radius=.3,minor_radius=.04,major_sections=64,minor_sections=24)
        (opening if reverse else section).export(self.a)
        (section if reverse else opening).export(self.b)

    def test_ellipse_gets_automatic_bounded_axis_sizing(self):
        self.meshes(.8)
        solver=ProgramSolver(self.a,self.b,Transform(),proposal(),[],require_scale=True)
        seed=solver.seed();old=solver.check(seed)["metrics"]["interface_fit"][0]
        transform,optimization=solver.optimize(seed);result=solver.check(transform)
        measured=result["metrics"]["interface_fit"][0]
        self.assertLess(measured["gap_median_ratio"],old["gap_median_ratio"])
        self.assertLessEqual(max(transform.scale)/min(transform.scale),1.35)
        self.assertGreater(max(transform.scale)/min(transform.scale),1.03)
        self.assertTrue(result["eligible"],result)
        self.assertIn("per-axis",optimization["scale_mode"])

    def test_same_measurement_and_sizing_work_in_reverse(self):
        self.meshes(.8,reverse=True)
        solver=ProgramSolver(self.a,self.b,Transform(),proposal(True),[],require_scale=True)
        transform,_=solver.optimize(solver.seed());result=solver.check(transform)
        self.assertTrue(result["eligible"],result)
        self.assertLess(result["metrics"]["interface_fit"][0]["gap_median_ratio"],.1)

    def test_impossible_profile_is_not_stretched_unbounded_or_called_fitted(self):
        self.meshes(.42)
        solver=ProgramSolver(self.a,self.b,Transform(),proposal(),[],require_scale=True)
        transform,_=solver.optimize(solver.seed());result=solver.check(transform)
        self.assertLessEqual(max(transform.scale)/min(transform.scale),1.35)
        self.assertFalse(result["eligible"])
        self.assertFalse(result["metrics"]["interface_fit"][0]["eligible"])

    def test_oversized_pose_fails_surface_test_even_with_no_intrusion(self):
        self.meshes()
        solver=ProgramSolver(self.a,self.b,Transform(),proposal(),[],require_scale=True)
        seed=solver.seed();seed.scale=[s*1.4 for s in seed.scale]
        result=solver.check(seed)
        self.assertFalse(result["eligible"])
        fit=result["metrics"]["interface_fit"][0]
        self.assertGreater(fit["gap_median_ratio"],.2)
        self.assertFalse(fit["snug"])

    def test_common_scene_scale_is_not_an_extra_fit_degree_of_freedom(self):
        self.meshes(.8);transforms=[];gaps=[]
        for scale in (1,2):
            solver=ProgramSolver(self.a,self.b,Transform(scale=[scale]*3),proposal(),[],require_scale=True)
            fitted,_=solver.optimize(solver.seed())
            transforms.append(np.array(fitted.scale)/scale)
            gaps.append(solver.check(fitted)["metrics"]["interface_fit"][0]["gap_median_ratio"])
        np.testing.assert_allclose(transforms[0],transforms[1],rtol=.03,atol=.001)
        self.assertAlmostEqual(gaps[0],gaps[1],delta=.02)

    def test_square_target_and_polygonal_aperture_use_the_same_pipeline(self):
        trimesh.creation.box(extents=[.10,.10,.8]).export(self.a)
        trimesh.creation.annulus(r_min=.25,r_max=.35,height=.12,sections=4).export(self.b)
        solver=ProgramSolver(self.a,self.b,Transform(),proposal(),[],require_scale=True)
        fitted,_=solver.optimize(solver.seed())
        result=solver.check(fitted)
        self.assertTrue(result["eligible"],result)
        self.assertLess(result["metrics"]["interface_fit"][0]["gap_median_ratio"],.1)

    def test_bad_clearance_prior_cannot_relax_gate_but_can_be_resized_to_valid_fit(self):
        self.meshes();plan=proposal();plan.program.constraints[0].target=1.5
        solver=ProgramSolver(self.a,self.b,Transform(),plan,[],require_scale=True)
        seed=solver.seed()
        original=solver.check(seed)["metrics"]["interface_fit"][0]
        self.assertFalse(original["eligible"])
        self.assertGreater(original["gap_median_ratio"],.1)
        fitted,_=solver.optimize(seed)
        fit=solver.check(fitted)["metrics"]["interface_fit"][0]
        self.assertTrue(fit["eligible"],fit)
        self.assertLess(np.mean(fitted.scale),np.mean(seed.scale))
        self.assertLessEqual(fit["gap_median_ratio"],.1)
        self.assertEqual(fit["limits"]["median_gap_ratio"],.1)

    def test_excessive_axis_stretch_is_rejected(self):
        self.meshes();solver=ProgramSolver(self.a,self.b,Transform(),proposal(),[],require_scale=True)
        fitted,_=solver.optimize(solver.seed());fitted.scale[1]*=1.6
        result=solver.check(fitted)
        self.assertFalse(result["eligible"])
        self.assertGreater(result["metrics"]["axis_scale_ratio"],1.35)

    def test_fit_changes_transforms_not_source_meshes(self):
        self.meshes(.8)
        before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.a,self.b)]
        solver=ProgramSolver(self.a,self.b,Transform(),proposal(),[],require_scale=True)
        solver.optimize(solver.seed())
        self.assertEqual(before,[hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.a,self.b)])

if __name__=="__main__":unittest.main()
