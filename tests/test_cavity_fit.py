"""Topology/fit primitives shared by closed-end covers and through openings."""
import tempfile,unittest
from pathlib import Path
import numpy as np,trimesh
from unicompose3d import cavity,aperture,goal_contract
from unicompose3d.schemas import SpatialProgram,PlacementPlan,Transform
from unicompose3d.constraint_solver import ProgramSolver

def cup():
    # Empty from z=0 towards a closed end at z=.5.
    return trimesh.creation.revolve(np.array([[.30,0],[.34,0],[.34,.50],[0,.50],[0,.46],[.30,.46],[.30,0]]),sections=64)
def plan(reverse=False):
    a,b=("space","body") if reverse else ("body","space")
    return PlacementPlan(relation="generic insertion",base_anchor=[.5]*3,product_anchor=[.5]*3,
        rotation_degrees=[0]*3,relative_scale=1,confidence=.9,explanation="geometry",
        program=SpatialProgram(fit_topology="cover",features=[
        {"id":"body","slot":"B" if reverse else "A","kind":"section","point":[1,.5,.05],"end":[1,.5,.85]},
        {"id":"space","slot":"A" if reverse else "B","kind":"cavity","point":[.5,.5,.05],"end":[.5,.5,.8]}],
        constraints=[{"kind":"clearance","a":a,"b":b,"target":1.03}]))
class CavityTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_closed_end_is_valid_cavity_but_never_a_through_aperture(self):
        mesh=cup()
        frame=cavity.cavity_frame(mesh,[0,0,1])
        self.assertTrue(frame["mouth_verified"]);self.assertGreater(frame["depth"],.35)
        with self.assertRaises(ValueError):aperture.hole_frame(mesh)
    def test_solid_and_through_structures_cannot_masquerade_as_closed_cavities(self):
        for mesh in (trimesh.creation.box(),trimesh.creation.torus(major_radius=.3,minor_radius=.04)):
            with self.assertRaises(ValueError):cavity.cavity_frame(mesh,[0,0,1])
    def test_explicit_through_intent_cannot_be_replaced_by_a_cavity(self):
        original=plan();original.program.fit_topology="through"
        original.program.features[1].kind="aperture"
        contract=goal_contract.derive(original)
        with self.assertRaises(goal_contract.GoalContractError):goal_contract.enforce(plan(),contract)
    def test_forward_and_reverse_fit_use_measured_mouth_and_depth(self):
        for reverse in (False,True):
            body=trimesh.creation.cylinder(radius=.25,height=.4,sections=64);body.apply_translation([0,0,.2])
            a,b=self.root/"a.glb",self.root/"b.glb"
            (cup() if reverse else body).export(a);(body if reverse else cup()).export(b)
            solver=ProgramSolver(a,b,Transform(),plan(reverse),[],require_scale=True)
            fitted,_=solver.optimize(solver.seed());result=solver.check(fitted)
            self.assertTrue(result["eligible"],result)
            measured=result["metrics"]["interface_fit"][0]
            self.assertEqual(measured["fit_region"],"mouth")
            self.assertIsNotNone(measured["roof_gap_ratio"])
            self.assertLess(measured["intrusion_fraction"],.05)
    def test_cover_uses_requested_plane_instead_of_searching_for_a_smaller_tip(self):
        body=trimesh.creation.cylinder(radius=.25,height=.4,sections=64);body.apply_translation([0,0,.2])
        a,b=self.root/"a.glb",self.root/"b.glb";body.export(a);cup().export(b)
        p=plan();p.program.features[0].extent=[.05,.05,.05]
        solver=ProgramSolver(a,b,Transform(),p,[])
        self.assertEqual(solver.features["body"]["search_offset"],0)
        self.assertAlmostEqual(solver.features["body"]["radius"],.25,delta=.01)
if __name__=="__main__":unittest.main()
