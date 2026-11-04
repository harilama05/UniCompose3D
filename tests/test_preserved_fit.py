"""Reported regression: preserve a through-fit and keep successful tool facts."""
import tempfile, unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np, trimesh
from fastapi.testclient import TestClient
from unicompose3d import aperture, goal_contract
from unicompose3d.constraint_solver import ProgramSolver, build_candidates
from unicompose3d.geometry_feedback import FeatureResolutionError
from unicompose3d.schemas import SpatialProgram, PlacementPlan, Transform, Capture
from unicompose3d.api import create_app
from unicompose3d.clients import Gemini
from test_reported_issues import camera
from test_agent_scale import png

def fit_plan(point=None, end=None):
    return PlacementPlan(relation='arbitrary through fitting', base_anchor=[1, 0.5, 0.5], product_anchor=[0.5, 0.5, 0.5], rotation_degrees=[0, 0, 0], relative_scale=0.3, confidence=0.9, explanation='Measured fit', program=SpatialProgram(features=[{'id': 'target', 'slot': 'A', 'kind': 'section', 'point': point or [1, 0.5, 0.5], 'end': end}, {'id': 'opening', 'slot': 'B', 'kind': 'aperture'}], constraints=[{'kind': 'clearance', 'a': 'target', 'b': 'opening', 'target': 1.05}]))

def weak_plan():
    plan = fit_plan()
    plan.program = SpatialProgram(features=[{'id': 'a1', 'slot': 'A', 'kind': 'point', 'point': [0.5, 1, 0.5]}, {'id': 'a2', 'slot': 'A', 'kind': 'point', 'point': [0.6, 1, 0.5]}, {'id': 'b1', 'slot': 'B', 'kind': 'point', 'point': [0.5, 0, 0.5]}, {'id': 'b2', 'slot': 'B', 'kind': 'point', 'point': [0.6, 0, 0.5]}], constraints=[{'kind': 'coincident', 'a': 'a1', 'b': 'b1'}, {'kind': 'coincident', 'a': 'a2', 'b': 'b2'}])
    return plan

class PreservedFitTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.a, self.b = (self.root / 'a.glb', self.root / 'b.glb')
        trimesh.creation.cylinder(radius=0.04, height=1, sections=48).export(self.a)
        trimesh.creation.torus(major_radius=0.3, minor_radius=0.05).export(self.b)

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_landmarks_cannot_replace_a_required_through_fit(self):
        required = goal_contract.derive(fit_plan(end=[1, 0.5, 0.9]))
        candidates, diagnostics = build_candidates(self.a, self.b, weak_plan(), [], Transform(), 1, goal_contract=required)
        self.assertIn('goal_error', diagnostics)
        self.assertFalse(candidates[0]['eligible'])
        self.assertEqual(candidates[0]['geometry_score'], 0)

    def test_open_boundary_does_not_discard_valid_radial_measurement(self):
        mesh = trimesh.creation.torus(major_radius=0.3, minor_radius=0.05, major_sections=64, minor_sections=24)
        angle = np.arctan2(mesh.triangles_center[:, 1], mesh.triangles_center[:, 0])
        mesh.update_faces(~((angle > 0.1) & (angle < 0.2)))
        self.assertFalse(mesh.is_watertight)
        faces = len(mesh.faces)
        frame = aperture.hole_frame(mesh)
        self.assertGreater(frame['angular_coverage'], 0.7)
        self.assertAlmostEqual(frame['inner_radius'], 0.25, delta=0.02)
        self.assertEqual(len(mesh.faces), faces)

    def test_other_feature_still_measured_when_section_direction_is_missing(self):
        with self.assertRaises(FeatureResolutionError) as context:
            ProgramSolver(self.a, self.b, Transform(), fit_plan(), [])
        observed = context.exception.measurements
        self.assertIn('opening', observed['resolved_features'])
        self.assertGreater(observed['resolved_features']['opening']['radius'], 0.2)
        self.assertEqual(observed['feature_issues'][0]['feature_id'], 'target')

    def test_uv_failure_includes_nearest_surface_coordinates_and_image(self):
        from unicompose3d.agent_loop import grounding_images
        plan = fit_plan(end=[1, 0.5, 0.9])
        feature = plan.program.features[0]
        feature.uv = [0.99, 0.99]
        feature.view = 0
        capture = camera('A')
        capture.image = png()
        with self.assertRaises(FeatureResolutionError) as context:
            ProgramSolver(self.a, self.b, Transform(), plan, [capture])
        observed = context.exception.measurements
        nearest = observed['feature_issues'][0]['measurements']['nearest_surface']
        self.assertIsNotNone(nearest)
        self.assertGreater(nearest['distance_uv'], 0.012)
        images = grounding_images([capture], observed)
        self.assertEqual(images[0]['kind'], 'grounding')

    def test_agent_internal_repair_cannot_relax_the_original_goal(self):
        llm = Mock(spec=Gemini)
        llm.key = 'fixture'
        llm.model = 'offline'
        llm.plan.side_effect = [fit_plan(), weak_plan()]
        app = create_app(self.root / 'data', 'token', llm)
        engine = app.state.engine
        a, b = (engine.asset(self.a, 'A'), engine.asset(self.b, 'B'))
        eye = np.eye(4).flatten(order='F').tolist()
        body = {'base_id': a['id'], 'product_id': b['id'], 'instruction': 'original intent', 'captures': [{'slot': slot, 'image': png(), 'projection': eye, 'world': eye} for slot in ('A', 'B')]}
        client = TestClient(app)
        client.headers['Authorization'] = 'Bearer token'
        try:
            response = client.post('/api/compositions', json=body)
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertTrue(result['goal_contract']['fits'])
            self.assertIn('goal_error', result['diagnostics'])
            self.assertFalse(result['candidates'][0]['eligible'])
            initial = result['diagnostics']['previous_attempts'][0]['diagnostics']
            self.assertIn('opening', initial['resolved_features'])
            self.assertIn('Geometry tool observations', llm.plan.call_args.args[2])
        finally:
            engine.pool.shutdown(wait=True)
            engine.segments.close()

    def test_fit_search_does_not_scale_to_a_wider_junction(self):
        profile = np.array([[0, -0.6], [0.2, -0.6], [0.2, -0.1], [0.04, 0.02], [0.04, 0.6], [0, 0.6]])
        mesh = trimesh.creation.revolve(profile, sections=64)
        mesh.export(self.a)
        start = np.array([0.18, 0, -0.1])
        end = np.array([0.04, 0, 0.35])
        fraction = lambda p: ((p - mesh.bounds[0]) / mesh.extents).tolist()
        plan = fit_plan(fraction(start), fraction(end))
        plan.program.features[0].extent = [0.2, 0.2, 0.07]
        solver = ProgramSolver(self.a, self.b, Transform(), plan, [])
        seed = solver.seed()
        self.assertLess(seed.scale[0], 0.3)
        self.assertGreater(solver.features['target']['center'][2], 0.0)
        self.assertTrue(solver.check(seed)['eligible'], solver.check(seed))
if __name__ == '__main__':
    unittest.main()
