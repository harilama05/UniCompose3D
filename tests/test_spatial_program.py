"""A category-independent program must generalize by constraints, not labels."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import trimesh
from pydantic import ValidationError
from unicompose3d.clients import Gemini, ProviderError
from unicompose3d.constraint_solver import ProgramSolver, build_candidates
from unicompose3d.schemas import PlacementPlan, RegionHints, SpatialProgram, Transform

def proposal(program, relation='an arbitrary user relationship'):
    return PlacementPlan(relation=relation, base_anchor=[0.5, 1, 0.5], product_anchor=[0.5, 0, 0.5], rotation_degrees=[0, 0, 0], relative_scale=0.3, confidence=0.9, explanation='Proposal for testing the spatial program.', program=SpatialProgram.model_validate(program))

def contact_program():
    return {'features': [{'id': 'a', 'slot': 'A', 'kind': 'surface', 'point': [0.5, 1, 0.5]}, {'id': 'b', 'slot': 'B', 'kind': 'surface', 'point': [0.5, 0, 0.5]}], 'constraints': [{'kind': 'contact', 'a': 'a', 'b': 'b', 'target': 0.005}, {'kind': 'proportion', 'a': 'a', 'b': 'b', 'target': 0.3}]}

class SpatialProgramTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.a, self.b = (self.root / 'a.glb', self.root / 'b.glb')
        trimesh.creation.box().export(self.a)
        trimesh.creation.box().export(self.b)

    def tearDown(self):
        self.tmp.cleanup()

    def solver(self, program, relation='unknown', hints=None):
        return ProgramSolver(self.a, self.b, Transform(), proposal(program, relation), [], hints)

    def test_changing_relation_label_does_not_change_solution(self):
        one = self.solver(contact_program(), 'place the wallet on the shelf')
        two = self.solver(contact_program(), 'a never-seen category or relationship')
        first, _ = one.optimize(one.seed())
        second, _ = two.optimize(two.seed())
        np.testing.assert_allclose(first.position, second.position, atol=1e-09)
        np.testing.assert_allclose(first.rotation, second.rotation, atol=1e-09)
        self.assertTrue(one.check(first)['eligible'])

    def test_multiple_landmarks_solve_rotation_translation_and_scale_together(self):
        program = {'features': [{'id': 'a1', 'slot': 'A', 'kind': 'point', 'point': [0.2, 1, 0.5]}, {'id': 'a2', 'slot': 'A', 'kind': 'point', 'point': [0.8, 1, 0.5]}, {'id': 'b1', 'slot': 'B', 'kind': 'point', 'point': [1, 0.2, 0.5]}, {'id': 'b2', 'slot': 'B', 'kind': 'point', 'point': [1, 0.8, 0.5]}], 'constraints': [{'kind': 'coincident', 'a': 'a1', 'b': 'b1'}, {'kind': 'coincident', 'a': 'a2', 'b': 'b2'}]}
        solver = self.solver(program)
        seed = solver.seed()
        transform, optimization = solver.optimize(seed)
        self.assertLess(optimization['objective_after'], optimization['objective_before'])
        self.assertTrue(all((np.sqrt(np.mean(solver.value(c, transform) ** 2)) < 0.1 for c in solver.program.constraints)))
        self.assertGreater(transform.scale[0], 0.9)

    def test_same_primitives_cover_surrounding_targets_of_different_proportions(self):
        for radius, length in ((0.07, 0.8), (0.16, 1.2), (0.035, 0.5)):
            with self.subTest(radius=radius):
                trimesh.creation.cylinder(radius=radius, height=length, sections=48).export(self.a)
                trimesh.creation.torus(major_radius=0.3, minor_radius=0.05).export(self.b)
                program = {'features': [{'id': 'a', 'slot': 'A', 'kind': 'section', 'point': [1, 0.5, 0.5], 'end': [1, 0.5, 0.9]}, {'id': 'b', 'slot': 'B', 'kind': 'aperture'}], 'constraints': [{'kind': 'coincident', 'a': 'a', 'b': 'b'}, {'kind': 'align', 'a': 'a', 'b': 'b'}, {'kind': 'clearance', 'a': 'a', 'b': 'b', 'target': 1.05}]}
                solver = self.solver(program)
                transform, _ = solver.optimize(solver.seed())
                self.assertTrue(solver.check(transform)['eligible'], solver.check(transform))
                self.assertEqual(proposal(program).fit_mode, 'generic')

    def test_offset_expresses_beside_or_below_without_relation_dispatch(self):
        for offset in ([1, 0, 0], [0, -1, 0]):
            program = {'features': [{'id': 'a', 'slot': 'A', 'kind': 'point'}, {'id': 'b', 'slot': 'B', 'kind': 'point'}], 'constraints': [{'kind': 'offset', 'a': 'a', 'b': 'b', 'offset': offset}]}
            solver = self.solver(program, 'arbitrary relation')
            transform, _ = solver.optimize(solver.seed())
            np.testing.assert_allclose(transform.position, offset, atol=0.01)
            self.assertTrue(solver.check(transform)['eligible'])

    def test_containment_does_not_allow_solid_material_penetration(self):
        program = {'features': [{'id': 'a', 'slot': 'A', 'kind': 'region'}, {'id': 'b', 'slot': 'B', 'kind': 'point'}], 'constraints': [{'kind': 'containment', 'a': 'a', 'b': 'b'}]}
        solver = self.solver(program)
        checked = solver.check(solver.seed())
        self.assertFalse(checked['eligible'])
        self.assertTrue(any(('material' in item for item in checked['failures'])))

    def test_real_empty_cavity_can_contain_a_product(self):
        trimesh.creation.annulus(r_min=0.35, r_max=0.5, height=1, sections=48).export(self.a)
        program = {'features': [{'id': 'a', 'slot': 'A', 'kind': 'region'}, {'id': 'b', 'slot': 'B', 'kind': 'point'}], 'constraints': [{'kind': 'containment', 'a': 'a', 'b': 'b'}]}
        solver = self.solver(program)
        self.assertTrue(solver.check(solver.seed())['eligible'])

    def test_solid_product_cannot_engulf_a_without_reverse_collision_check(self):
        solver = self.solver(contact_program())
        checked = solver.check(Transform(scale=[3, 3, 3]))
        self.assertFalse(checked['eligible'])
        self.assertGreater(checked['metrics']['base_inside_product_fraction'], 0.05)

    def test_proportion_combines_with_offset_in_same_program(self):
        program = {'features': [{'id': 'a', 'slot': 'A', 'kind': 'point'}, {'id': 'b', 'slot': 'B', 'kind': 'point'}], 'constraints': [{'kind': 'offset', 'a': 'a', 'b': 'b', 'offset': [1, 0, 0]}, {'kind': 'proportion', 'a': 'a', 'b': 'b', 'target': 0.6}]}
        solver = self.solver(program)
        transform, _ = solver.optimize(solver.seed())
        self.assertAlmostEqual(transform.scale[0], 0.6, delta=0.01)
        self.assertTrue(solver.check(transform)['eligible'])

    def test_user_landmark_overrides_model_landmark(self):
        program = contact_program()
        program['features'][0]['source'] = 'user'
        solver = self.solver(program, hints=RegionHints(A_point=[0.3, 0.5, 0]))
        np.testing.assert_allclose(solver.features['a']['center'], [0.3, 0.5, 0], atol=1e-09)
        with self.assertRaisesRegex(ValueError, 'user'):
            self.solver(program)

    def test_bad_reference_duplicate_and_wrong_slot_are_rejected(self):
        for update in (lambda p: p['constraints'][0].update(a='missing'), lambda p: p['features'][1].update(id='a'), lambda p: p['features'][0].update(slot='B')):
            program = contact_program()
            update(program)
            with self.assertRaises(ValidationError):
                SpatialProgram.model_validate(program)

    def test_invalid_axis_program_is_reviewable_instead_of_silent_fallback(self):
        program = {'features': [{'id': 'a', 'slot': 'A', 'kind': 'point'}, {'id': 'b', 'slot': 'B', 'kind': 'point'}], 'constraints': [{'kind': 'align', 'a': 'a', 'b': 'b'}]}
        candidates, diagnostic = build_candidates(self.a, self.b, proposal(program), [], Transform(), 3)
        self.assertIn('feature_error', diagnostic)
        self.assertTrue(all((not c['eligible'] for c in candidates)))

    def test_capped_product_cannot_fake_an_aperture(self):
        program = {'features': [{'id': 'a', 'slot': 'A', 'kind': 'section', 'point': [1, 0.5, 0.5], 'end': [1, 0.5, 0.9]}, {'id': 'b', 'slot': 'B', 'kind': 'aperture'}], 'constraints': [{'kind': 'clearance', 'a': 'a', 'b': 'b', 'target': 1.05}]}
        candidates, diagnostic = build_candidates(self.a, self.b, proposal(program), [], Transform(), 3)
        self.assertIn('feature_error', diagnostic)
        self.assertFalse(any((c['eligible'] for c in candidates)))

    def test_deformation_requirement_is_not_reported_as_rigid_success(self):
        program = contact_program()
        program['deformation_required'] = True
        solver = self.solver(program)
        checked = solver.check(solver.seed())
        self.assertTrue(any(('deformation' in warning for warning in checked['warnings'])))

    def test_provider_must_return_a_program_not_just_prose_and_anchors(self):
        transport = Mock()
        transport.post.return_value.status_code = 200
        proposed = proposal(contact_program())
        proposed.program = None
        transport.post.return_value.json.return_value = {'candidates': [{'content': {'parts': [{'text': proposed.model_dump_json()}]}}]}
        with self.assertRaisesRegex(ProviderError, 'program'):
            Gemini('fake-key', 'test', transport).plan({}, {}, '', [])
        self.assertEqual(transport.post.call_count, 1)

    def test_api_passes_hints_to_planner_and_includes_them_in_cache(self):
        from fastapi.testclient import TestClient
        from unicompose3d.api import create_app
        planner = Mock(spec=Gemini)
        planner.key, planner.model = ('fake', 'offline')
        graph = contact_program()
        graph['features'][0]['source'] = 'user'
        planner.plan.return_value = proposal(graph)
        app = create_app(self.root / 'store', 'test-token', planner)
        engine = app.state.engine
        try:
            a, b = (engine.asset(self.a, 'arbitrary A'), engine.asset(self.b, 'previously unseen B'))
            client = TestClient(app)
            client.headers['Authorization'] = 'Bearer test-token'
            identity = np.eye(4).flatten(order='F').tolist()
            body = {'base_id': a['id'], 'product_id': b['id'], 'candidate_count': 3, 'hints': {'A_point': [0.2, 0.5, 0]}, 'captures': [{'slot': slot, 'image': '', 'world': identity, 'projection': identity} for slot in ('A', 'B')]}
            first = client.post('/api/compositions', json=body)
            self.assertEqual(first.status_code, 200, first.text)
            self.assertEqual(first.json()['diagnostics']['program_source'], 'AI')
            self.assertEqual(planner.plan.call_args.args[0]['user_hints']['point_local'], [0.2, 0.5, 0])
            again = client.post('/api/compositions', json=body).json()
            self.assertEqual(first.json()['id'], again['id'])
            body['hints']['A_point'] = [0.3, 0.5, 0]
            changed = client.post('/api/compositions', json=body).json()
            self.assertNotEqual(first.json()['id'], changed['id'])
            self.assertEqual(planner.plan.call_count, 2)
        finally:
            engine.pool.shutdown(wait=True)
if __name__ == '__main__':
    unittest.main()
