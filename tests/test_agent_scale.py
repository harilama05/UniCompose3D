"""Local size compatibility and bounded, visual-feedback composition agent."""
import base64
import io
import json
import zipfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import trimesh
from PIL import Image
from fastapi.testclient import TestClient
from unicompose3d.api import create_app
from unicompose3d.clients import Gemini
from unicompose3d.constraint_solver import ProgramSolver, build_candidates
from unicompose3d.schemas import PlacementPlan, SpatialProgram, Transform, RegionHints, VisualEvaluation, CandidateReview

def proposal(features, constraints):
    return PlacementPlan(relation='arbitrary goal', base_anchor=[0.5, 0.5, 0.5], product_anchor=[0.5, 0.5, 0.5], rotation_degrees=[0, 0, 0], relative_scale=3, confidence=0.95, explanation='Test measured compatibility.', program=SpatialProgram(features=features, constraints=constraints))

def contact():
    return proposal([{'id': 'a', 'slot': 'A', 'kind': 'surface', 'point': [0.5, 1, 0.5]}, {'id': 'b', 'slot': 'B', 'kind': 'surface', 'point': [0.5, 0, 0.5]}], [{'kind': 'contact', 'a': 'a', 'b': 'b', 'target': 0.005}, {'kind': 'proportion', 'a': 'a', 'b': 'b', 'target': 0.3}])

def png():
    stream = io.BytesIO()
    Image.new('RGB', (256, 256), 'white').save(stream, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()

class LocalScaleTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.a, self.b = (self.root / 'a.glb', self.root / 'b.glb')

    def tearDown(self):
        self.temp.cleanup()

    def test_opening_scales_to_local_part_even_when_whole_base_is_large(self):
        cylinder = trimesh.creation.cylinder(radius=0.025, height=0.8, sections=48)
        palm = trimesh.creation.box(extents=[1, 0.3, 0.2])
        palm.apply_translation([0, 0, -0.55])
        trimesh.util.concatenate([cylinder, palm]).export(self.a)
        trimesh.creation.torus(major_radius=0.3, minor_radius=0.04).export(self.b)
        p = proposal([{'id': 'a', 'slot': 'A', 'kind': 'section', 'source': 'user'}, {'id': 'b', 'slot': 'B', 'kind': 'aperture'}], [{'kind': 'coincident', 'a': 'a', 'b': 'b'}, {'kind': 'align', 'a': 'a', 'b': 'b'}])
        solver = ProgramSolver(self.a, self.b, Transform(), p, [], RegionHints(A_point=[0.025, 0, 0.1], A_end=[0.025, 0, 0.3]), require_scale=True)
        seed = solver.seed()
        ratio = solver.features['b']['radius'] * seed.scale[0] / solver.features['a']['radius']
        self.assertAlmostEqual(ratio, 1.05, places=6)
        self.assertLess(seed.scale[0], 0.12)
        self.assertTrue(solver.check(seed)['eligible'], solver.check(seed))
        self.assertEqual(len(p.program.constraints), 2)
        self.assertTrue(any((c.kind == 'clearance' for c in solver.program.constraints)))
        huge = seed.model_copy(deep=True)
        huge.scale = [1, 1, 1]
        self.assertFalse(solver.check(huge)['eligible'])

    def test_reverse_fit_product_section_inside_base_opening(self):
        trimesh.creation.torus(major_radius=0.3, minor_radius=0.05).export(self.a)
        trimesh.creation.cylinder(radius=0.5, height=2, sections=48).export(self.b)
        p = proposal([{'id': 'a', 'slot': 'A', 'kind': 'aperture'}, {'id': 'b', 'slot': 'B', 'kind': 'section', 'point': [1, 0.5, 0.5], 'end': [1, 0.5, 0.8]}], [{'kind': 'coincident', 'a': 'a', 'b': 'b'}])
        solver = ProgramSolver(self.a, self.b, Transform(), p, [], require_scale=True)
        transform = solver.seed()
        self.assertAlmostEqual(solver.features['a']['radius'] / (solver.features['b']['radius'] * transform.scale[0]), 1.05, places=6)
        self.assertTrue(solver.check(transform)['eligible'], solver.check(transform))

    def test_proportion_uses_local_support_and_large_prior_does_not_limit_fit(self):
        trimesh.creation.box(extents=[10, 1, 1]).export(self.a)
        trimesh.creation.box(extents=[2, 2, 2]).export(self.b)
        p = proposal([{'id': 'a', 'slot': 'A', 'kind': 'point', 'extent': [0.02, 0.2, 0.2]}, {'id': 'b', 'slot': 'B', 'kind': 'point'}], [{'kind': 'offset', 'a': 'a', 'b': 'b', 'offset': [0, 1, 0]}, {'kind': 'proportion', 'a': 'a', 'b': 'b', 'target': 1}])
        solver = ProgramSolver(self.a, self.b, Transform(), p, [], require_scale=True)
        seed = solver.seed()
        self.assertAlmostEqual(solver.features['a']['extent'], 0.2)
        self.assertAlmostEqual(seed.scale[0], 0.1)
        fitted, _ = solver.optimize(seed)
        self.assertTrue(solver.check(fitted)['eligible'])

    def test_location_only_program_is_not_accepted_as_size_compatible(self):
        trimesh.creation.box().export(self.a)
        trimesh.creation.box().export(self.b)
        p = proposal([{'id': 'a', 'slot': 'A', 'kind': 'point'}, {'id': 'b', 'slot': 'B', 'kind': 'point'}], [{'kind': 'coincident', 'a': 'a', 'b': 'b'}])
        candidates, diagnostic = build_candidates(self.a, self.b, p, [], Transform(), 1)
        self.assertIn('scale_error', diagnostic)
        self.assertFalse(candidates[0]['eligible'])

class AgentLoopTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.llm = Mock(spec=Gemini)
        self.llm.model = 'fixture'
        self.llm.key = 'fixture'
        self.llm.plan.return_value = contact()
        self.llm.revise.return_value = contact()
        self.llm.revise.return_value.program.constraints[-1].target = 0.25
        self.app = create_app(self.root / 'data', 'test-token', self.llm)
        self.engine = self.app.state.engine
        raw = self.root / 'box.glb'
        trimesh.creation.box().export(raw)
        self.asset = self.engine.asset(raw, 'arbitrary object')
        self.client = TestClient(self.app)
        self.client.headers['Authorization'] = 'Bearer test-token'
        eye = np.eye(4).flatten(order='F').tolist()
        self.body = {'base_id': self.asset['id'], 'product_id': self.asset['id'], 'instruction': 'original goal', 'captures': [{'slot': slot, 'image': png(), 'projection': eye, 'world': eye} for slot in ('A', 'B')]}

    def tearDown(self):
        self.engine.pool.shutdown(wait=True)
        self.engine.segments.close()
        self.temp.cleanup()

    def prepare(self, **overrides):
        response = self.client.post('/api/compositions', json=self.body | overrides)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def evaluate(self, result, acceptable):
        self.llm.evaluate.return_value = VisualEvaluation(reviews=[CandidateReview(candidate_id=c['id'], acceptable=acceptable, contact_score=0.95, placement_score=0.95, proportion_score=0.95, appearance_score=0.95, explanation='Correct after revision' if acceptable else 'Wrong target region', issues=[] if acceptable else ['Move to the requested part']) for c in result['candidates']], summary='accepted' if acceptable else 'revise target')
        response = self.client.post('/api/compositions/' + result['id'] + '/evaluate', json={'renders': [{'candidate_id': c['id'], 'image': png()} for c in result['candidates']]})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_next_round_receives_geometry_render_and_original_goal(self):
        one = self.evaluate(self.prepare(), False)
        self.assertFalse(one['auto_apply'])
        two = self.prepare(refinement_of=one['id'])
        self.assertEqual(two['agent_round'], 2)
        self.assertEqual(len(two['agent_history']), 1)
        self.assertNotEqual(one['candidates'][0]['transform']['scale'], two['candidates'][0]['transform']['scale'])
        args = self.llm.revise.call_args.args
        self.assertEqual(args[2], 'original goal')
        self.assertEqual(args[4]['summary'], 'revise target')
        self.assertTrue(args[5][0]['image'])
        self.assertIn('resolved_features', args[4]['diagnostics'])
        two = self.evaluate(two, True)
        self.assertTrue(two['auto_apply'])
        self.assertEqual(self.llm.plan.call_count, 1)
        self.assertEqual(self.llm.revise.call_count, 1)
        self.assertEqual(self.llm.evaluate.call_count, 2)

    def test_server_enforces_three_round_cap_and_unchanged_intent(self):
        one = self.prepare()
        two = self.prepare(refinement_of=one['id'])
        three = self.prepare(refinement_of=two['id'])
        response = self.client.post('/api/compositions', json=self.body | {'refinement_of': three['id']})
        self.assertEqual(response.status_code, 400)
        self.assertIn('three-round limit', response.text)
        changed = self.client.post('/api/compositions', json=self.body | {'refinement_of': one['id'], 'instruction': 'different goal'})
        self.assertEqual(changed.status_code, 400)

    def test_quick_endpoint_also_uses_geometrically_fitted_scale(self):
        self.llm.plan.return_value = contact()
        response = self.client.post('/api/compose', json=self.body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()['eligible'])
        self.assertAlmostEqual(response.json()['transform']['scale'][0], 0.3, delta=0.005)

    def test_identical_refinement_is_marked_as_no_concrete_change(self):
        self.llm.revise.return_value = contact()
        one = self.evaluate(self.prepare(), False)
        two = self.prepare(refinement_of=one['id'])
        self.assertTrue(two['agent_no_change'])
        self.assertFalse(two['auto_apply'])
        self.assertEqual(self.llm.evaluate.call_count, 1)

    def test_failed_placeholder_pose_does_not_stop_a_repair_round(self):
        from test_preserved_fit import fit_plan
        self.llm.plan.return_value = fit_plan()
        self.llm.revise.return_value = fit_plan()
        one = self.prepare()
        two = self.prepare(refinement_of=one['id'])
        self.assertFalse(one['candidates'][0]['eligible'])
        self.assertFalse(two['candidates'][0]['eligible'])
        self.assertEqual(one['candidates'][0]['transform'], two['candidates'][0]['transform'])
        self.assertFalse(two['agent_no_change'])
        self.assertEqual(two['agent_round'], 2)
        self.assertEqual(self.llm.revise.call_count, 1)

    def test_authenticated_diagnostic_zip_contains_models_and_all_rounds(self):
        one = self.evaluate(self.prepare(), False)
        two = self.evaluate(self.prepare(refinement_of=one['id']), False)
        endpoint = '/api/compositions/' + two['id'] + '/diagnostics.zip'
        unauthorized = TestClient(self.app).get(endpoint)
        self.assertIn(unauthorized.status_code, (401, 403))
        response = self.client.get(endpoint)
        self.assertEqual(response.status_code, 200, response.text if response.status_code != 200 else '')
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = set(archive.namelist())
            self.assertIn('A.glb', names)
            self.assertIn('B.glb', names)
            self.assertEqual(archive.read('A.glb'), self.engine.model_path(self.asset['id']).read_bytes())
            self.assertEqual(archive.read('B.glb'), archive.read('A.glb'))
            for record in (one, two):
                prefix = 'round_' + str(record['agent_round']) + '/'
                report = json.loads(archive.read(prefix + 'report.json'))
                self.assertEqual(report['id'], record['id'])
                self.assertEqual(report['instruction'], self.body['instruction'])
                self.assertTrue(any((name.startswith(prefix) and name.endswith('.png') for name in names)))
            for name in names:
                if name.endswith('.json'):
                    self.assertNotIn('test-token', archive.read(name).decode())
            self.assertFalse(any((name.endswith(('.env', '.log')) for name in names)))

    def test_best_accepted_round_is_retained_after_a_later_rejection(self):
        one = self.evaluate(self.prepare(), False)
        self.llm.revise.return_value.program.uncertainty = 'Keep manual review for incomplete geometry'
        two = self.evaluate(self.prepare(refinement_of=one['id']), True)
        self.assertFalse(two['auto_apply'])
        self.llm.revise.return_value.program.constraints[-1].target = 0.23
        three = self.evaluate(self.prepare(refinement_of=two['id']), False)
        response = self.client.get('/api/compositions/' + three['id'] + '/agent-result')
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['id'], two['id'])
        self.assertEqual(result['agent_selected_round'], 2)
        self.assertEqual(result['agent_total_rounds'], 3)
        self.assertEqual(len(result['agent_attempts']), 3)
        self.assertFalse(result['auto_apply'])
        self.assertEqual(result['candidates'][0]['transform'], two['candidates'][0]['transform'])
        self.assertEqual(result['agent_run_id'], three['id'])

    def test_best_round_selector_cannot_promote_an_ineligible_candidate(self):
        one = self.evaluate(self.prepare(), True)
        two = self.evaluate(self.prepare(refinement_of=one['id']), True)
        broken = two['candidates']
        broken[0]['eligible'] = False
        broken[0]['combined_score'] = 1.0
        self.engine.store.update('compositions', two['id'], candidates=broken)
        result = self.client.get('/api/compositions/' + two['id'] + '/agent-result').json()
        self.assertEqual(result['id'], one['id'])

class AgentTransportTests(unittest.TestCase):

    def test_openai_revision_sends_previous_render_and_measured_feedback(self):
        from unicompose3d.openai_client import OpenAIClient
        from unicompose3d.schemas import Capture
        from test_openai_planning import response, plan_data
        http = Mock()
        http.post.return_value = response(plan_data())
        client = OpenAIClient(key='fixture-secret', transport=http)
        eye = np.eye(4).flatten(order='F').tolist()
        captures = [Capture(slot=slot, image=png(), world=eye, projection=eye) for slot in ('A', 'B')]
        result = client.revise({}, {}, 'keep original goal', captures, {'measured_radius': 0.025, 'failure': 'opening too large'}, [{'candidate_id': 'c0', 'image': png()}])
        self.assertIsNotNone(result.program)
        self.assertEqual(http.post.call_count, 1)
        body = http.post.call_args.kwargs['json']
        parts = body['input'][0]['content']
        self.assertEqual(sum((part['type'] == 'input_image' for part in parts)), 3)
        text = ' '.join((part['text'] for part in parts if part['type'] == 'input_text'))
        self.assertIn('keep original goal', text)
        self.assertIn('measured_radius', text)
        self.assertIn('Previous composed candidate', text)
if __name__ == '__main__':
    unittest.main()
