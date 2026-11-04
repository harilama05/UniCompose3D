"""New composition search/evaluation tests. No real provider calls."""
import base64
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import trimesh
from fastapi.testclient import TestClient
from PIL import Image
from unicompose3d.api import create_app
from unicompose3d.clients import Gemini, ProviderError
from unicompose3d.composition import GeometryContext, build_candidates, contains_points, rank_candidates
from unicompose3d.geometry import solve_placement
from unicompose3d.schemas import CandidateReview, PlacementPlan, Transform, VisualEvaluation

def make_plan(relation='on'):
    return PlacementPlan(relation=relation, base_anchor=[0.5, 1, 0.5], product_anchor=[0.5, 0, 0.5], rotation_degrees=[0, 0, 0], relative_scale=0.3, confidence=0.9, explanation='Place the product on the surface.')

def review(candidates, winner='c1', acceptable=True):
    return VisualEvaluation(reviews=[CandidateReview(candidate_id=c['id'], acceptable=acceptable, contact_score=0.95 if c['id'] == winner else 0.55, placement_score=0.95 if c['id'] == winner else 0.55, proportion_score=0.95 if c['id'] == winner else 0.55, appearance_score=0.95 if c['id'] == winner else 0.55, explanation='Fixture review for selection tests.') for c in candidates], summary='Fixture evaluation; not evidence of real-world quality.')

def image_data():
    buffer = io.BytesIO()
    Image.new('RGB', (512, 512), 'white').save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()

class GeometrySearchTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.a, self.b = (self.root / 'a.glb', self.root / 'b.glb')
        self.a.write_bytes(trimesh.creation.box().export(file_type='glb'))
        self.b.write_bytes(trimesh.creation.box().export(file_type='glb'))
        self.plan = make_plan()
        self.original, self.diagnostics = solve_placement(self.a, self.b, self.plan, [], Transform())
        self.context = GeometryContext(self.a, self.b, Transform(), self.plan, self.diagnostics)

    def tearDown(self):
        self.temp.cleanup()

    def test_closed_mesh_point_containment(self):
        inside = contains_points(trimesh.creation.box(), np.array([[0, 0, 0], [0.49, 0.1, -0.1], [0.6, 0, 0], [0, 1, 0]]))
        np.testing.assert_array_equal(inside, [True, True, False, False])

    def test_valid_contact_and_floating_product_differ(self):
        contact = self.context.check(self.original)
        floating = self.context.check(self.original.model_copy(update={'position': [0, 1.2, 0]}))
        self.assertTrue(contact['eligible'], contact)
        self.assertFalse(floating['eligible'])
        self.assertLess(contact['metrics']['anchor_error_relative'], 0.01)
        self.assertGreater(contact['geometry_score'], floating['geometry_score'])

    def test_solid_intersection_blocks_rigid_composition(self):
        wrong = self.original.model_copy(update={'position': [0, 0, 0]})
        metrics = self.context.check(wrong)
        self.assertFalse(metrics['eligible'])
        self.assertGreater(metrics['metrics']['penetration_fraction'], 0.8)
        self.assertTrue(any(('solid overlap' in m for m in metrics['failures'])))

    def test_around_rejects_material_penetration_even_with_matching_centers(self):
        plan = make_plan('around').model_copy(update={'base_anchor': [0.5, 0.5, 0.5], 'product_anchor': [0.5, 0.5, 0.5]})
        transform, diagnostics = solve_placement(self.a, self.b, plan, [], Transform())
        context = GeometryContext(self.a, self.b, Transform(), plan, diagnostics)
        checked = context.check(transform)
        self.assertFalse(checked['eligible'], checked)
        self.assertGreater(checked['metrics']['penetration_fraction'], 0.8)
        self.assertTrue(checked['warnings'])

    def test_inside_has_explicit_cavity_limit(self):
        plan = make_plan('inside').model_copy(update={'base_anchor': [0.5, 0.5, 0.5], 'product_anchor': [0.5, 0.5, 0.5]})
        transform, diagnostics = solve_placement(self.a, self.b, plan, [], Transform())
        context = GeometryContext(self.a, self.b, Transform(), plan, diagnostics)
        self.assertTrue(any(('cavity' in warning.lower() for warning in context.check(transform)['warnings'])))
        outside = transform.model_copy(update={'position': [2, 0, 0]})
        self.assertFalse(context.check(outside)['eligible'])

    def test_inside_rejects_solid_material_even_when_bbox_contains_product(self):
        plan = make_plan('inside').model_copy(update={'base_anchor': [0.5, 0.5, 0.5], 'product_anchor': [0.5, 0.5, 0.5]})
        transform, diagnostics = solve_placement(self.a, self.b, plan, [], Transform())
        checked = GeometryContext(self.a, self.b, Transform(), plan, diagnostics).check(transform)
        self.assertFalse(checked['eligible'])
        self.assertTrue(any(('solid material' in value for value in checked['failures'])))

    def test_attached_aligns_contact_normals_without_assuming_global_up(self):
        plan = make_plan('attached').model_copy(update={'base_anchor': [1, 0.5, 0.5], 'product_anchor': [0.5, 0, 0.5]})
        transform, diagnostics = solve_placement(self.a, self.b, plan, [], Transform(rotation=[0.3, 0.2, -0.4]))
        context = GeometryContext(self.a, self.b, Transform(rotation=[0.3, 0.2, -0.4]), plan, diagnostics)
        self.assertTrue(context.check(transform)['eligible'])
        flipped = transform.model_copy(update={'rotation': [0, 0, 0]})
        self.assertFalse(context.check(flipped)['eligible'])

    def test_hanging_rejects_product_above_support(self):
        plan = make_plan('hanging').model_copy(update={'base_anchor': [1, 0.8, 0.5], 'product_anchor': [0, 1, 0.5]})
        transform, diagnostics = solve_placement(self.a, self.b, plan, [], Transform())
        context = GeometryContext(self.a, self.b, Transform(), plan, diagnostics)
        self.assertTrue(context.check(transform)['eligible'])
        above = transform.model_copy(update={'position': [0.65, 0.9, 0]})
        self.assertTrue(any(('hanging point' in m for m in context.check(above)['failures'])))

    def test_bounded_search_keeps_baseline_and_improves_only_its_objective(self):
        candidates, _ = build_candidates(self.a, self.b, self.plan, [], Transform(), 5)
        self.assertEqual(len(candidates), 5)
        self.assertTrue(candidates[0]['baseline'])
        self.assertFalse(candidates[0]['optimization']['performed'])
        for c in candidates[1:]:
            self.assertTrue(c['optimization']['performed'])
            self.assertLessEqual(c['optimization']['objective_after'], c['optimization']['objective_before'] + 1e-10)
            self.assertLessEqual(c['optimization']['evaluations'], 60)
        json.dumps(candidates, allow_nan=False)

    def test_open_mesh_reports_unavailable_volume_test(self):
        mesh = trimesh.creation.box()
        mesh.update_faces(np.arange(10))
        self.a.write_bytes(mesh.export(file_type='glb'))
        context = GeometryContext(self.a, self.b, Transform(), self.plan, self.diagnostics)
        result = context.check(self.original)
        self.assertFalse(result['metrics']['base_watertight'])
        self.assertTrue(any(('open mesh' in warning.lower() for warning in result['warnings'])))

class CompositionAPITests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.gemini = Mock(spec=Gemini)
        self.gemini.key = 'fake-key'
        self.gemini.model = 'test-model'
        self.gemini.plan.return_value = make_plan()
        self.app = create_app(self.root, 'test-access', self.gemini)
        self.client = TestClient(self.app)
        self.client.headers['Authorization'] = 'Bearer test-access'
        raw = self.root / 'cube.glb'
        raw.write_bytes(trimesh.creation.box().export(file_type='glb'))
        self.a = self.app.state.engine.asset(raw, 'A')
        self.b = self.app.state.engine.asset(raw, 'B')
        identity = np.eye(4).flatten(order='F').tolist()
        self.request = {'base_id': self.a['id'], 'product_id': self.b['id'], 'candidate_count': 3, 'captures': [{'slot': slot, 'image': image_data(), 'projection': identity, 'world': identity} for slot in ('A', 'B')]}

    def tearDown(self):
        self.app.state.engine.pool.shutdown(wait=True)
        self.temp.cleanup()

    def prepare(self):
        response = self.client.post('/api/compositions', json=self.request)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['status'], 'prepared')
        self.assertEqual(result['api_calls'], {'plan': 1, 'evaluation': 0})
        return result

    def renders(self, run):
        return {'renders': [{'candidate_id': c['id'], 'image': image_data()} for c in run['candidates']]}

    def test_two_step_pipeline_persists_renders_and_selects_candidate(self):
        run = self.prepare()
        self.gemini.evaluate.return_value = review(run['candidates'])
        response = self.client.post(f"/api/compositions/{run['id']}/evaluate", json=self.renders(run))
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['evaluation_status'], 'completed')
        self.assertEqual(result['selected_candidate_id'], 'c1')
        self.assertTrue(result['auto_apply'])
        self.assertEqual(result['api_calls'], {'plan': 1, 'evaluation': 1})
        report = self.client.get(f"/api/compositions/{run['id']}/report")
        self.assertEqual(len(report.json()['renders']), 3)
        self.assertNotIn('fake-key', report.text)
        self.assertEqual(self.client.get(f"/api/compositions/{run['id']}/renders/c1.png").status_code, 200)

    def test_completed_run_reuses_plan_and_visual_result(self):
        run = self.prepare()
        self.gemini.evaluate.return_value = review(run['candidates'])
        self.client.post(f"/api/compositions/{run['id']}/evaluate", json=self.renders(run))
        again = self.client.post('/api/compositions', json=self.request).json()
        self.assertEqual(again['id'], run['id'])
        self.assertEqual(again['evaluation_status'], 'completed')
        self.client.post(f"/api/compositions/{run['id']}/evaluate", json={})
        self.assertEqual(self.gemini.plan.call_count, 1)
        self.assertEqual(self.gemini.evaluate.call_count, 1)

    def test_changed_base_pose_replans_with_the_new_spatial_context(self):
        first = self.prepare()
        self.request['base_transform'] = {'position': [1, 2, 3], 'rotation': [0, 0.5, 0], 'scale': [1, 1, 1]}
        second = self.client.post('/api/compositions', json=self.request).json()
        self.assertNotEqual(second['id'], first['id'])
        self.assertFalse(second['plan_cached'])
        self.assertEqual(self.gemini.plan.call_count, 2)

    def test_missing_or_duplicate_renders_rejected_before_provider(self):
        run = self.prepare()
        payload = self.renders(run)
        payload['renders'][1]['candidate_id'] = payload['renders'][0]['candidate_id']
        response = self.client.post(f"/api/compositions/{run['id']}/evaluate", json=payload)
        self.assertEqual(response.status_code, 400)
        self.gemini.evaluate.assert_not_called()

    def test_corrupt_png_rejected_before_provider(self):
        run = self.prepare()
        payload = self.renders(run)
        payload['renders'][0]['image'] = 'not-base64!'
        self.assertEqual(self.client.post(f"/api/compositions/{run['id']}/evaluate", json=payload).status_code, 400)
        self.gemini.evaluate.assert_not_called()

    def test_visual_failure_is_manual_fallback_and_can_retry_without_new_plan(self):
        run = self.prepare()
        self.gemini.evaluate.side_effect = ProviderError('Quota exhausted')
        result = self.client.post(f"/api/compositions/{run['id']}/evaluate", json=self.renders(run)).json()
        self.assertEqual(result['selection_method'], 'geometry_only')
        self.assertFalse(result['auto_apply'])
        self.assertTrue(result['needs_review'])
        self.assertIsNone(result['selected_candidate_id'])
        self.assertTrue(any(('Quota' in warning for warning in result['warnings'])))
        self.gemini.evaluate.side_effect = None
        self.gemini.evaluate.return_value = review(run['candidates'])
        result = self.client.post(f"/api/compositions/{run['id']}/evaluate", json=self.renders(run)).json()
        self.assertEqual(result['evaluation_status'], 'completed')
        self.assertEqual(result['api_calls'], {'plan': 1, 'evaluation': 2})
        self.assertEqual(self.gemini.plan.call_count, 1)

    def test_vlm_can_reject_all_and_nothing_is_auto_applied(self):
        run = self.prepare()
        self.gemini.evaluate.return_value = review(run['candidates'], acceptable=False)
        result = self.client.post(f"/api/compositions/{run['id']}/evaluate", json=self.renders(run)).json()
        self.assertIsNone(result['selected_candidate_id'])
        self.assertFalse(result['auto_apply'])
        self.assertTrue(result['suggested_candidate_id'])

    def test_candidate_count_is_bounded(self):
        self.request['candidate_count'] = 100
        self.assertEqual(self.client.post('/api/compositions', json=self.request).status_code, 422)
        self.gemini.plan.assert_not_called()

class RankingTests(unittest.TestCase):

    def candidates(self):
        return [{'id': f'c{i}', 'eligible': i != 2, 'geometry_score': 0.9 if i == 0 else 0.8, 'baseline': i == 0, 'warnings': [], 'failures': [] if i != 2 else ['collision']} for i in range(3)]

    def test_visual_score_cannot_override_geometry_rejection(self):
        candidates = self.candidates()
        result = rank_candidates(candidates, review(candidates, winner='c2'))
        self.assertNotEqual(result['selected_candidate_id'], 'c2')
        self.assertEqual(result['candidates'][-1]['id'], 'c2')

    def test_baseline_can_remain_best(self):
        candidates = self.candidates()
        result = rank_candidates(candidates, review(candidates, winner='c0'))
        self.assertEqual(result['selected_candidate_id'], 'c0')
        self.assertEqual(result['comparison']['combined_delta'], 0)

    def test_duplicate_or_wrong_candidate_ids_rejected(self):
        candidates = self.candidates()
        evaluation = review(candidates)
        evaluation.reviews[1].candidate_id = 'c0'
        with self.assertRaises(ValueError):
            rank_candidates(candidates, evaluation)

    def test_no_geometry_candidate_passes_requires_manual_input(self):
        candidates = self.candidates()
        for c in candidates:
            c['eligible'] = False
        result = rank_candidates(candidates, review(candidates))
        self.assertIsNone(result['selected_candidate_id'])
        self.assertFalse(result['auto_apply'])

    def test_warning_or_low_plan_confidence_prevents_auto_apply(self):
        candidates = self.candidates()
        candidates[0]['warnings'] = ['Open mesh']
        result = rank_candidates(candidates, review(candidates, winner='c0'))
        self.assertFalse(result['auto_apply'])
        candidates[0]['warnings'] = []
        self.assertFalse(rank_candidates(candidates, review(candidates), plan_confidence=0.3)['auto_apply'])

class EvaluationClientTests(unittest.TestCase):

    def provider(self):
        candidates = [{'id': f'c{i}', 'eligible': True, 'metrics': {}, 'failures': [], 'warnings': []} for i in range(3)]
        transport = Mock()
        response = transport.post.return_value
        response.status_code = 200
        response.json.return_value = {'candidates': [{'content': {'parts': [{'text': review(candidates).model_dump_json()}]}}]}
        client = Gemini('fake-key', 'test-model', transport)
        return (client, transport, candidates)

    def test_evaluation_uses_one_batch_and_distinct_schema(self):
        from unicompose3d.schemas import CandidateRender
        client, transport, candidates = self.provider()
        renders = [CandidateRender(candidate_id=c['id'], image=image_data()) for c in candidates]
        result = client.evaluate(make_plan().model_dump(), candidates, renders, '')
        self.assertEqual(len(result.reviews), 3)
        self.assertEqual(transport.post.call_count, 1)
        body = transport.post.call_args.kwargs['json']
        self.assertIn('reviews', body['generationConfig']['responseJsonSchema']['properties'])
        self.assertEqual(sum(('inline_data' in p for p in body['contents'][0]['parts'])), 3)

    def test_missing_candidate_from_provider_is_rejected(self):
        client, transport, candidates = self.provider()
        response = transport.post.return_value
        wrong = review(candidates)
        wrong.reviews[1].candidate_id = 'c0'
        response.json.return_value = {'candidates': [{'content': {'parts': [{'text': wrong.model_dump_json()}]}}]}
        with self.assertRaisesRegex(ProviderError, 'missing, duplicate'):
            client.evaluate(make_plan().model_dump(), candidates, [], '')

    def test_judge_quota_does_not_retry_or_switch_provider(self):
        client, transport, candidates = self.provider()
        transport.post.return_value.status_code = 429
        with self.assertRaises(ProviderError):
            client.evaluate(make_plan().model_dump(), candidates, [], '')
        self.assertEqual(transport.post.call_count, 1)

    def test_out_of_range_judge_score_is_not_accepted(self):
        client, transport, candidates = self.provider()
        wrong = review(candidates).model_dump()
        wrong['reviews'][0]['appearance_score'] = 10
        transport.post.return_value.json.return_value = {'candidates': [{'content': {'parts': [{'text': json.dumps(wrong)}]}}]}
        with self.assertRaises(ProviderError):
            client.evaluate(make_plan().model_dump(), candidates, [], '')
if __name__ == '__main__':
    unittest.main(verbosity=2)
