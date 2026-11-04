"""Completed-plan repair is bounded, observable and category-independent."""
import base64
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import trimesh
from PIL import Image
from unicompose3d.api import Engine
from unicompose3d.clients import ProviderError
from unicompose3d.openai_client import OpenAIClient, strict_schema
from unicompose3d.schemas import PlacementPlan, ComposeRequest, Capture
from unicompose3d.storage import Store

def plan_data():
    plan = PlacementPlan(relation='arbitrary surface relation', base_anchor=[0.5, 1, 0.5], product_anchor=[0.5, 0, 0.5], rotation_degrees=[0, 0, 0], relative_scale=0.3, confidence=0.85, explanation='Two surfaces touch.', program={'features': [{'id': 'a_surface', 'slot': 'A', 'kind': 'surface', 'point': [0.5, 1, 0.5]}, {'id': 'b_surface', 'slot': 'B', 'kind': 'surface', 'point': [0.5, 0, 0.5]}], 'constraints': [{'kind': 'contact', 'a': 'a_surface', 'b': 'b_surface', 'target': 0.003}, {'kind': 'proportion', 'a': 'a_surface', 'b': 'b_surface', 'target': 0.3}]})
    from unicompose3d.schemas import SemanticIntent
    plan.intent = SemanticIntent(target_region='target surface', moving_region='product support', spatial_relationship='contact', orientation='normals opposed')
    data = plan.model_dump()
    for name in ('fit_mode', 'base_axis_end', 'base_axis_end_view', 'base_axis_end_uv'):
        data.pop(name, None)
    return data

def response(value, status='completed', code=200):
    text = value if isinstance(value, str) else json.dumps(value)
    return SimpleNamespace(status_code=code, json=lambda: {'status': status, 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}]})

class PlanningTests(unittest.TestCase):

    def test_wire_schema_exposes_coordinate_bounds_requires_program_and_clearance_ratio(self):
        schema = strict_schema(PlacementPlan)
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        validator.validate(plan_data())
        invalid = plan_data()
        invalid['base_anchor'][0] = -0.5
        self.assertTrue(list(validator.iter_errors(invalid)))
        self.assertEqual(schema['properties']['base_anchor']['items']['minimum'], 0)
        self.assertEqual(schema['properties']['base_anchor']['items']['maximum'], 1)
        self.assertEqual(schema['properties']['program'], {'anyOf': [{'$ref': '#/$defs/SpatialProgram'}, {'type': 'null'}]})
        self.assertNotIn('fit_mode', schema['properties'])
        branches = schema['$defs']['SpatialConstraint']['anyOf']
        clearance = next((x for x in branches if x['properties']['kind']['enum'] == ['clearance']))
        self.assertEqual(clearance['properties']['target']['minimum'], 1.015)

    def test_invalid_coordinates_and_graph_are_repaired_once_using_validation_feedback(self):
        for kind in ('coordinate', 'reference', 'null_program', 'json'):
            valid = plan_data()
            invalid = json.loads(json.dumps(valid))
            if kind == 'coordinate':
                invalid['base_anchor'][0] = -0.5
            elif kind == 'reference':
                invalid['program']['constraints'][0]['b'] = 'not_a_feature'
            elif kind == 'null_program':
                invalid['program'] = None
            else:
                invalid = '{"relation":'
            http = Mock()
            http.post.side_effect = [response(invalid), response(valid)]
            result = OpenAIClient(key='fixture-secret', transport=http)._generate([{'text': 'arbitrary objects'}], PlacementPlan)
            self.assertEqual(result.program.constraints[0].b, 'b_surface')
            self.assertEqual(result._provider_calls, 2)
            self.assertNotIn('_provider_calls', result.model_dump())
            self.assertEqual(http.post.call_count, 2)
            first = http.post.call_args_list[0].kwargs['json']
            second = http.post.call_args_list[1].kwargs['json']
            self.assertEqual(len(first['input']), 1)
            self.assertEqual(len(second['input']), 3)
            feedback = second['input'][-1]['content'][0]['text']
            self.assertIn('Validation failed', feedback)
            self.assertNotIn('fixture-secret', feedback)

    def test_second_invalid_plan_is_rejected_with_actionable_error_and_no_more_calls(self):
        invalid = plan_data()
        invalid['program']['constraints'][0]['b'] = 'missing'
        http = Mock()
        http.post.return_value = response(invalid)
        with self.assertRaises(ProviderError) as error:
            OpenAIClient(key='fixture', transport=http)._generate([], PlacementPlan)
        self.assertEqual(http.post.call_count, 2)
        self.assertIn('program', str(error.exception))
        self.assertIn('2', str(error.exception))

    def test_http_incomplete_refusal_and_empty_response_are_not_repaired(self):
        payloads = [(429, {'error': {'message': 'rate limited'}}), (200, {'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'}}), (200, {'status': 'failed', 'error': {'message': 'backend failure'}}), (200, {'status': 'completed', 'output': []}), (200, {'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'refusal'}]}]})]
        for code, payload in payloads:
            http = Mock()
            http.post.return_value = SimpleNamespace(status_code=code, json=lambda p=payload: p)
            with self.assertRaises(ProviderError):
                OpenAIClient(key='fixture', transport=http)._generate([], PlacementPlan)
            self.assertEqual(http.post.call_count, 1)

    def test_repaired_plan_executes_general_solver_and_reports_actual_calls_then_uses_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mesh = root / 'box.glb'
            trimesh.creation.box().export(mesh)
            valid = plan_data()
            invalid = json.loads(json.dumps(valid))
            invalid['base_anchor'][0] = -1
            http = Mock()
            http.post.side_effect = [response(invalid), response(valid)]
            engine = Engine(Store(root / 'data'), gemini=OpenAIClient(key='fixture', transport=http))
            self.addCleanup(engine.pool.shutdown, wait=False, cancel_futures=True)
            self.addCleanup(engine.segments.close)
            a = engine.asset(mesh, 'unknown rigid base')
            b = engine.asset(mesh, 'unknown rigid product')
            image = io.BytesIO()
            Image.new('RGB', (8, 8), 'white').save(image, format='PNG')
            uri = 'data:image/png;base64,' + base64.b64encode(image.getvalue()).decode()
            identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
            request = ComposeRequest(base_id=a['id'], product_id=b['id'], instruction='place these surfaces together', captures=[Capture(slot=slot, image=uri, projection=identity, world=identity) for slot in ('A', 'B')])
            result = engine.prepare_composition(request)
            self.assertEqual(result['api_calls'], {'plan': 2, 'evaluation': 0})
            self.assertTrue(any((c['eligible'] for c in result['candidates'])))
            self.assertTrue(engine.prepare_composition(request)['reused'])
            self.assertEqual(http.post.call_count, 2)
if __name__ == '__main__':
    unittest.main()
