"""Offline functional and contract tests. Never uses real API credits."""
import base64
import io
import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch
import cv2
import numpy as np
import requests
import trimesh
from fastapi.testclient import TestClient
from PIL import Image
from unicompose3d.api import create_app
from unicompose3d.clients import Gemini, ProviderError
from unicompose3d.geometry import align_anchors, camera_anchor, deform_mesh, export_scene, flat_mesh, load_scene, normalize, ray_hit, solve_placement, transform_matrix
from unicompose3d.media import normalize_image, video_frames
from unicompose3d.schemas import Capture, PlacementPlan, Transform

def plan(**changes):
    return PlacementPlan(relation='on', base_anchor=[0.5, 1, 0.5], product_anchor=[0.5, 0, 0.5], rotation_degrees=[0, 0, 0], relative_scale=0.3, confidence=0.8, explanation='Place the vase on the table.', **changes)

class SystemTests(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.gemini = Mock(spec=Gemini)
        self.gemini.key = 'fake-never-sent'
        self.gemini.model = 'test-model'
        self.gemini.plan.return_value = plan()
        self.app = create_app(self.root, 'access-test', self.gemini)
        self.client = TestClient(self.app)
        self.client.headers['Authorization'] = 'Bearer access-test'
        self.assets = self.client.post('/api/demo').json()
        self.a, self.b = (self.assets['A'], self.assets['B'])
        self.engine = self.app.state.engine

    def tearDown(self):
        self.engine.pool.shutdown(wait=True)
        self.temporary.cleanup()

    def payload(self):
        return {'base_id': self.a['id'], 'product_id': self.b['id'], 'base_transform': {'position': [0.2, 0.4, 0.1], 'rotation': [0.1, 0.2, 0.3], 'scale': [1, 1, 1]}, 'product_transform': {'position': [0.1, 0.8, 0], 'rotation': [0.2, -0.1, 0.5], 'scale': [0.3, 0.4, 0.2]}, 'title': 'Test scene'}

    def captures(self):
        output = io.BytesIO()
        Image.new('RGB', (32, 32), 'white').save(output, format='PNG')
        identity = np.eye(4).flatten(order='F').tolist()
        return [{'slot': slot, 'image': 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode(), 'projection': identity, 'world': identity} for slot in ('A', 'B')]

    def test_api_requires_access_token(self):
        response = self.client.get('/api/assets', headers={'Authorization': 'Bearer wrong'})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.client.get('/health', headers={'Authorization': ''}).status_code, 200)

    def test_config_does_not_expose_provider_keys(self):
        self.assertNotIn('fake-never-sent', self.client.get('/api/config').text)

    def test_demo_is_real_geometry_and_never_calls_providers(self):
        again = self.client.post('/api/demo').json()
        self.assertEqual(again['A']['id'], self.a['id'])
        self.assertGreater(self.b['faces'], 500)
        self.gemini.plan.assert_not_called()

    def test_export_preserves_transforms_and_independent_meshes(self):
        value = self.payload()
        response = self.client.post('/api/export', json=value)
        self.assertEqual(response.status_code, 200)
        result = trimesh.load(io.BytesIO(response.content), file_type='glb', force='scene')
        self.assertGreaterEqual(len(result.geometry), 2)
        base = load_scene(self.engine.model_path(self.a['id']))
        product = load_scene(self.engine.model_path(self.b['id']))
        base.apply_transform(transform_matrix(Transform(**value['base_transform'])))
        product.apply_transform(transform_matrix(Transform(**value['product_transform'])))
        expected = np.array([np.minimum(base.bounds[0], product.bounds[0]), np.maximum(base.bounds[1], product.bounds[1])])
        np.testing.assert_allclose(result.bounds, expected, atol=1e-05)

    def test_project_round_trip_reopens_original_poses(self):
        payload = self.payload()
        response = self.client.post('/api/project/export', json=payload)
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(set(archive.namelist()), {'project.json', 'A.glb', 'B.glb'})
        restored = self.client.post('/api/project/import', files={'file': ('project.zip', response.content, 'application/zip')})
        self.assertEqual(restored.status_code, 200, restored.text)
        value = restored.json()
        self.assertNotEqual(value['base_id'], payload['base_id'])
        self.assertEqual(value['product_transform'], payload['product_transform'])
        output = self.client.post('/api/export', json=value)
        self.assertEqual(output.status_code, 200)

    def test_zip_path_traversal_rejected(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('../../outside.py', "print('bad')")
        response = self.client.post('/api/project/import', files={'file': ('bad.zip', stream.getvalue())})
        self.assertEqual(response.status_code, 400)

    def test_bad_ids_cannot_read_outside_assets(self):
        self.assertEqual(self.client.get('/api/assets/not-an-id/model.glb').status_code, 400)

    def test_import_glb_requires_no_api(self):
        data = self.client.get(f"/api/assets/{self.b['id']}/model.glb").content
        response = self.client.post('/api/sources', data={'kind': 'model', 'name': 'Imported'}, files=[('files', ('model.glb', data, 'model/gltf-binary'))])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['asset']['status'], 'ready')

    def test_external_gltf_reference_rejected(self):
        data = json.dumps({'asset': {'version': '2.0'}, 'buffers': [{'uri': '../../secret.bin', 'byteLength': 12}]}).encode()
        response = self.client.post('/api/sources', data={'kind': 'model'}, files=[('files', ('model.gltf', data))])
        self.assertEqual(response.status_code, 400)

    def upload_image(self, kind='image', count=1):
        image = io.BytesIO()
        Image.new('RGB', (64, 64), 'teal').save(image, format='PNG')
        response = self.client.post('/api/sources', data={'kind': kind, 'name': 'Photo'}, files=[('files', (f'image{i}.png', image.getvalue(), 'image/png')) for i in range(count)])
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_multiview_rejects_reused_image(self):
        source = self.upload_image('multiview', 2)
        self.engine.quality = Mock()
        self.engine.quality.ready = True
        image = source['images'][0]['file']
        response = self.client.post('/api/generate', json={'source_id': source['id'], 'provider': 'hunyuan', 'views': {'front': image, 'left': image}})
        self.assertEqual(response.status_code, 400)

    def test_ai_cached_plan_does_not_call_api_twice(self):
        payload = {'base_id': self.a['id'], 'product_id': self.b['id'], 'captures': self.captures()}
        first = self.client.post('/api/compose', json=payload)
        self.assertEqual(first.status_code, 200, first.text)
        second = self.client.post('/api/compose', json=payload)
        self.assertTrue(second.json()['cached'])
        self.assertEqual(self.gemini.plan.call_count, 1)

    def test_no_ai_images_is_an_explicit_error(self):
        response = self.client.post('/api/compose', json={'base_id': self.a['id'], 'product_id': self.b['id']})
        self.assertEqual(response.status_code, 400)
        self.gemini.plan.assert_not_called()

    def test_manual_anchor_alignment_respects_rotated_and_scaled_base(self):
        a = Transform(position=[1, 2, 3], rotation=[0.1, 0.2, 0.5], scale=[1, 2, 0.5])
        b = Transform(position=[0, 0, 0], rotation=[0.5, -0.2, 0.6], scale=[0.3, 0.2, 0.4])
        pa, pb = ([0.1, 0.2, 0.3], [-0.2, 0.1, 0.2])
        result = align_anchors(pa, pb, a, b)
        np.testing.assert_allclose(transform_matrix(a) @ [*pa, 1], transform_matrix(result) @ [*pb, 1], atol=1e-10)

    def test_soft_deformation_preserves_faces_and_outside_vertices(self):
        source = self.engine.model_path(self.b['id'])
        target = self.root / 'deformed.glb'
        before = flat_mesh(source)
        info = deform_mesh(source, target, [0, 0.5, 0], [0.1, 0, 0], 0.3)
        after = flat_mesh(target)
        self.assertEqual(info['faces'], self.b['faces'])
        outside = np.linalg.norm(before.vertices - [0, 0.5, 0], axis=1) > 0.3
        np.testing.assert_allclose(after.vertices[outside], before.vertices[outside], atol=1e-06)
        self.assertGreater(np.max(np.abs(before.vertices - after.vertices)), 0.001)

    def test_invalid_transform_rejected(self):
        payload = self.payload()
        payload['product_transform']['scale'] = [-1, 1, 1]
        self.assertEqual(self.client.post('/api/export', json=payload).status_code, 422)

    def test_ray_hits_nearest_visible_surface(self):
        mesh = trimesh.creation.box()
        hit = ray_hit(mesh, np.array([0, 0, 2]), np.array([0, 0, -1]))
        np.testing.assert_allclose(hit[0], [0, 0, 0.5])

    def test_ai_rotation_follows_base_rotation(self):
        a = Transform(rotation=[0.2, 0.7, 0.1])
        result, _ = solve_placement(self.engine.model_path(self.a['id']), self.engine.model_path(self.b['id']), plan(), [], a)
        from scipy.spatial.transform import Rotation
        np.testing.assert_allclose(Rotation.from_euler('XYZ', result.rotation).as_matrix(), Rotation.from_euler('XYZ', a.rotation).as_matrix(), atol=1e-08)

    def test_render_camera_unprojection_matches_three_column_major(self):
        near, far, f = (0.1, 10, 1 / np.tan(np.deg2rad(35) / 2))
        projection = np.array([[f, 0, 0, 0], [0, f, 0, 0], [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]])
        world = np.eye(4)
        world[2, 3] = 2
        capture = Capture(slot='A', image='', projection=projection.flatten(order='F').tolist(), world=world.flatten(order='F').tolist())
        hit = camera_anchor(trimesh.creation.box(), capture, [0.5, 0.5])
        np.testing.assert_allclose(hit[0], [0, 0, 0.5], atol=1e-08)

    def test_video_extraction_produces_inspectable_frames(self):
        video = self.root / 'input.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 64))
        for i in range(30):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.circle(frame, (10 + i, 32), 8, (0, 255, 100), -1)
            writer.write(frame)
        writer.release()
        output = self.root / 'frames'
        output.mkdir()
        result = video_frames(video, output, count=6)
        self.assertEqual(len(result), 6)
        self.assertTrue(all(((output / r['file']).exists() for r in result)))

    def test_texture_survives_normalization_and_scene_export(self):
        mesh = trimesh.creation.box()
        image = Image.new('RGB', (16, 16), 'orange')
        mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)), image=image)
        raw = self.root / 'textured.glb'
        raw.write_bytes(mesh.export(file_type='glb'))
        output = self.root / 'normalized.glb'
        normalize(raw, output)
        result = export_scene(output, output, Transform(), Transform(position=[2, 0, 0]))
        scene = trimesh.load(io.BytesIO(result), file_type='glb', force='scene')
        self.assertTrue(all((g.visual.kind == 'texture' for g in scene.geometry.values())))

class ProviderContractTests(unittest.TestCase):

    def test_gemini_rate_limit_is_explicit(self):
        transport = Mock()
        transport.post.return_value.status_code = 429
        client = Gemini('not-real', 'test', transport)
        with self.assertRaisesRegex(ProviderError, 'quota'):
            client.plan({}, {}, '', [])
        self.assertEqual(transport.post.call_count, 1)

    def test_gemini_schema_and_response_validated(self):
        from unicompose3d.constraint_solver import compatibility_program
        transport = Mock()
        response = transport.post.return_value
        response.status_code = 200
        proposed = plan()
        proposed.program = compatibility_program(proposed)
        response.json.return_value = {'candidates': [{'content': {'parts': [{'text': proposed.model_dump_json()}]}}]}
        client = Gemini('not-real', 'test', transport)
        result = client.plan({}, {}, '', [])
        self.assertEqual(result.relation, 'on')
        self.assertIn('responseJsonSchema', transport.post.call_args.kwargs['json']['generationConfig'])
if __name__ == '__main__':
    unittest.main(verbosity=2)
