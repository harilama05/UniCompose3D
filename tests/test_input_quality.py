"""Real image/mask/GLB checks; GPU models and Gemini are offline fixtures."""
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from unicompose3d.api import create_app
from unicompose3d.clients import Gemini, ProviderError
from unicompose3d.input_processing import make_cutout
from unicompose3d.media import normalize_image
from unicompose3d.schemas import TargetLocation, SegmentRequest

class InputQualityTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        gemini = Mock(spec=Gemini)
        gemini.key = 'test'
        gemini.model = 'offline'
        gemini.locate.return_value = TargetLocation(found=True, confidence=0.9, box=[0.1, 0.1, 0.9, 0.9], explanation='fixture')
        self.app = create_app(self.root / 'data', 'test', gemini)
        self.engine = self.app.state.engine
        self.engine.quality = Mock()
        self.engine.quality.ready = True
        self.engine.quality.segment_ready = True
        self.client = TestClient(self.app)
        self.client.headers['Authorization'] = 'Bearer test'
        self.gemini = gemini

        def mask_fixture(image, box, points, labels, directory, stopping):
            with Image.open(image) as photo:
                mask = Image.new('L', photo.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.ellipse((10, 10, 110, 110), fill=255)
                draw.ellipse((40, 40, 80, 80), fill=0)
                mask.save(directory / 'mask.png')
            return directory / 'mask.png'
        self.engine.quality.segment.side_effect = mask_fixture
        demo = self.client.post('/api/demo').json()
        self.engine.quality.generate.return_value = self.engine.model_path(demo['B']['id'])

    def test_retired_backends_are_rejected_without_running_reconstruction(self):
        source = self.source()
        for provider in ("tripo", "triposr", "sf3d", "trellis", "unknown"):
            with self.subTest(provider=provider):
                response = self.client.post("/api/generate", json={
                    "source_id": source["id"], "provider": provider})
                self.assertEqual(response.status_code, 422)
        self.engine.quality.generate.assert_not_called()
        self.assertEqual(self.engine.store.list("jobs"), [])

    def tearDown(self):
        self.engine.pool.shutdown(wait=True)
        self.tmp.cleanup()

    def source(self, count=1):
        files = []
        for index in range(count):
            output = io.BytesIO()
            Image.new('RGB', (128, 128), (120, 30 + index * 50, 70)).save(output, 'PNG')
            files.append(('files', (str(index) + '.png', output.getvalue(), 'image/png')))
        response = self.client.post('/api/sources', data={'kind': 'image' if count == 1 else 'multiview', 'name': 'an arbitrary object'}, files=files)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def segment(self, source, index=0, **kwargs):
        return self.client.post(f"/api/sources/{source['id']}/segment", json={'filename': source['images'][index]['file'], 'target': 'an arbitrary object', **kwargs})

    def payload(self, source, cutouts):
        views = dict(zip(['front', 'left'], [item['file'] for item in source['images']]))
        return {'provider': 'hunyuan', 'source_id': source['id'], 'views': views, 'target': 'an arbitrary object', 'use_cutouts': True, 'confirm_cutouts': True, 'cutout_files': {image['file']: cutout['file'] for image, cutout in zip(source['images'], cutouts)}}

    def wait_job(self, ident):
        for _ in range(300):
            job = self.client.get('/api/jobs/' + ident).json()
            if job['status'] in {'success', 'error'}:
                return job
            time.sleep(0.01)
        self.fail('job timeout')

    def test_manual_box_skips_gemini_and_keeps_holes_transparent(self):
        source = self.source()
        response = self.segment(source, box=[0.05, 0.05, 0.95, 0.95], points=[[0.5, 0.5]], labels=[0])
        self.assertEqual(response.status_code, 200, response.text)
        self.gemini.locate.assert_not_called()
        result = self.client.get(f"/api/sources/{source['id']}/{response.json()['file']}")
        with Image.open(io.BytesIO(result.content)) as image:
            self.assertEqual(image.mode, 'RGBA')
            self.assertEqual(image.getpixel((60, 60))[3], 0)
            self.assertEqual(image.getpixel((60, 20))[3], 255)
            self.assertEqual(image.getpixel((0, 0))[3], 0)
        self.assertIn('quality', source['images'][0])

    def test_auto_localizes_by_name_and_reuses_location_and_mask(self):
        source = self.source()
        first, second = (self.segment(source), self.segment(source))
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()['file'], second.json()['file'])
        self.assertEqual(self.gemini.locate.call_count, 1)
        self.assertEqual(self.engine.quality.segment.call_count, 1)
        self.assertEqual(self.gemini.locate.call_args.args[1], 'an arbitrary object')

    def test_unknown_target_needs_manual_selection_never_reconstructs(self):
        source = self.source()
        self.gemini.locate.return_value = TargetLocation(found=False, confidence=0.2, box=[0, 0, 1, 1], explanation='Multiple similar objects')
        response = self.segment(source)
        self.assertEqual(response.status_code, 400)
        self.engine.quality.segment.assert_not_called()
        self.engine.quality.generate.assert_not_called()

    def test_multiview_default_routes_all_cutouts_to_free_quality_backend(self):
        source = self.source(2)
        cutouts = [self.segment(source, i).json() for i in range(2)]
        payload = self.payload(source, cutouts)
        response = self.client.post('/api/generate', json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        job = self.wait_job(response.json()['id'])
        self.assertEqual(job['status'], 'success', job)
        self.assertEqual(job['provider'], 'hunyuan')
        self.assertEqual(job['credits_consumed'], 0)
        views = self.engine.quality.generate.call_args.args[0]
        self.assertEqual(set(views), {'front', 'left'})
        self.assertEqual([views[v].name for v in ('front', 'left')], [c['file'] for c in cutouts])
        again = self.client.post('/api/generate', json=payload).json()
        self.assertEqual(again['id'], job['id'])
        self.assertEqual(self.engine.quality.generate.call_count, 1)

    def test_confirmation_missing_mask_wrong_target_and_stale_preview_rejected(self):
        source = self.source(2)
        first = self.segment(source).json()
        payload = self.payload(source, [first])
        response = self.client.post('/api/generate', json=payload)
        self.assertEqual(response.status_code, 400)
        second = self.segment(source, 1).json()
        payload = self.payload(source, [first, second])
        for override in ({'confirm_cutouts': False}, {'target': 'another object'}, {'cutout_files': {}}, {'use_cutouts': False}):
            response = self.client.post('/api/generate', json={**payload, **override})
            self.assertEqual(response.status_code, 400, response.text)
        self.engine.quality.generate.assert_not_called()

    def test_mask_edit_changes_generation_cache_key(self):
        source = self.source()
        first = self.segment(source).json()
        first_job = self.client.post('/api/generate', json=self.payload(source, [first])).json()
        self.wait_job(first_job['id'])
        original = self.engine.quality.segment.side_effect

        def edited(*args):
            path = original(*args)
            with Image.open(path) as image:
                mask = image.copy()
            ImageDraw.Draw(mask).rectangle((0, 0, 20, 20), fill=255)
            mask.save(path)
            return path
        self.engine.quality.segment.side_effect = edited
        second = self.segment(source, box=[0, 0, 1, 1]).json()
        second_job = self.client.post('/api/generate', json=self.payload(source, [second])).json()
        self.assertNotEqual(first_job['id'], second_job['id'])
        self.wait_job(second_job['id'])

    def test_texture_failure_preserves_shape_without_claiming_success(self):
        source = self.source()
        cutout = self.segment(source).json()
        mesh = self.engine.quality.generate.return_value

        def fail(views, directory, stopping, texture=True):
            (directory / 'shape.glb').write_bytes(mesh.read_bytes())
            raise ProviderError('paint OOM')
        self.engine.quality.generate.side_effect = fail
        response = self.client.post('/api/generate', json=self.payload(source, [cutout]))
        job = self.wait_job(response.json()['id'])
        self.assertEqual(job['status'], 'error')
        self.assertIn('asset_id', job)
        self.assertIn('without texture', job['message'])

    def test_bad_prompts_and_path_traversal_rejected(self):
        source = self.source()
        for payload in ({'box': [1, 0, 0, 1]}, {'points': [[0, 0]], 'labels': []}, {'points': [[2, 0]], 'labels': [1]}):
            response = self.segment(source, **payload)
            self.assertEqual(response.status_code, 422)
        response = self.client.post(f"/api/sources/{source['id']}/segment", json={'filename': '../meta.json', 'target': 'x'})
        self.assertEqual(response.status_code, 400)
        self.engine.quality.segment.assert_not_called()

    def test_normalize_preserves_existing_alpha_and_empty_mask_fails(self):
        original = self.root / 'alpha.png'
        target = self.root / 'normalized.png'
        image = Image.new('RGBA', (100, 100), (20, 50, 60, 0))
        ImageDraw.Draw(image).ellipse((10, 10, 90, 90), fill=(20, 50, 60, 255))
        image.save(original)
        normalize_image(original, target)
        with Image.open(target) as normalized:
            self.assertEqual(normalized.getpixel((0, 0))[3], 0)
        empty = self.root / 'empty.png'
        Image.new('L', (100, 100), 0).save(empty)
        with self.assertRaises(ValueError):
            make_cutout(original, empty, self.root / 'output.png')
if __name__ == '__main__':
    unittest.main()
