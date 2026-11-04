"""No live LLM/GPU: prove manual work bypasses stalled automatic requests."""
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import trimesh
from unicompose3d.api import create_app
from unicompose3d.clients import Gemini
from unicompose3d.schemas import TargetLocation

class AsyncSegmentationTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        llm = Mock(spec=Gemini)
        llm.key = 'fixture'
        llm.model = 'fixture'
        llm.locate.return_value = TargetLocation(found=True, confidence=0.9, box=[0.1, 0.1, 0.9, 0.9], explanation='fixture')
        self.app = create_app(self.root / 'data', 'fixture-token', llm)
        self.engine = self.app.state.engine
        self.llm = llm
        self.engine.quality = Mock()
        self.engine.quality.segment_ready = True

        def mask(image, box, points, labels, directory, stopping):
            out = directory / 'mask.png'
            with Image.open(image) as photo:
                Image.new('L', photo.size, 255).save(out)
            return out
        self.engine.quality.segment.side_effect = mask
        self.client = TestClient(self.app)
        self.client.headers['Authorization'] = 'Bearer fixture-token'
        photo = io.BytesIO()
        Image.new('RGB', (64, 64), 'red').save(photo, 'PNG')
        self.source = self.client.post('/api/sources', data={'kind': 'image', 'name': 'object'}, files={'files': ('photo.png', photo.getvalue(), 'image/png')}).json()
        self.request = {'filename': self.source['images'][0]['file'], 'target': 'object'}

    def tearDown(self):
        self.engine.segments.close()
        self.engine.pool.shutdown(wait=True)
        self.temp.cleanup()

    def start(self, **fields):
        result = self.client.post('/api/sources/' + self.source['id'] + '/segment-jobs', json={**self.request, **fields})
        self.assertEqual(result.status_code, 200, result.text)
        return result.json()

    def done(self, job):
        for _ in range(100):
            state = self.client.get('/api/segment-jobs/' + job['id']).json()
            if state['status'] in {'success', 'cancelled', 'error'}:
                return state
            time.sleep(0.01)
        self.fail('Segmentation job did not finish')

    def test_cancelled_llm_does_not_block_manual_mask_or_overwrite_it_later(self):
        entered, release = (threading.Event(), threading.Event())

        def locate(*args):
            entered.set()
            release.wait(5)
            return TargetLocation(found=True, confidence=0.9, box=[0.2, 0.2, 0.8, 0.8], explanation='late')
        self.llm.locate.side_effect = locate
        auto = self.start()
        self.assertTrue(entered.wait(1))
        cancelled = self.client.delete('/api/segment-jobs/' + auto['id']).json()
        self.assertEqual(cancelled['status'], 'cancelled')
        try:
            manual = self.start(box=[0.05, 0.05, 0.95, 0.95])
            result = self.done(manual)
            self.assertEqual(result['status'], 'success', result)
            self.assertFalse(release.is_set())
            self.assertEqual(result['result']['box'], [0.05, 0.05, 0.95, 0.95])
        finally:
            release.set()
        time.sleep(0.05)
        source = self.engine.store.read('sources', self.source['id'])
        self.assertEqual(source['cutouts'][self.request['filename']]['box'], [0.05, 0.05, 0.95, 0.95])
        self.assertEqual(self.engine.quality.segment.call_count, 1)

    def test_manual_job_never_calls_llm_and_reports_timings(self):
        job = self.start(box=[0.1, 0.1, 0.9, 0.9])
        result = self.done(job)
        self.assertEqual(result['status'], 'success')
        self.assertIn('locate_s', result['result']['timings'])
        self.llm.locate.assert_not_called()

    def test_hunyuan_creates_single_view_and_creates_a_reusable_library_asset(self):
        cut = self.done(self.start(box=[0.1, 0.1, 0.9, 0.9]))['result']
        mesh = self.root / 'fixture.glb'
        trimesh.creation.icosphere(subdivisions=1).export(mesh)
        self.engine.quality = Mock()
        self.engine.quality.ready = True
        self.engine.quality.generate.return_value = mesh
        body = {'source_id': self.source['id'], 'name': 'object', 'views': {'front': self.request['filename']}, 'use_cutouts': True, 'confirm_cutouts': True, 'target': 'object', 'cutout_files': {self.request['filename']: cut['file']}}
        response = self.client.post('/api/generate', json=body)
        self.assertEqual(response.status_code, 200, response.text)
        job = response.json()
        for _ in range(100):
            state = self.engine.store.read('jobs', job['id'])
            if state['status'] in {'success', 'error'}:
                break
            time.sleep(0.01)
        self.assertEqual(state['status'], 'success', state)
        self.assertEqual(state['provider'], 'hunyuan')
        self.assertTrue(self.engine.model_path(state['asset_id']).is_file())
        again = self.client.post('/api/generate', json=body).json()
        self.assertEqual(again['id'], job['id'])
        self.assertTrue(again['reused'])
        self.engine.quality.generate.assert_called_once()
if __name__ == '__main__':
    unittest.main()
