"""Retry bookkeeping and saved-mesh routing; GPU Paint is an offline fixture."""
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

import trimesh
from PIL import Image
from unicompose3d.storage import Store

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from retry_quality_texture import retry_texture
spec = importlib.util.spec_from_file_location('texture_worker', ROOT / 'tools/quality_worker.py')
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


class TextureRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name))
        self.source_id, source = self.store.create('sources', {})
        self.image = source / 'cutout.png'; Image.new('RGBA', (16, 16), 'red').save(self.image)
        self.ident, self.directory = self.store.create('jobs', {'name': 'test', 'provider': 'hunyuan', 'texture': True,
            'status': 'error', 'source_id': self.source_id, 'asset_id': 'original-shape'})
        self.mesh = trimesh.creation.icosphere(subdivisions=1)
        self.mesh.export(self.directory / 'shape.glb')
        self.shape_bytes = (self.directory / 'shape.glb').read_bytes()
        self.original_request = {'views': {'front': str(self.image)}, 'texture': True, 'output': str(self.directory / 'result.glb')}
        (self.directory / 'reconstruct-request.json').write_text(json.dumps(self.original_request))
        (self.directory / 'reconstruct.log').write_text('original failure log')
        self.engine = Mock(store=self.store, active=set(), lock=threading.RLock(), stopping=threading.Event())
        self.engine.asset.return_value = {'id': 'new-textured-asset'}

    def test_retries_paint_only_and_preserves_shape_request_and_log(self):
        def complete(task, request, directory, stopping):
            self.assertEqual(task, 'texture')
            self.assertEqual(request['shape'], str(self.directory / 'shape.glb'))
            Path(request['output']).write_bytes(self.shape_bytes)
        self.engine.quality.run.side_effect = complete
        result = retry_texture(self.engine)
        job = self.store.read('jobs', self.ident)
        self.assertEqual(job['status'], 'success'); self.assertEqual(job['asset_id'], result['id'])
        self.engine.quality.generate.assert_not_called()
        self.assertEqual((self.directory / 'shape.glb').read_bytes(), self.shape_bytes)
        self.assertEqual((self.directory / 'reconstruct.log').read_text(), 'original failure log')
        self.assertEqual(json.loads((self.directory / 'reconstruct-request.json').read_text()), self.original_request)
        self.assertFalse(self.engine.active)

    def test_failure_keeps_previous_asset_and_releases_active_slot(self):
        self.engine.quality.run.side_effect = RuntimeError('texture fixture failure')
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'): retry_texture(self.engine)
        job = self.store.read('jobs', self.ident)
        self.assertEqual(job['status'], 'error'); self.assertEqual(job['asset_id'], 'original-shape')
        self.assertFalse(self.engine.active)

    def test_busy_and_invalid_front_image_never_run_model(self):
        self.engine.active.add('other-job')
        with self.assertRaises(RuntimeError): retry_texture(self.engine)
        self.engine.active.clear()
        request = {**self.original_request, 'views': {'front': str(self.directory / 'shape.glb')}}
        (self.directory / 'reconstruct-request.json').write_text(json.dumps(request))
        with self.assertRaises(ValueError): retry_texture(self.engine)
        self.engine.quality.run.assert_not_called()

    def test_worker_texture_uses_saved_mesh_without_shape_reconstruction(self):
        request = {'shape': str(self.directory / 'shape.glb'), 'views': {'front': str(self.image)}, 'output': str(self.directory / 'painted.glb')}
        with patch('torch.cuda.is_available', return_value=True), patch.object(worker, 'paint_mesh', return_value=self.mesh) as paint, patch.object(worker, 'reconstruct') as reconstruct:
            worker.texture(request, Path('/repo'), Path('/weights'))
        reconstruct.assert_not_called(); paint.assert_called_once()
        self.assertTrue(Path(request['output']).is_file())
        self.assertEqual(len(paint.call_args.args[0].faces), len(self.mesh.faces))



if __name__ == '__main__': unittest.main()
