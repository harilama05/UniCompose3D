"""Offline regression for the observed HTTP 503 after successful CUDA build."""
import importlib.util
import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

ROOT = Path(__file__).resolve().parents[1]
with patch.object(sys, 'path', [str(ROOT / 'tools'), *sys.path]):
    spec = importlib.util.spec_from_file_location('quality_download', ROOT / 'tools/download_quality_weights.py')
    download = importlib.util.module_from_spec(spec); spec.loader.exec_module(download)
    import setup_quality as setup


def http_error(status):
    response = requests.Response(); response.status_code = status
    return requests.HTTPError('test HTTP ' + str(status), response=response)


class DownloadTests(unittest.TestCase):
    def test_503_then_success_preserves_destination_and_disables_forced_redownload(self):
        target = '/existing/cache'
        fetch = Mock(side_effect=[http_error(503), target]); sleep = Mock()
        download.download_snapshots([['repo', 'pinned-sha', target, ['weights/*']]], fetch, sleep)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(fetch.call_args.kwargs['local_dir'], target)
        self.assertEqual(fetch.call_args.kwargs['max_workers'], 1)
        self.assertFalse(fetch.call_args.kwargs['force_download'])
        self.assertEqual(fetch.call_args.kwargs['ignore_patterns'], ['hunyuan3d-paint-v2-0-turbo/unet/diffusion_pytorch_model.bin'])
        sleep.assert_called_once_with(2)

    def test_permissions_and_missing_revisions_are_not_retried(self):
        for status in (401, 403, 404):
            fetch = Mock(side_effect=http_error(status)); sleep = Mock()
            with self.assertRaises(requests.HTTPError):
                download.download_snapshots([['repo', 'sha', 'cache', ['*']]], fetch, sleep)
            self.assertEqual(fetch.call_count, 1); sleep.assert_not_called()

    def test_network_timeout_is_retried_but_failure_is_bounded(self):
        fetch = Mock(side_effect=requests.ReadTimeout('temporary')); sleep = Mock()
        with self.assertRaisesRegex(RuntimeError, 'resume-downloads'):
            download.download_snapshots([['repo', 'sha', 'cache', ['*']]], fetch, sleep, attempts=3)
        self.assertEqual(fetch.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])

    def test_resume_cli_does_not_install_packages_or_compile(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); repository = root / 'repo'; (repository / 'hy3dgen').mkdir(parents=True)
            env = root / 'env'; (env / 'bin').mkdir(parents=True); (env / 'bin/python').touch()
            torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: 'fixture T4'), __version__='fixture')
            argv = ['setup_quality.py', '--resume-downloads', '--directory', str(repository), '--environment', str(env), '--weights', str(root / 'weights')]
            with patch.object(sys, 'platform', 'linux'), patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'torch': torch}), patch.object(setup, 'run') as run, patch.object(setup, 'create_model_environment') as create:
                setup.main()
            create.assert_not_called()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[2].name, 'download_quality_weights.py')
            self.assertEqual(run.call_args.kwargs['env']['HF_HUB_DOWNLOAD_TIMEOUT'], '120')

    def test_fresh_finish_patches_paint_before_ready_and_rejects_incomplete_weights(self):
        from test_paint_compat import LOADER
        from unicompose3d import weight_storage as storage
        from paint_compat import MODULE, MARKER
        for incomplete in (False, True):
            with self.subTest(incomplete=incomplete), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); weights = root / 'weights'
                tiny_files = {name: (2, sha) for name, (_, sha) in storage.HEAVY_FILES.items()}
                with patch.dict(storage.HEAVY_FILES, tiny_files, clear=True):
                    for name in storage.selected_files():
                        path = weights / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'xx')
                    module = weights / MODULE; module.write_text(LOADER)
                    if incomplete:
                        (weights / storage.shape_file(False)).write_bytes(b'x')
                    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True), zeros=Mock())
                    modules = {'torch': torch, 'torchvision': types.ModuleType('torchvision'), 'custom_rasterizer': types.ModuleType('custom_rasterizer'),
                               'transformers': types.SimpleNamespace(SamModel=Mock(), SamProcessor=Mock(), CLIPVisionModel=object),
                               'hy3dgen.shapegen': types.SimpleNamespace(Hunyuan3DDiTFlowMatchingPipeline=object),
                               'hy3dgen.texgen': types.SimpleNamespace(Hunyuan3DPaintPipeline=object)}
                    with patch('hunyuan_compat.patch_repository'), patch('hunyuan_compat.probe_decoder'), patch.dict(sys.modules, modules), patch.object(download, 'check_space'), patch.object(download, 'download_snapshots') as fetch:
                        if incomplete:
                            with self.assertRaises(RuntimeError): download.finish(root / 'repo', weights)
                        else:
                            download.finish(root / 'repo', weights)
                            self.assertIn(MARKER, module.read_text())
                            self.assertEqual((weights / '.quality-ready').read_text(), setup.QUALITY_SETTINGS)
                        self.assertEqual((weights / '.quality-ready').exists(), not incomplete)
                        self.assertEqual((weights / '.segment-ready').exists(), not incomplete)
                        self.assertEqual(len(fetch.call_args.args[0]), 2)  # No second shape checkpoint.



if __name__ == '__main__': unittest.main()
