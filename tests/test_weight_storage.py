"""Disk-full regression: exact cache eviction, resumable accounting, mode selection."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
from unicompose3d import weight_storage as storage
import download_quality_weights as download
from setup_quality import download_spec


class StorageTests(unittest.TestCase):
    def test_readiness_requires_all_selected_weights_with_complete_sizes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {name: (2, sha) for name, (_, sha) in storage.HEAVY_FILES.items()}
            with patch.dict(storage.HEAVY_FILES, files, clear=True):
                for name in storage.selected_files():
                    path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'xx')
                storage.verify_weights(root)
                target = root / storage.shape_file(False); target.write_bytes(b'x')
                with self.assertRaisesRegex(RuntimeError, 'hunyuan3d-dit-v2-0'):
                    storage.verify_weights(root)
                target.write_bytes(b'xx')
                paint = root / 'single/hunyuan3d-paint-v2-0-turbo/vae/diffusion_pytorch_model.bin'; paint.unlink()
                with self.assertRaisesRegex(RuntimeError, 'vae'):
                    storage.verify_weights(root)

    def test_only_redundant_unet_binary_and_its_exact_partial_are_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            duplicate = root / storage.REDUNDANT
            partial = storage.incomplete_path(root, storage.REDUNDANT, storage.REDUNDANT_SHA)
            keep = [root / 'photos/hand.jpg', root / 'result.glb', root / 'project.zip',
                    root / 'single/hunyuan3d-paint-v2-0-turbo/text_encoder/pytorch_model.bin',
                    root / 'single/hunyuan3d-paint-v2-0-turbo/vae/diffusion_pytorch_model.bin',
                    duplicate.with_suffix('.safetensors'), partial.with_name('different.incomplete')]
            for path in [duplicate, partial, *keep]:
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'12345')
            self.assertEqual(storage.remove_redundant(root), 10)
            self.assertFalse(duplicate.exists()); self.assertFalse(partial.exists())
            self.assertTrue(all(path.read_bytes() == b'12345' for path in keep))

    def test_all_paths_are_validated_before_any_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); root = base / 'weights'; root.mkdir()
            inside = root / 'cache'; inside.write_bytes(b'x')
            outside = base / 'user.glb'; outside.write_bytes(b'y')
            with self.assertRaises(ValueError):
                storage.unlink_cache(root, [inside, outside])
            self.assertEqual(inside.read_bytes(), b'x'); self.assertEqual(outside.read_bytes(), b'y')

    def test_complete_and_partial_bytes_reduce_required_space(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); name = storage.shape_file(False)
            with patch.dict(storage.HEAVY_FILES, {name: (10, 'fixture-sha')}):
                self.assertEqual(storage.needed(root, name), 10)
                partial = storage.incomplete_path(root, name, 'fixture-sha'); partial.write_bytes(b'123456')
                self.assertEqual(storage.needed(root, name), 4)
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'0123456789')
                self.assertEqual(storage.needed(root, name), 0)

    def test_preflight_stops_when_insufficient_and_accepts_resumed_download(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(storage, 'needed', return_value=100), patch.object(storage, 'RESERVE', 10), patch.object(storage.shutil, 'disk_usage', return_value=SimpleNamespace(free=109)):
                with self.assertRaises(RuntimeError): storage.check_space(temp, shape_only=True)
            with patch.object(storage, 'needed', return_value=4), patch.object(storage, 'RESERVE', 10), patch.object(storage.shutil, 'disk_usage', return_value=SimpleNamespace(free=14)):
                storage.check_space(temp, shape_only=True)

    def test_mode_switch_evicts_only_other_shape_leaves_texture_and_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); other = root / storage.shape_file(False)
            partial = storage.incomplete_path(root, storage.shape_file(False), storage.HEAVY_FILES[storage.shape_file(False)][1])
            keep = [root / storage.shape_file(True), root / 'single/hunyuan3d-paint-v2-0-turbo/vae/diffusion_pytorch_model.bin', root / 'scene.glb']
            for path in [other, partial, *keep]:
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'x')
            self.assertEqual(storage.evict_other_shape(root, True), 2)
            self.assertTrue(all(path.is_file() for path in keep))

    def test_spec_selects_one_shape_checkpoint_and_ignores_only_duplicate(self):
        for multiview in (False, True):
            specs = download_spec(Path('/weights'), multiview)
            patterns = [pattern for spec in specs for pattern in spec[3]]
            self.assertIn(storage.shape_file(multiview).split('/', 1)[1], patterns)
            self.assertNotIn(storage.shape_file(not multiview).split('/', 1)[1], patterns)
        self.assertEqual(storage.IGNORE_PATTERNS, [storage.REDUNDANT.split('/', 1)[1]])

    def test_lazy_multiview_fetch_evicts_only_when_space_insufficient(self):
        with patch.object(download, 'remove_redundant'), patch.object(download, 'check_space', side_effect=[RuntimeError('space'), None]) as check, patch.object(download, 'evict_other_shape') as evict, patch.object(download, 'needed', return_value=10), patch.object(download, 'download_snapshots') as fetch:
            download.ensure_shape(Path('/weights'), True)
            self.assertEqual(check.call_count, 2); evict.assert_called_once_with(Path('/weights'), True)
            spec = fetch.call_args.args[0][0]
            self.assertEqual(spec[0], 'tencent/Hunyuan3D-2mv')
            self.assertEqual(spec[3], ['hunyuan3d-dit-v2-mv-turbo/config.yaml', 'hunyuan3d-dit-v2-mv-turbo/model.fp16.safetensors'])

    def test_no_download_when_selected_checkpoint_and_config_already_exist(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); config = (root / storage.shape_file(False)).with_name('config.yaml')
            config.parent.mkdir(parents=True); config.touch()
            with patch.object(download, 'remove_redundant'), patch.object(download, 'check_space'), patch.object(download, 'needed', return_value=0), patch.object(download, 'download_snapshots') as fetch, patch.object(download, 'evict_other_shape') as evict:
                download.ensure_shape(root, False)
            fetch.assert_not_called(); evict.assert_not_called()

    def test_still_full_after_eviction_never_starts_download(self):
        with patch.object(download, 'remove_redundant'), patch.object(download, 'check_space', side_effect=RuntimeError('space')), patch.object(download, 'evict_other_shape'), patch.object(download, 'download_snapshots') as fetch:
            with self.assertRaisesRegex(RuntimeError, 'space'):
                download.ensure_shape(Path('/weights'), True)
            fetch.assert_not_called()




if __name__ == '__main__': unittest.main()
