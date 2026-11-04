"""Observed custom-loader failure; actual CPU torch/safetensors, strict checks."""
import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from paint_compat import MODULE, patch_loader, patch_paint_weights
from unicompose3d.clients import ProviderError
from unicompose3d.quality import QualityBackend

# The loader body from Tencent's pinned checkpoint modules.py (Apache 2.0
# attribution in the original). Keep the failure-triggering code unchanged.
LOADER = '''# Upstream license/comment retained
class UNet2p5DConditionModel:
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        torch_dtype = kwargs.pop('torch_dtype', torch.float32)
        config_path = os.path.join(pretrained_model_name_or_path, 'config.json')
        unet_ckpt_path = os.path.join(pretrained_model_name_or_path, 'diffusion_pytorch_model.bin')
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        unet = UNet2DConditionModel(**config)
        unet = UNet2p5DConditionModel(unet)
        unet_ckpt = torch.load(unet_ckpt_path, map_location='cpu', weights_only=True)
        unet.load_state_dict(unet_ckpt, strict=True)
        unet = unet.to(torch_dtype)
        return unet
'''


class TinyUNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.unet = torch.nn.Module()
        self.unet.block = torch.nn.Module()
        self.unet.block.attn2 = torch.nn.Module()
        self.unet.block.attn2.to_k = torch.nn.Linear(3, 2, bias=False)


def invoke(source, directory):
    tree = ast.parse(source)
    method = tree.body[0].body[0]; method.decorator_list = []
    namespace = {'torch': torch, 'os': os, 'json': json, 'UNet2DConditionModel': TinyUNet,
                 'UNet2p5DConditionModel': lambda model: model}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<custom-loader>', 'exec'), namespace)
    return namespace['from_pretrained'](str(directory), torch_dtype=torch.float16)


class PaintTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'config.json').write_text('{}')
        self.state = {key: torch.full_like(value, .25) for key, value in TinyUNet().state_dict().items()}

    def save(self, state=None):
        save_file(self.state if state is None else state, str(self.root / 'diffusion_pytorch_model.safetensors'))

    def test_observed_loader_requires_bin_even_when_safetensors_exists(self):
        self.save()
        with self.assertRaises(FileNotFoundError): invoke(LOADER, self.root)

    def test_patched_loader_loads_real_safetensors_and_keeps_dtype(self):
        self.save()
        result = invoke(patch_loader(LOADER), self.root)
        self.assertTrue(all(torch.equal(value, self.state[key].half()) for key, value in result.state_dict().items()))
        self.assertEqual(next(result.parameters()).dtype, torch.float16)
        self.assertFalse((self.root / 'diffusion_pytorch_model.bin').exists())

    def test_unused_known_ip_projections_are_handled_without_disabling_strict(self):
        state = {**self.state, 'unet.block.attn2.to_k_ip.weight': torch.ones(2, 3),
                 'unet.image_proj_model.norm.bias': torch.ones(1024)}
        self.save(state)
        model = invoke(patch_loader(LOADER), self.root)
        self.assertEqual(set(model.state_dict()), set(self.state))

    def test_missing_wrong_shape_and_unrelated_extra_weights_still_fail(self):
        key = next(iter(self.state))
        cases = [{}, {key: torch.zeros(4, 5)}, {**self.state, 'unexpected': torch.ones(1)},
                 {**self.state, 'unet.block.attn2.to_k_ip.weight': torch.ones(5, 5)},
                 {**self.state, 'unet.image_proj_model.norm.bias': torch.ones(5)}]
        for state in cases:
            self.save(state)
            with self.assertRaises(RuntimeError): invoke(patch_loader(LOADER), self.root)

    def test_bin_fallback_and_safetensors_preference(self):
        torch.save({key: torch.zeros_like(value) for key, value in self.state.items()}, self.root / 'diffusion_pytorch_model.bin')
        model = invoke(patch_loader(LOADER), self.root)
        self.assertEqual(float(next(model.parameters()).detach().sum()), 0)
        self.save()
        model = invoke(patch_loader(LOADER), self.root)
        self.assertGreater(float(next(model.parameters()).detach().sum()), 0)

    def test_patch_is_idempotent_keeps_license_and_rejects_wrong_code(self):
        source = patch_loader(LOADER)
        self.assertEqual(source, patch_loader(source))
        self.assertTrue(source.startswith('# Upstream license/comment retained'))
        with self.assertRaises(RuntimeError): patch_loader('class Other: pass')
        target = self.root / MODULE; target.parent.mkdir(parents=True)
        target.write_text(LOADER); target.with_name('diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
        patch_paint_weights(self.root); first = target.read_text(); patch_paint_weights(self.root)
        self.assertEqual(first, target.read_text())

    def test_backend_preserves_final_exception_and_redacts_credentials(self):
        backend = QualityBackend()
        def failed_process(*args, **kwargs):
            import io,json
            kwargs['stderr'].write('traceback frame\n'*200+'FileNotFoundError: missing.bin secret-value\n')
            process=Mock(returncode=1,poll=Mock(return_value=None))
            process.stdin=io.StringIO()
            process.stdout=io.StringIO(json.dumps({'event':'error','error':'FileNotFoundError: missing.bin secret-value'})+'\n')
            return process
        with patch.object(QualityBackend, 'ready', new_callable=PropertyMock, return_value=True), patch('subprocess.Popen', side_effect=failed_process), patch.dict(os.environ, {'GEMINI_API_KEY': 'secret-value'}):
            with self.assertRaises(ProviderError) as caught:
                backend.run('reconstruct', {}, self.root, Mock(is_set=Mock(return_value=False)))
        message = str(caught.exception)
        self.assertIn('FileNotFoundError: missing.bin', message)
        self.assertNotIn('secret-value', message)


if __name__ == '__main__': unittest.main()
