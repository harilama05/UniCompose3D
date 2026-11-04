"""Regression for the observed Kaggle rasterizer patch failure. No GPU/network."""
import ast
import importlib.util
import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
with patch.object(sys, 'path', [str(ROOT / 'tools'), *sys.path]):
    spec = importlib.util.spec_from_file_location('setup_quality', ROOT / 'tools/setup_quality.py')
    setup = importlib.util.module_from_spec(spec); spec.loader.exec_module(setup)


class SetupQualityTests(unittest.TestCase):
    def source(self, separator='\n\n', flags=''):
        return ("# Keep upstream attribution/comment\n"
                "custom_rasterizer_module = CUDAExtension('custom_rasterizer_kernel', [\n    'rasterizer.cpp',\n]" + flags + ")"
                + separator + "setup(ext_modules=[custom_rasterizer_module])\n")

    def flags(self, source):
        node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'CUDAExtension')
        return ast.literal_eval(next(kw.value for kw in node.keywords if kw.arg == 'extra_compile_args'))

    def test_observed_blank_line_breaks_old_anchor_and_new_patch_works(self):
        source = self.source()
        self.assertNotIn('\n])\nsetup(', source)
        result = setup.patch_rasterizer(source)
        self.assertEqual(self.flags(result), {'cxx': ['-std=c++20'], 'nvcc': ['-std=c++20']})
        self.assertTrue(result.startswith('# Keep upstream attribution/comment\n'))
        self.assertTrue(result.endswith('setup(ext_modules=[custom_rasterizer_module])\n'))

    def test_whitespace_comments_crlf_and_inline_layout_do_not_control_patch(self):
        sources = [self.source(s) for s in ('\n', '\n\n\n', '\n# comment\n\n', '\r\n\r\n')]
        sources.append("# unicode: \u0111\u00e3\nmodule = CUDAExtension('custom_rasterizer_kernel', ['file.cpp'],)\nsetup()\n")
        for source in sources:
            with self.subTest(source=source):
                result = setup.patch_rasterizer(source)
                self.assertEqual(self.flags(result)['nvcc'], ['-std=c++20'])
                self.assertEqual(setup.patch_rasterizer(result), result)

    def test_existing_flags_are_preserved_and_old_standard_is_replaced(self):
        source = self.source(flags=", extra_compile_args={'cxx': ['-O2', '-std=c++17'], 'nvcc': ['--use_fast_math']}")
        flags = self.flags(setup.patch_rasterizer(source))
        self.assertEqual(flags['cxx'], ['-O2', '-std=c++20'])
        self.assertEqual(flags['nvcc'], ['--use_fast_math', '-std=c++20'])

    def test_unrecognized_extension_is_rejected_without_writing_wrong_call(self):
        with self.assertRaises(RuntimeError): setup.patch_rasterizer("setup(name='unrelated')")

    def test_fresh_install_uses_isolated_environment_no_pip_cache_and_pinned_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); repository = root / 'repo'; (repository / '.git').mkdir(parents=True)
            rasterizer = repository / 'hy3dgen/texgen/custom_rasterizer'; rasterizer.mkdir(parents=True)
            target = rasterizer / 'setup.py'; target.write_text(self.source())
            python = root / 'env/bin/python'
            torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: 'fixture T4'), __version__='fixture')
            argv = ['setup_quality.py', '--directory', str(repository), '--environment', str(root / 'env'), '--weights', str(root / 'weights')]
            result = types.SimpleNamespace(stdout='torch==2.11.0+cu128\ntorchvision==0.26.0+cu128\n')
            with patch.object(sys, 'platform', 'linux'), patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'torch': torch}), patch.object(setup, 'run', return_value=result) as run, patch.object(setup, 'create_model_environment', return_value=python) as create, patch.object(setup, 'complete_downloads') as complete:
                setup.main()
            create.assert_called_once_with(root / 'env')
            complete.assert_called_once_with(python, repository.resolve(), (root / 'weights').resolve())
            self.assertEqual(self.flags(target.read_text())['nvcc'], ['-std=c++20'])
            installs = [call for call in run.call_args_list if 'pip' in call.args and 'install' in call.args]
            self.assertTrue(installs)
            for call in installs:
                self.assertEqual(call.kwargs['env']['PIP_NO_CACHE_DIR'], '1')
                self.assertNotIn('torch', call.args)
            deps = next(call for call in installs if 'diffusers==0.32.2' in call.args)
            self.assertIn('-c', deps.args)
            self.assertEqual((root / 'weights/cuda-constraints.txt').read_text(), result.stdout)



if __name__ == '__main__': unittest.main()
