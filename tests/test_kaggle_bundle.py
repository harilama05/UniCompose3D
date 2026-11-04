"""Exercise the generated release and notebook bootstrap without API/GPU calls."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import nbformat

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bundle_builder', ROOT / 'tools/build_bundle.py')
builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)


class KaggleBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.addClassCleanup(cls.temp.cleanup)
        cls.output = Path(cls.temp.name)
        with patch.object(builder, 'OUTPUT', cls.output), contextlib.redirect_stdout(io.StringIO()):
            builder.main()
        cls.book = nbformat.read(cls.output / 'UniCompose3D_Kaggle.ipynb', as_version=4)

    def test_bundle_contains_runtime_fixes_and_excludes_user_data_logs_and_old_cells(self):
        archive = self.output / 'UniCompose3D.zip'
        with zipfile.ZipFile(archive) as z:
            self.assertIsNone(z.testzip())
            names = z.namelist()
            for name in names:
                relative = Path(name).relative_to('UniCompose3D')
                self.assertFalse(set(relative.parts) & {'data', '__pycache__', '.venv', 'test-results'})
                self.assertNotIn(relative.suffix, ('.log', '.pyc', '.env'))
                self.assertFalse(relative.name.startswith(('KAGGLE_SUA_', 'KAGGLE_TAI_')))
                if relative.suffix == '.py': ast.parse(z.read(name).decode('utf-8'))
            for name in ('tools/paint_compat.py', 'tools/setup_quality.py', 'tools/retry_quality_texture.py', 'unicompose3d/weight_storage.py', 'tools/resident_worker.py'):
                self.assertIn('UniCompose3D/' + name, names)
        expected = (self.output / 'UniCompose3D.sha256').read_text().split()[0]
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), expected)
        self.assertEqual((self.output / 'KAGGLE.md').read_bytes(), (ROOT / 'docs/kaggle.md').read_bytes())

    def test_default_setup_is_hunyuan_and_sam_without_optional_secrets(self):
        code = next(c.source for c in self.book.cells if c.cell_type == "code" and 'BACKEND = "hunyuan"' in c.source)
        module = types.SimpleNamespace(UserSecretsClient=Mock())
        with patch.dict(os.environ,{},clear=True), patch.dict(sys.modules,{"kaggle_secrets":module}), patch("subprocess.check_call") as run, contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code,"<integrated-setup>","exec"),{"PROJECT":ROOT})
            self.assertEqual(run.call_count,1)
            self.assertIn("setup_quality.py",run.call_args.args[0][2])
            self.assertEqual(os.environ["TRYON_SEGMENT_PYTHON"],os.environ["TRYON_HUNYUAN_PYTHON"])
            self.assertEqual(os.environ["TRYON_DEFAULT_PROVIDER"],"hunyuan")
            module.UserSecretsClient.assert_not_called()

    def test_primary_notebook_does_not_require_external_trial_or_obsolete_live_patches(self):
        source = "\\n".join(c.source for c in self.book.cells)
        self.assertNotIn("TRELLIS_KIEM_CHUNG_KAGGLE",source)
        self.assertNotIn("fixes =",source)
        self.assertNotIn("## 3B.",source)
        self.assertIn('BACKEND = "hunyuan"',source)

    def test_fresh_notebook_bootstrap_extracts_release_and_verifies_vendor_assets(self):
        nbformat.validate(self.book)
        code = next(cell.source for cell in self.book.cells if cell.cell_type == 'code')
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); inputs = base / 'input'; inputs.mkdir(); work = base / 'working'; work.mkdir()
            shutil.copyfile(self.output / 'UniCompose3D.zip', inputs / 'UniCompose3D.zip')
            code = code.replace('Path("/kaggle/input")', 'Path(' + repr(str(inputs)) + ')')
            code = code.replace('Path("/kaggle/working")', 'Path(' + repr(str(work)) + ')')
            with patch.dict(sys.modules), patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stdout(io.StringIO()):
                sys.modules.pop('unicompose3d', None); sys.modules.pop('launch', None)
                state = {}; exec(compile(code, '<kaggle-step-1>', 'exec'), state)
                self.assertIn('__version__ = "' + state['EXPECTED_VERSION'] + '"', (state['PROJECT'] / 'unicompose3d/__init__.py').read_text())
                self.assertTrue((state['PROJECT'] / 'tools/paint_compat.py').is_file())
                sys.modules['unicompose3d'] = types.ModuleType('unicompose3d')
                with self.assertRaises(RuntimeError): exec(compile(code, '<kaggle-step-1>', 'exec'), {})

    def test_notebook_secrets_are_read_without_printing_key_and_optional_actions_are_inert(self):
        code = next(cell.source for cell in self.book.cells if cell.cell_type == 'code' and 'for key in (' in cell.source and 'OPENAI_API_KEY' in cell.source)
        client = Mock(); client.get_secret.side_effect = lambda key: {'GEMINI_API_KEY':'fixture-gemini-secret','OPENAI_API_KEY':'fixture-openai-secret','HF_TOKEN':'fixture-hf-token'}.get(key,'')
        module = types.SimpleNamespace(UserSecretsClient=Mock(return_value=client))
        output = io.StringIO()
        with patch.dict(sys.modules, {'kaggle_secrets': module}), patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(output):
            exec(compile(code, '<kaggle-secrets>', 'exec'), {})
            self.assertNotIn('GEMINI_API_KEY', os.environ)
            self.assertNotIn('TRIPO_API_KEY', os.environ)
            self.assertEqual(os.environ['OPENAI_API_KEY'], 'fixture-openai-secret')
            self.assertEqual(os.environ['TRYON_LLM_PROVIDER'], 'openai')
        self.assertNotIn('fixture-gemini-secret', output.getvalue())
        self.assertNotIn('fixture-openai-secret', output.getvalue())
        self.assertNotIn('fixture-hf-token', output.getvalue())
        texts = '\n'.join(cell.source for cell in self.book.cells)
        self.assertNotIn('KAGGLE_SUA_', texts)
        for cell in self.book.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                if 'retry_quality_texture.py' in cell.source:
                    self.assertFalse(ast.parse(cell.source).body)  # Comments until explicitly enabled.


if __name__ == '__main__': unittest.main()

