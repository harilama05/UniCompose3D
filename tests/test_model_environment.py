"""Offline regression for Kaggle ensurepip failure; never downloads model weights."""
import ast
import importlib.util
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("model_environment", ROOT / "tools/model_environment.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class SetupTests(unittest.TestCase):


    def test_environment_never_calls_broken_ensurepip_and_targets_private_pip(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(venv.EnvBuilder, "_setup_pip", side_effect=AssertionError("ensurepip unavailable")), patch.object(setup, "run") as run:
            target = Path(tmp) / "model-env"
            python = setup.create_model_environment(target)
            self.assertTrue(python.is_file())
            command = tuple(str(arg) for arg in run.call_args.args)
            self.assertIn("--python", command)
            self.assertEqual(command[command.index("--python") + 1], str(python))
            self.assertIn("--ignore-installed", command)
            self.assertIn("--no-deps", command)

    def test_host_pip_installs_into_target_without_ensurepip_offline(self):
        # An actual local wheel proves --python installs into the empty venv,
        # rather than the notebook interpreter. No registry/network required.
        with tempfile.TemporaryDirectory() as tmp, patch.object(venv.EnvBuilder, "_setup_pip", side_effect=AssertionError("ensurepip unavailable")):
            root = Path(tmp); target = root / "model-env"
            venv.EnvBuilder(with_pip=False, system_site_packages=True).create(target)
            python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            wheel = root / "unicompose3d_bootstrap_probe-1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("unicompose3d_bootstrap_probe.py", "VALUE = 42\n")
                archive.writestr("unicompose3d_bootstrap_probe-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: unicompose3d-bootstrap-probe\nVersion: 1.0\n")
                archive.writestr("unicompose3d_bootstrap_probe-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nGenerator: unicompose3d-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
                archive.writestr("unicompose3d_bootstrap_probe-1.0.dist-info/RECORD", "")
            result = subprocess.run([sys.executable, "-m", "pip", "--python", str(python), "install", "--no-index", "--no-deps", str(wheel)], env=setup.install_environment(python), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            check = subprocess.run([str(python), "-c", "import sys,json,unicompose3d_bootstrap_probe as p; print(json.dumps([sys.prefix, p.__file__, p.VALUE]))"], env=setup.install_environment(python), capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr)
            prefix, location, value = json.loads(check.stdout)
            self.assertEqual(Path(prefix).resolve(), target.resolve())
            self.assertTrue(Path(location).resolve().is_relative_to(target.resolve()))
            self.assertEqual(value, 42)


    def test_installer_environment_does_not_inherit_pip_destinations_or_keys(self):
        with patch.dict(os.environ, {"PIP_USER": "1", "PIP_TARGET": "wrong", "GEMINI_API_KEY": "secret", "TRIPO_API_KEY": "secret"}):
            env = setup.install_environment(Path("model-env/bin/python"))
        for key in ("PIP_USER", "PIP_TARGET", "GEMINI_API_KEY", "TRIPO_API_KEY"):
            self.assertNotIn(key, env)
        self.assertEqual(env["PIP_CONFIG_FILE"], os.devnull)
        self.assertEqual(env["USE_TF"], "0")
        self.assertEqual(env["PIP_NO_CACHE_DIR"], "1")




if __name__ == "__main__":
    unittest.main()
