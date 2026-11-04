"""Shared isolated model environment; reuse notebook CUDA packages."""
from __future__ import annotations
import os,subprocess,sys,venv
from pathlib import Path

def install_environment(python=None):
    env = dict(os.environ)
    for name in ("PIP_USER", "PIP_TARGET", "PIP_PREFIX", "PYTHONHOME", "PYTHONPATH", "TRIPO_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "HF_TOKEN", "TRYON_ACCESS_TOKEN"):
        env.pop(name, None)
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_CACHE_DIR="1", USE_TF="0", USE_FLAX="0")
    if python is not None:
        env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    return env

def run(*args, **kwargs):
    kwargs.setdefault("env", install_environment())
    return subprocess.run([str(value) for value in args], check=True, **kwargs)

def create_model_environment(environment):
    """Avoid ensurepip entirely; host pip explicitly manages the target venv."""
    environment = Path(environment).resolve()
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # --ignore-installed seeds a PRIVATE pip even if host pip is inherited.
    run(sys.executable, "-m", "pip", "--python", python, "install", "--ignore-installed", "--no-deps", "pip", env=install_environment(python))
    return python
