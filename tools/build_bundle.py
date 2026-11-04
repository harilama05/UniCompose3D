"""Build a clean Kaggle source bundle with a validated launcher notebook."""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dist"
ARCHIVE_ROOT = "UniCompose3D"
EXCLUDED = {"data", "test-results", "__pycache__", ".venv", "venv", ".git",
            ".pytest_cache", ".cache", "dist", "build", "weights", "models",
            ".idea", ".vscode"}
PRIVATE_SUFFIXES = {".pyc", ".log", ".pem", ".key", ".glb", ".gltf", ".zip",
                    ".safetensors", ".ckpt", ".pth", ".pt", ".onnx"}


def included(relative):
    """Allow source and documentation, never user data or local secrets."""
    return (not set(relative.parts) & EXCLUDED
            and not any(part.endswith(".egg-info") for part in relative.parts)
            and relative.suffix.lower() not in PRIVATE_SUFFIXES
            and not (relative.name.startswith(".env") and relative.name != ".env.example"))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(ROOT / "notebooks/UniCompose3D_Kaggle.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            if cell.get("outputs") or cell.get("execution_count") is not None:
                raise ValueError("Release notebooks must have no outputs or execution state.")
    notebook_path = OUTPUT / "UniCompose3D_Kaggle.ipynb"
    nbformat.write(notebook, notebook_path)
    shutil.copyfile(ROOT / "docs/kaggle.md", OUTPUT / "KAGGLE.md")
    archive_path = OUTPUT / "UniCompose3D.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            relative = path.relative_to(ROOT)
            if path.is_file() and not path.is_symlink() and included(relative):
                archive.write(path, (Path(ARCHIVE_ROOT) / relative).as_posix())
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (OUTPUT / "UniCompose3D.sha256").write_text(digest + "  UniCompose3D.zip\n", encoding="utf-8")
    print("Created:", notebook_path, archive_path)
    print("Source ZIP bytes:", archive_path.stat().st_size)
    print("SHA256:", digest)


if __name__ == "__main__":
    main()
