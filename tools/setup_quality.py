"""Install pinned Hunyuan3D/SAM into their own environment on Kaggle GPU."""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_environment import create_model_environment, install_environment, run
from unicompose3d.quality import CODE_REVISION, SHAPE_REVISION, MULTIVIEW_REVISION, SAM_REVISION, QUALITY_SETTINGS, SEGMENT_SETTINGS

REQUIREMENTS = [
    "numpy==2.2.6", "transformers==4.48.3", "tokenizers==0.21.1", "huggingface-hub==0.28.1",
    "diffusers==0.32.2", "accelerate==1.3.0", "omegaconf==2.3.0", "einops==0.8.1",
    "trimesh==4.7.4", "pymeshlab==2025.7.post1", "pygltflib==1.16.5", "xatlas==0.0.11",
    "scikit-image>=0.25,<0.26", "Pillow>=10.4,<12", "opencv-python-headless==4.11.0.86",
    "safetensors>=0.5,<1", "timm==1.0.15", "psutil>=5.9,<8", "setuptools", "wheel", "ninja", "pybind11>=3,<4",
]


def download_spec(weights, multiview=False):
    specs = [
        ["facebook/sam-vit-base", SAM_REVISION, str(weights / "sam"), ["config.json", "preprocessor_config.json", "model.safetensors"]],
        ["tencent/Hunyuan3D-2", SHAPE_REVISION, str(weights / "single"), ["hunyuan3d-paint-v2-0-turbo/*"]],
    ]
    if multiview:
        specs.append(["tencent/Hunyuan3D-2mv", MULTIVIEW_REVISION, str(weights / "multiview"), ["hunyuan3d-dit-v2-mv-turbo/config.yaml", "hunyuan3d-dit-v2-mv-turbo/model.fp16.safetensors"]])
    else:
        specs[1][3].extend(["hunyuan3d-dit-v2-0-turbo/config.yaml", "hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors"])
    return specs


def patch_rasterizer(source):
    """Add compiler flags to the CUDAExtension call, independent of whitespace.

    Preserve upstream comments/license and all other setup() arguments.
    AST column offsets are UTF-8 bytes, whereas string slices use characters.
    """
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "CUDAExtension"
             and node.args and isinstance(node.args[0], ast.Constant)
             and node.args[0].value == "custom_rasterizer_kernel"]
    if len(calls) != 1:
        raise RuntimeError("Expected exactly one custom_rasterizer_kernel CUDAExtension to patch compiler flags.")
    call = calls[0]
    lines = source.splitlines(keepends=True)

    def offset(line, column):
        return sum(len(text) for text in lines[:line - 1]) + len(lines[line - 1].encode("utf-8")[:column].decode("utf-8"))

    keyword = next((item for item in call.keywords if item.arg == "extra_compile_args"), None)
    if keyword:
        flags = ast.literal_eval(keyword.value)
        if not isinstance(flags, dict) or any(not isinstance(v, list) for v in flags.values()):
            raise RuntimeError("Unexpected extra_compile_args structure; inspect it before patching.")
        for language in ("cxx", "nvcc"):
            flags[language] = [flag for flag in flags.get(language, []) if not flag.startswith("-std=")] + ["-std=c++20"]
        start = offset(keyword.value.lineno, keyword.value.col_offset)
        end = offset(keyword.value.end_lineno, keyword.value.end_col_offset)
        result = source[:start] + repr(flags) + source[end:]
    else:
        last = call.keywords[-1] if call.keywords else call.args[-1]
        end = offset(last.end_lineno, last.end_col_offset)
        result = source[:end] + ", extra_compile_args={'cxx': ['-std=c++20'], 'nvcc': ['-std=c++20']}" + source[end:]
    ast.parse(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="/kaggle/working/Hunyuan3D-2")
    parser.add_argument("--environment", default="/kaggle/working/tryon-quality-env")
    parser.add_argument("--weights", default="/kaggle/working/tryon-quality-weights")
    parser.add_argument("--resume-downloads", action="store_true", help="Only verify existing CUDA runtime and resume weights; skip installation/build")
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("This installer targets Linux/Kaggle GPU.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a T4/P100 GPU and Internet access before installation.")
    print("GPU:", torch.cuda.get_device_name(0), "Torch:", torch.__version__, flush=True)
    repository, weights = Path(args.directory).resolve(), Path(args.weights).resolve()
    weights.mkdir(parents=True, exist_ok=True)
    if args.resume_downloads:
        python = Path(args.environment).resolve() / "bin/python"
        complete_downloads(python, repository, weights)
        return
    (weights / ".quality-ready").unlink(missing_ok=True)
    (weights / ".segment-ready").unlink(missing_ok=True)
    if not (repository / ".git").is_dir():
        if repository.exists() and any(repository.iterdir()):
            raise RuntimeError("The destination contains unrelated data. Choose a separate --directory.")
        run("git", "clone", "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git", repository)
    run("git", "fetch", "origin", CODE_REVISION, cwd=repository)
    run("git", "checkout", "--detach", CODE_REVISION, cwd=repository)
    python = create_model_environment(Path(args.environment))
    env = install_environment(python)
    env.update(MAX_JOBS="2", CMAKE_BUILD_PARALLEL_LEVEL="2")
    # Freeze inherited CUDA packages: pip must not replace Kaggle's matched
    # torch/torchvision pair while resolving accelerate/timm dependencies.
    constraints = weights / "cuda-constraints.txt"
    versions = "import importlib.metadata as m; print('torch==' + m.version('torch')); print('torchvision==' + m.version('torchvision'))"
    result = run(python, "-c", versions, env=env, stdout=-1, text=True)
    constraints.write_text(result.stdout, encoding="utf-8")
    run(python, "-m", "pip", "install", "--upgrade", "-c", constraints, *REQUIREMENTS, env=env)
    # No torchmcubes build: Hunyuan's default MC extractor uses scikit-image.
    rasterizer = repository / "hy3dgen/texgen/custom_rasterizer"
    setup = rasterizer / "setup.py"
    source = setup.read_text(encoding="utf-8")
    setup.write_text(patch_rasterizer(source), encoding="utf-8")
    run(python, "-m", "pip", "install", "--no-build-isolation", "--no-deps", rasterizer, env=env)
    # Relative .mesh_processor import requires building this extension in place.
    renderer = repository / "hy3dgen/texgen/differentiable_renderer"
    run(python, "setup.py", "build_ext", "--inplace", cwd=renderer, env=env)
    complete_downloads(python, repository, weights)
    print("SAM/Hunyuan3D installed. Real reconstruction has not run yet. Continue with steps 3 and 4.", flush=True)


def complete_downloads(python, repository, weights):
    if not Path(python).is_file() or not (repository / "hy3dgen").is_dir():
        raise RuntimeError("No installed model environment was found. Run the complete setup cell first.")
    env = install_environment(python)
    env.update(HF_HUB_DOWNLOAD_TIMEOUT="120", HF_HUB_ETAG_TIMEOUT="30")
    run(python, "-u", ROOT / "tools/download_quality_weights.py", "--repository", repository,
        "--weights", weights, env=env)


if __name__ == "__main__":
    main()
