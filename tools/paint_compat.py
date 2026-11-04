"""Adapt Hunyuan's custom UNet loader, which bypasses Diffusers weight selection."""
from __future__ import annotations

import ast
from pathlib import Path

MARKER = "# TRYON_PAINT_SAFETENSORS_V1"
MODULE = "single/hunyuan3d-paint-v2-0-turbo/unet/modules.py"


def patch_loader(source):
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UNet2p5DConditionModel"]
    methods = [node for cls in classes for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "from_pretrained"]
    if len(methods) != 1:
        raise RuntimeError("The expected Paint UNet loader was not found; no changes made.")
    method = methods[0]
    if MARKER in source:
        if not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_file" for node in ast.walk(method)):
            raise RuntimeError("The Paint patch is incomplete; inspect the file.")
        return source
    assignments = [node for node in ast.walk(method) if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "unet_ckpt"
                   and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "torch.load"
                   and node.value.args and ast.unparse(node.value.args[0]) == "unet_ckpt_path"]
    if len(assignments) != 1:
        raise RuntimeError("The UNet loader has an unexpected structure; no changes made.")
    node = assignments[0]
    lines = source.splitlines(keepends=True)
    indent = " " * node.col_offset
    replacement = [MARKER,
                   "safe_ckpt_path = os.path.splitext(unet_ckpt_path)[0] + '.safetensors'",
                   "if os.path.isfile(safe_ckpt_path):",
                   "    from safetensors.torch import load_file",
                   "    unet_ckpt = load_file(safe_ckpt_path, device='cpu')",
                   "    expected = unet.state_dict()",
                   "    unused_image_projection = {'unet.image_proj_model.norm.bias': (1024,), 'unet.image_proj_model.norm.weight': (1024,), 'unet.image_proj_model.proj.bias': (4096,), 'unet.image_proj_model.proj.weight': (4096, 1024)}",
                   "    for key in tuple(unet_ckpt):",
                   "        if key not in expected and key in unused_image_projection and tuple(unet_ckpt[key].shape) == unused_image_projection[key]:",
                   "            del unet_ckpt[key]  # Inactive IP-Adapter image projection in the official checkpoint.",
                   "            continue",
                   "        if key not in expected and key.startswith('unet.') and key.endswith(('.attn2.to_k_ip.weight', '.attn2.to_v_ip.weight')):",
                   "            base = key.replace('.to_k_ip.', '.to_k.').replace('.to_v_ip.', '.to_v.')",
                   "            if base in expected and unet_ckpt[key].shape == expected[base].shape:",
                   "                del unet_ckpt[key]  # Unused IP-Adapter projection, absent in this architecture.",
                   "else:",
                   "    " + ast.get_source_segment(source, node)]
    result = "".join(lines[:node.lineno - 1]) + "".join(indent + line + "\n" for line in replacement) + "".join(lines[node.end_lineno:])
    ast.parse(result)
    return result


def patch_paint_weights(weights):
    root = Path(weights).resolve()
    path = root / MODULE
    if not path.resolve().is_relative_to(root):
        raise ValueError("The Paint loader is outside the weights directory; refusing to modify it.")
    safe = path.with_name("diffusion_pytorch_model.safetensors")
    if not path.is_file() or not safe.is_file() or safe.stat().st_size == 0:
        raise RuntimeError("Paint loader or .safetensors weights are missing; resume the model download.")
    source = path.read_text(encoding="utf-8")
    result = patch_loader(source)
    if result != source:
        path.write_text(result, encoding="utf-8")
        print("Patched Paint to load .safetensors with strict=True; no duplicate .bin download.", flush=True)
    return path
