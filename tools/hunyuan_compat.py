"""Narrow compatibility fix for the pinned Hunyuan hierarchical volume decoder."""
import ast
from pathlib import Path

MARKER = "# TRYON_HIERARCHICAL_FLOAT_COORDINATES_V1"

def patch_source(source):
    # PyTorch >=2.9 no longer treats a list of slices as tuple indexing.
    source = source.replace("padded[slice_dims]", "padded[tuple(slice_dims)]")
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HierarchicalVolumeDecoding"]
    if len(classes) != 1:
        raise RuntimeError("Unknown Hunyuan hierarchical decoder; refusing to patch.")
    node = classes[0]
    lines = source.splitlines(keepends=True)
    body = "".join(lines[node.lineno - 1:node.end_lineno])
    old = "next_points = torch.stack(nidx, dim=1)"
    new = old + ".to(torch.float32)  " + MARKER
    if MARKER in body:
        if new not in body:
            raise RuntimeError("Incomplete hierarchical-coordinate patch.")
        return source
    if body.count(old) != 1 or "dtype=next_points.dtype" not in body:
        raise RuntimeError("Unexpected hierarchical coordinate mapping; check the pinned code.")
    body = body.replace(old, new)
    result = "".join(lines[:node.lineno - 1]) + body + "".join(lines[node.end_lineno:])
    ast.parse(result)
    return result

def patch_repository(repository):
    path = Path(repository) / "hy3dgen/shapegen/models/autoencoders/volume_decoders.py"
    source = path.read_text(encoding="utf-8")
    updated = patch_source(source)
    if updated != source:
        path.write_text(updated, encoding="utf-8")
    return path


def probe_decoder():
    """Small real-CUDA analytic field; no pretrained inference or quality claim."""
    import torch
    from hy3dgen.shapegen.models.autoencoders.volume_decoders import HierarchicalVolumeDecoding
    def sphere(queries, latents):
        return queries.float().square().sum(-1, keepdim=True).sqrt().to(queries.dtype) - .6
    with torch.inference_mode():
        volume = HierarchicalVolumeDecoding()(torch.zeros(1, 8, 1, device="cuda", dtype=torch.float16),
            sphere, num_chunks=1000, octree_resolution=16, min_resolution=8)
    if not bool((volume < 0).any()) or not bool((volume > 0).any()):
        raise RuntimeError("CUDA decoder coordinate probe failed; weights were not downloaded.")
    import trimesh
    vertices, faces = finite_marching_cubes(volume[0].float().cpu().numpy(), 0.)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    if not mesh.is_watertight or mesh.euler_number != 2:
        raise RuntimeError("CUDA decoder sphere surface probe failed; weights were not downloaded.")
    torch.cuda.synchronize()
    del volume
    torch.cuda.empty_cache()
    print("Real CUDA hierarchical decoder probe OK (analytic sphere, not learned reconstruction).", flush=True)

def finite_marching_cubes(grid, level):
    """Extract only cells with finite neighbours; never turn unknown space into surfaces."""
    import numpy as np
    from scipy.ndimage import binary_erosion
    from skimage.measure import marching_cubes
    grid = np.asarray(grid, dtype=np.float32)
    finite = np.isfinite(grid)
    if finite.all():
        mask = None
    else:
        # Conservative 3x3x3 neighbourhood avoids relying on skimage's cell
        # anchor convention. Unknown values remain excluded, not imputed geometry.
        mask = binary_erosion(finite, structure=np.ones((3, 3, 3), dtype=bool))
        if not mask.any():
            raise RuntimeError("No finite surface neighbourhood in decoded field.")
        grid = np.where(finite, grid, np.float32(level + 1))
    vertices, faces, _, _ = marching_cubes(grid, level, method="lewiner", mask=mask)
    if not np.isfinite(vertices).all() or not len(faces):
        raise RuntimeError("Decoded surface contains invalid vertices or no faces.")
    return vertices, faces


def install_surface_extractor(vae):
    from hy3dgen.shapegen.models.autoencoders.surface_extractors import MCSurfaceExtractor
    class FiniteMCSurfaceExtractor(MCSurfaceExtractor):
        def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
            vertices, faces = finite_marching_cubes(grid_logit.float().cpu().numpy(), mc_level)
            grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
            # Grid has resolution+1 samples but resolution intervals.
            vertices = vertices / (grid_logit.shape[0] - 1) * bbox_size + bbox_min
            return vertices, faces
    vae.surface_extractor = FiniteMCSurfaceExtractor()

def patch_renderer(repository):
    """Keep the pinned normal computation on GPU without changing mesh faces."""
    import re
    path = Path(repository) / "hy3dgen/texgen/differentiable_renderer/mesh_render.py"
    source = path.read_text(encoding="utf-8")
    marker = "# TRYON_GPU_MEAN_NORMALS_V1"
    if marker in source:
        if source.count(marker) != 2:
            raise RuntimeError("Incomplete GPU normal patch.")
        return path
    pattern = (r"vertex_normals = trimesh\.geometry\.mean_vertex_normals\(.*?"
               r"vertex_normals = torch\.from_numpy\(\s*vertex_normals\)"
               r"\.float\(\)\.to\(self\.device\)\.contiguous\(\)")
    replacement = ("from quality_progress import gpu_vertex_normals\n        "
                   "vertex_normals = gpu_vertex_normals(self.vtx_pos, self.pos_idx, face_normals)  " + marker)
    updated, count = re.subn(pattern, lambda match: replacement, source, flags=re.S)
    if count != 2:
        raise RuntimeError("Unknown Hunyuan renderer normal calculation; refusing to patch.")
    ast.parse(updated)
    path.write_text(updated,encoding="utf-8")
    return path
