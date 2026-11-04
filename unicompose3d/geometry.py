"""Geometry in a single Y-up, centered, longest-extent=1 coordinate system.

Camera matrices and Euler XYZ conventions match Three.js exactly.
"""
from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from functools import lru_cache

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .schemas import Transform
from .mesh_analysis import geometry_profile


def load_scene(path: Path):
    # No external file or network references in uploaded GLTF documents.
    value = {}
    if path.suffix.lower() == ".gltf":
        value = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".glb":
        with path.open("rb") as handle:
            header = handle.read(20)
            if len(header) < 20 or header[:4] != b"glTF":
                raise ValueError("Invalid GLB file.")
            length, chunk_type = struct.unpack_from("<II", header, 12)
            if chunk_type != 0x4E4F534A or length > 30 * 1024 * 1024:
                raise ValueError("Invalid GLB JSON chunk.")
            value = json.loads(handle.read(length).decode("utf-8"))
    for item in value.get("buffers", []) + value.get("images", []):
        uri = item.get("uri", "")
        if uri and not uri.startswith("data:"):
            raise ValueError("GLTF/GLB must embed buffers and textures. Export a self-contained GLB.")
    with path.open("rb") as handle:
        scene = trimesh.load(handle, file_type=path.suffix[1:].lower(), force="scene", process=False)
    if not isinstance(scene, trimesh.Scene) or not scene.geometry:
        raise ValueError("The model has no valid mesh.")
    meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    if not meshes or sum(len(m.faces) for m in meshes) > 2_000_000:
        raise ValueError("The mesh is empty or exceeds two million triangles. Reduce its face count before importing.")
    if any(not np.isfinite(m.vertices).all() for m in meshes):
        raise ValueError("The model contains non-finite coordinates.")
    bounds = scene.bounds
    if bounds is None or not np.isfinite(bounds).all() or np.max(bounds[1] - bounds[0]) < 1e-9:
        raise ValueError("The model has degenerate dimensions.")
    return scene


def normalize(input_path: Path, output_path: Path):
    scene = load_scene(input_path)
    bounds = scene.bounds.copy()
    longest = float(np.max(bounds[1] - bounds[0]))
    matrix = np.eye(4)
    matrix[:3, :3] /= longest
    matrix[:3, 3] = -(bounds[0] + bounds[1]) / (2 * longest)
    scene.apply_transform(matrix)
    output_path.write_bytes(scene.export(file_type="glb"))
    return describe(scene)


def describe(scene):
    meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    return {
        "geometry_profile": geometry_profile(scene),
        "bounds": scene.bounds.tolist(),
        "extents": scene.extents.tolist(),
        "vertices": sum(len(g.vertices) for g in meshes),
        "faces": sum(len(g.faces) for g in meshes),
        "has_texture": any(g.visual.kind == "texture" for g in meshes),
    }


def transform_matrix(transform: Transform):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("XYZ", transform.rotation).as_matrix() @ np.diag(transform.scale)
    result[:3, 3] = transform.position
    return result


def flat_mesh(path):
    path = Path(path).resolve()
    info = path.stat()
    return _flat_mesh(str(path), info.st_mtime_ns, info.st_size).copy()


@lru_cache(maxsize=2)
def _flat_mesh(path, modified, size):
    scene = load_scene(Path(path))
    meshes = []
    for node in scene.graph.nodes_geometry:
        matrix, name = scene.graph[node]
        geometry = scene.geometry[name]
        if isinstance(geometry, trimesh.Trimesh):
            mesh = geometry.copy()
            mesh.apply_transform(matrix)
            meshes.append(mesh)
    result = trimesh.util.concatenate(meshes)
    # This cache is for geometry queries, not rendering; release texture images.
    result.visual = trimesh.visual.ColorVisuals(mesh=result)
    return result


def ray_hit_bruteforce(mesh, origin, direction):
    """Nearest triangle intersection, vectorized; no optional rtree dependency."""
    tri = mesh.triangles
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    p = np.cross(np.broadcast_to(direction, e2.shape), e2)
    determinant = np.einsum("ij,ij->i", e1, p)
    valid = np.abs(determinant) > 1e-10
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1 / determinant[valid]
    t = origin - tri[:, 0]
    u = np.einsum("ij,ij->i", t, p) * inverse
    q = np.cross(t, e1)
    v = q @ direction * inverse
    distance = np.einsum("ij,ij->i", e2, q) * inverse
    valid &= (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7) & (distance > 1e-7)
    if not valid.any():
        return None
    index = int(np.argmin(np.where(valid, distance, np.inf)))
    normal = np.cross(e1[index], e2[index])
    normal /= max(np.linalg.norm(normal), 1e-12)
    return origin + direction * distance[index], normal


def ray_hit(mesh, origin, direction):
    locations, rays, faces = mesh.ray.intersects_location(
        np.asarray(origin).reshape(1,3), np.asarray(direction).reshape(1,3),
        multiple_hits=True)
    if not len(locations):
        return None
    valid = ((locations - origin) @ direction) > 1e-7
    if not valid.any():
        return None
    indices = np.flatnonzero(valid)
    index = indices[np.argmin(np.linalg.norm(locations[indices] - origin, axis=1))]
    return locations[index], mesh.face_normals[faces[index]].copy()


def camera_anchor(mesh, capture, uv):
    projection = np.asarray(capture.projection).reshape(4, 4, order="F")
    world = np.asarray(capture.world).reshape(4, 4, order="F")
    view_point = np.linalg.inv(projection) @ [uv[0] * 2 - 1, 1 - uv[1] * 2, -1, 1]
    view_point /= view_point[3]
    point = (world @ view_point)[:3]
    origin = world[:3, 3]
    direction = point - origin
    direction /= max(np.linalg.norm(direction), 1e-12)
    return ray_hit(mesh, origin, direction)


def anchor_point(mesh, fraction):
    return mesh.bounds[0] + np.asarray(fraction) * mesh.extents


def choose_anchor(mesh, fraction, capture, uv):
    if capture is not None and uv is not None:
        hit = camera_anchor(mesh, capture, uv)
        if hit is not None:
            return hit[0], True
    return anchor_point(mesh, fraction), False


def surface_frame(mesh, point):
    """Nearest real surface point and normal, without optional spatial indexes."""
    nearest, _, faces = trimesh.proximity.closest_point(mesh, np.asarray(point).reshape(1, 3))
    closest, index = nearest[0], int(faces[0])
    normal = mesh.face_normals[index].copy()
    if mesh.is_watertight and mesh.volume < 0:
        normal *= -1
    return closest, normal


def solve_placement(base_path, product_path, plan, captures, base_transform):
    base, product = flat_mesh(base_path), flat_mesh(product_path)
    def capture(index, slot):
        return captures[index] if 0 <= index < len(captures) and captures[index].slot == slot else None
    a, hit_a = choose_anchor(base, plan.base_anchor, capture(plan.base_view, "A"), plan.base_uv)
    b, hit_b = choose_anchor(product, plan.product_anchor, capture(plan.product_view, "B"), plan.product_uv)
    a_matrix = transform_matrix(base_transform)
    base_size = float(np.max(base.extents * np.asarray(base_transform.scale)))
    product_size = float(np.max(product.extents))
    scale = float(plan.relative_scale * base_size / max(product_size, 1e-12))
    rotation = Rotation.from_euler("XYZ", base_transform.rotation) * Rotation.from_euler("XYZ", plan.rotation_degrees, degrees=True)
    contact_normals = {}
    if plan.relation in {"on", "attached"}:
        a, normal_a = surface_frame(base, a)
        b, normal_b = surface_frame(product, b)
        world_normal = np.linalg.inv(a_matrix[:3, :3]).T @ normal_a
        world_normal /= np.linalg.norm(world_normal)
        correction, _ = Rotation.align_vectors([-world_normal], [rotation.apply(normal_b)])
        rotation = correction * rotation
        contact_normals = {"base_normal_world": world_normal.tolist(), "product_normal_local": normal_b.tolist()}
    product_matrix = rotation.as_matrix() * scale
    target = (a_matrix @ np.append(a, 1))[:3]
    if contact_normals:
        target += np.asarray(contact_normals["base_normal_world"]) * max(0, plan.clearance) * base_size
    position = target - product_matrix @ b
    transform = Transform(position=position.tolist(), rotation=rotation.as_euler("XYZ").tolist(), scale=[scale] * 3)
    diagnostics = {"raycast_base": hit_a, "raycast_product": hit_b, "anchor_base": a.tolist(), "anchor_product": b.tolist(), "anchor_target_world": target.tolist(), "rotation_world": transform.rotation, **contact_normals}
    # Correct gross bottom-plane penetration for placing on a top surface only.
    # Around/inside must permit intentional overlap, e.g. a hat around the head.
    if plan.relation == "on" and plan.base_anchor[1] > 0.8:
        vertices = trimesh.transform_points(product.vertices, transform_matrix(transform))
        base_vertices = trimesh.transform_points(base.vertices, a_matrix)
        up = a_matrix[:3, 1] / np.linalg.norm(a_matrix[:3, 1])
        floor = float((base_vertices @ up).max()) + plan.clearance * base_size
        correction = max(0.0, floor - float((vertices @ up).min()))
        transform.position = (np.asarray(transform.position) + up * correction).tolist()
        diagnostics["surface_clearance_correction"] = correction
    return transform, diagnostics


def align_anchors(base_point, product_point, base_transform, product_transform):
    target = (transform_matrix(base_transform) @ np.append(base_point, 1))[:3]
    matrix = transform_matrix(product_transform)
    position = target - matrix[:3, :3] @ np.asarray(product_point)
    return product_transform.model_copy(update={"position": position.tolist()})


def export_scene(base_path, product_path, base_transform, product_transform):
    result = trimesh.Scene()
    for label, path, transform in (("A", base_path, base_transform), ("B", product_path, product_transform)):
        scene = load_scene(path)
        outer = transform_matrix(transform)
        for i, node in enumerate(scene.graph.nodes_geometry):
            matrix, name = scene.graph[node]
            geometry = scene.geometry[name]
            result.add_geometry(geometry.copy(), node_name=f"{label}_{i}", geom_name=f"{label}_{i}", transform=outer @ matrix)
    return result.export(file_type="glb")


def deform_mesh(input_path, output_path, point, delta, radius):
    """Smooth compact-support vertex deformation, preserving topology and UVs."""
    scene = load_scene(input_path)
    for node in scene.graph.nodes_geometry:
        matrix, name = scene.graph[node]
        # Unique geometry for each instance so shared mesh instances stay correct.
        mesh = scene.geometry[name].copy()
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        vertices = trimesh.transform_points(mesh.vertices, matrix)
        distance = np.linalg.norm(vertices - np.asarray(point), axis=1) / radius
        t = np.clip(1 - distance, 0, 1)
        weight = t * t * (3 - 2 * t)
        vertices += weight[:, None] * np.asarray(delta)
        mesh.vertices = trimesh.transform_points(vertices, np.linalg.inv(matrix))
        unique = f"deformed_{node}"
        scene.geometry[unique] = mesh
        scene.graph.update(frame_to=node, matrix=matrix, geometry=unique)
    # Remove unused geometries so exports and statistics describe actual nodes.
    used = {scene.graph[n][1] for n in scene.graph.nodes_geometry}
    for name in list(scene.geometry):
        if name not in used:
            scene.geometry.pop(name)
    output_path.write_bytes(scene.export(file_type="glb"))
    return describe(scene)


def create_demo(directory):
    table = trimesh.Scene()
    surface = trimesh.creation.box([1, 0.07, 0.65])
    surface.visual.vertex_colors = [164, 111, 65, 255]
    table.add_geometry(surface, transform=trimesh.transformations.translation_matrix([0, 0.35, 0]))
    for x in (-0.41, 0.41):
        for z in (-0.24, 0.24):
            leg = trimesh.creation.box([0.055, 0.65, 0.055])
            leg.visual.vertex_colors = [49, 53, 60, 255]
            table.add_geometry(leg, transform=trimesh.transformations.translation_matrix([x, 0, z]))
    # Revolve is Z-up; rotate to the application's Y-up convention.
    profile = np.array([[0.00, 0.0], [0.11, 0.0], [0.16, 0.08], [0.17, 0.22], [0.08, 0.35], [0.07, 0.43], [0.09, 0.45], [0.07, 0.45], [0.05, 0.42], [0.06, 0.35], [0.15, 0.21], [0.14, 0.08], [0.09, 0.02], [0, 0.02]])
    vase = trimesh.creation.revolve(profile, sections=48)
    vase.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    vase.visual.vertex_colors = [47, 150, 145, 255]
    a, b = directory / "table.glb", directory / "vase.glb"
    a.write_bytes(table.export(file_type="glb"))
    b.write_bytes(vase.export(file_type="glb"))
    return a, b


def surface_patch_frame(mesh,point,radius):
 """Area-weighted local normal on the same-facing measured surface sheet."""
 from scipy.spatial import cKDTree
 center,reference=surface_frame(mesh,point)
 tree=mesh._cache["spatial_patch_tree"]
 if tree is None:
  tree=cKDTree(np.asarray(mesh.triangles_center))
  mesh._cache["spatial_patch_tree"]=tree
 ids=np.asarray(tree.query_ball_point(center,radius),dtype=int)
 if len(ids)<3:return center,reference
 normals=np.asarray(mesh.face_normals)[ids]
 same=(normals@reference)>.2
 if not np.any(same):return center,reference
 ids=ids[same];normals=normals[same]
 distance=np.linalg.norm(np.asarray(mesh.triangles_center)[ids]-center,axis=1)
 weights=np.asarray(mesh.area_faces)[ids]*np.exp(-2*(distance/max(radius,1e-8))**2)
 normal=np.sum(normals*weights[:,None],axis=0)
 return center,normal/np.linalg.norm(normal) if np.linalg.norm(normal)>1e-10 else reference
