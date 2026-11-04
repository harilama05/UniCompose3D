"""Category-independent geometry observations; never a reconstruction verdict."""
from __future__ import annotations
import numpy as np
import trimesh

PROFILE_VERSION = "principal-spread-v1"
SAMPLE_LIMIT = 12000


def geometry_profile(scene):
    nodes = []
    for node in scene.graph.nodes_geometry:
        matrix, name = scene.graph[node]
        mesh = scene.geometry[name]
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices):
            nodes.append((matrix, mesh))
    total = sum(len(mesh.vertices) for _, mesh in nodes)
    if total == 0:
        raise ValueError("The mesh has no vertices for shape analysis.")
    selected = np.linspace(0, total - 1, min(total, SAMPLE_LIMIT), dtype=np.int64)
    samples = []
    offset = 0
    for matrix, mesh in nodes:
        local = selected[(selected >= offset) & (selected < offset + len(mesh.vertices))] - offset
        if len(local):
            samples.append(trimesh.transform_points(mesh.vertices[local], matrix))
        offset += len(mesh.vertices)
    points = np.concatenate(samples)
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(points) - 1)
    values, axes = np.linalg.eigh(covariance)
    values = np.maximum(values, 0)
    ratio = float(np.sqrt(values[0] / max(values[-1], 1e-20)))
    spans = np.ptp(centered @ axes, axis=0)
    thin = ratio < .035
    observations = []
    if thin:
        observations.append("The mesh is very thin along one axis. If this differs from the object, reconstruct it again; composition cannot create missing volume or openings.")
    return {"version": PROFILE_VERSION, "sample_count": len(points),
            "principal_spans": spans.tolist(), "thin_ratio": ratio,
            "thin_threshold": .035, "thin_geometry": thin,
            "diagnostic_only": True, "observations": observations}
