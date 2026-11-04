"""CPU spatial indexing on original triangles: no face-count bypass or mesh shrink."""
import numpy as np
import trimesh


def contains(mesh, points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    result = np.zeros(len(points), dtype=bool)
    if not len(points):
        return result
    # Broad phase excludes rays starting outside the mesh's bounds.
    bounded = np.all((points >= mesh.bounds[0]) & (points <= mesh.bounds[1]), axis=1)
    indices = np.flatnonzero(bounded)
    direction = np.array([1., .000173, .000317])
    direction /= np.linalg.norm(direction)
    for start in range(0, len(indices), 32):
        selected = indices[start:start + 32]
        # intersects_location deduplicates coincident hits on shared triangle edges.
        _, rays, _ = mesh.ray.intersects_location(points[selected],
            np.tile(direction, (len(selected), 1)), multiple_hits=True)
        result[selected] = np.bincount(rays, minlength=len(selected)) % 2 == 1
    return result


def distances(mesh, points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points):
        return np.empty(0)
    batches = []
    for start in range(0, len(points), 32):
        _, distance, _ = trimesh.proximity.closest_point(mesh, points[start:start + 32])
        batches.append(distance)
    return np.concatenate(batches)


def penetration_fraction(mesh, points, tolerance):
    inside = contains(mesh, points)
    # Count ALL inside samples. Capping at 64 used to undercount deep collisions.
    depths = distances(mesh, np.asarray(points)[inside])
    return float(np.count_nonzero(depths > tolerance) / max(1, len(points)))
