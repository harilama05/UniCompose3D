"""Bounded candidate search and relation-aware geometric checks.

Metrics are deterministic heuristics, not probabilities or a physics solver.
Surface distances and solid-intersection fractions use bounded samples. Open
meshes explicitly report that solid intersection could not be checked.
"""
from __future__ import annotations

import copy

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import geometry, aperture
from .schemas import Transform

ALGORITHM_VERSION = "spatial-program-10-grounded-multistart-intent"
LIMITATIONS = [
    "Scores are heuristic comparisons, not calibrated probabilities.",
    "Contact and intersection checks sample surfaces and may miss thin details.",
    "No cloth simulation or guarantee of correct geometry for every object.",
]


def surface_samples(mesh, count, seed=2026):
    rng = np.random.default_rng(seed)
    areas = np.asarray(mesh.area_faces)
    if float(areas.sum()) < 1e-12:
        raise ValueError("The mesh has no valid surface area for composition.")
    probabilities = areas / max(float(areas.sum()), 1e-12)
    indices = rng.choice(len(mesh.faces), size=count, p=probabilities)
    triangles = mesh.triangles[indices]
    u = np.sqrt(rng.random(count))
    v = rng.random(count)
    points = (1 - u[:, None]) * triangles[:, 0] + (u * (1 - v))[:, None] * triangles[:, 1] + (u * v)[:, None] * triangles[:, 2]
    return points, mesh.face_normals[indices]


def solid_mesh(mesh):
    result = mesh.copy()
    result.merge_vertices(merge_tex=True, merge_norm=True)
    return result, bool(result.is_watertight)


def contains_points(mesh, points):
    """Indexed ray parity on the original mesh; shared-edge hits are deduplicated."""
    from .collision import contains
    return contains(mesh, points)



class GeometryContext:
    def __init__(self, base_path, product_path, base_transform, plan, diagnostics):
        self.base = geometry.flat_mesh(base_path)
        base_local_extents = self.base.extents.copy()
        self.product = geometry.flat_mesh(product_path)
        self.base_matrix = geometry.transform_matrix(base_transform)
        self.base.apply_transform(self.base_matrix)
        self.base_solid, self.base_closed = solid_mesh(self.base)
        self.product_solid, self.product_closed = solid_mesh(self.product)
        self.base_volume_test = self.base_closed
        self.product_volume_test = self.product_closed
        self.plan = plan
        self.base_size = max(float(np.max(self.base.extents)), 1e-6)
        self.base_rotation = Rotation.from_euler("XYZ", base_transform.rotation)
        self.up = self.base_matrix[:3, 1] / np.linalg.norm(self.base_matrix[:3, 1])
        self.base_anchor = (self.base_matrix @ [*diagnostics["anchor_base"], 1])[:3]
        self.product_anchor = np.asarray(diagnostics["anchor_product"])
        self.target = self.base_anchor.copy()
        if plan.relation == "on":
            self.target += self.up * plan.clearance * self.base_size
        if "anchor_target_world" in diagnostics:
            self.target = np.asarray(diagnostics["anchor_target_world"])
        self.contact_normals = {key: diagnostics[key] for key in ("base_normal_world", "product_normal_local") if key in diagnostics}
        self.seed_rotation = Rotation.from_euler("XYZ", diagnostics["rotation_world"]) if "rotation_world" in diagnostics else self.base_rotation * Rotation.from_euler("XYZ", plan.rotation_degrees, degrees=True)
        self.base_points, self.base_normals = surface_samples(self.base, 2048)
        self.product_points, _ = surface_samples(self.product, 256, seed=37)
        self.base_tree = cKDTree(self.base_points)
        self.original_scale = plan.relative_scale * float(np.max(base_local_extents * base_transform.scale)) / max(float(np.max(self.product.extents)), 1e-12)
        self.proxy_trustworthy = self.base_closed and self.base_solid.is_winding_consistent and self.base_solid.volume > 0

    def pose(self, factor, yaw):
        twist_axis = np.asarray(self.contact_normals.get("base_normal_world", self.up))
        rotation = Rotation.from_rotvec(twist_axis * np.deg2rad(yaw)) * self.seed_rotation
        scale = self.original_scale * factor
        position = self.target - rotation.apply(self.product_anchor * scale)
        result = Transform(position=position.tolist(), rotation=rotation.as_euler("XYZ").tolist(), scale=[scale] * 3)
        if self.plan.relation == "on" and self.plan.base_anchor[1] > 0.8:
            points = trimesh.transform_points(self.product.vertices, geometry.transform_matrix(result))
            correction = max(0.0, float((self.base.vertices @ self.up).max() + self.plan.clearance * self.base_size - (points @ self.up).min()))
            result.position = (position + self.up * correction).tolist()
        return result

    def fast_values(self, transform):
        matrix = geometry.transform_matrix(transform)
        points = trimesh.transform_points(self.product_points, matrix)
        distance, nearest = self.base_tree.query(points)
        anchor = (matrix @ [*self.product_anchor, 1])[:3]
        error = np.linalg.norm(anchor - self.target) / self.base_size
        depth = np.zeros(len(points))
        if self.proxy_trustworthy:
            signed = np.einsum("ij,ij->i", points - self.base_points[nearest], self.base_normals[nearest])
            # Nearest-normal signs only guide local optimization. Final checks
            # use ray crossings, not this proxy.
            depth = np.maximum(-signed / self.base_size - 0.005, 0)
        return points, distance, float(error), depth

    def objective(self, transform):
        _, distance, error, depth = self.fast_values(transform)
        relation = self.plan.relation
        contact_weight = 2 if relation in {"inside", "beside"} else 12
        collision_weight = 25
        return float(contact_weight * error ** 2 + 0.3 * (distance.min() / self.base_size) ** 2 + collision_weight * np.mean(depth ** 2))

    def refine(self, seed):
        before = self.objective(seed)
        def offset_pose(offset):
            return seed.model_copy(update={"position": (np.asarray(seed.position) + np.asarray(offset) * self.base_size).tolist()})
        result = minimize(lambda x: self.objective(offset_pose(x)), np.zeros(3), method="Powell", bounds=[(-0.06, 0.06)] * 3, options={"maxiter": 8, "maxfev": 60, "xtol": 0.002, "ftol": 0.0001})
        candidate = offset_pose(result.x)
        after = self.objective(candidate)
        if not np.isfinite(after) or after > before:
            candidate, after = seed, before
        return candidate, {"performed": True, "objective_before": before, "objective_after": after, "evaluations": int(result.nfev), "method": "bounded Powell translation"}

    def check(self, transform):
        matrix = geometry.transform_matrix(transform)
        points, distance, anchor_error, _ = self.fast_values(transform)
        product_vertices = trimesh.transform_points(self.product.vertices, matrix)
        product_bounds = np.array([product_vertices.min(axis=0), product_vertices.max(axis=0)])
        intersection = np.minimum(self.base.bounds[1], product_bounds[1]) - np.maximum(self.base.bounds[0], product_bounds[0])
        b_fraction, a_fraction = None, None
        if np.all(intersection > 1e-6):
            if self.base_volume_test:
                inside = contains_points(self.base_solid, points[:192])
                b_fraction = float(np.mean(inside & (distance[:192] > self.base_size * 0.01)))
            if self.product_volume_test:
                local = trimesh.transform_points(self.base_points[:96], np.linalg.inv(matrix))
                inside = contains_points(self.product_solid, local)
                # Distance to product samples excludes near-contact samples.
                product_tree = cKDTree(points)
                a_distance, _ = product_tree.query(self.base_points[:96])
                a_fraction = float(np.mean(inside & (a_distance > self.base_size * 0.01)))
        else:
            b_fraction = 0.0 if self.base_closed else None
            a_fraction = 0.0 if self.product_closed else None
        solid_fractions = [v for v in (b_fraction, a_fraction) if v is not None]
        penetration = max(solid_fractions) if solid_fractions else None
        local_points = trimesh.transform_points(points, np.linalg.inv(self.base_matrix))
        # Use original local bounds, not an inverse-transformed world AABB.
        original_vertices = trimesh.transform_points(self.base.vertices, np.linalg.inv(self.base_matrix))
        lo, hi = original_vertices.min(axis=0), original_vertices.max(axis=0)
        containment = float(np.mean(np.all((local_points >= lo - 0.01) & (local_points <= hi + 0.01), axis=1)))
        relation = self.plan.relation
        failures, warnings = [], []
        if self.contact_normals:
            normal = np.linalg.inv(matrix[:3, :3]).T @ self.contact_normals["product_normal_local"]
            normal /= np.linalg.norm(normal)
            if -float(np.dot(normal, self.contact_normals["base_normal_world"])) < 0.9:
                failures.append("The contact surfaces do not face each other at the selected interface.")
        if not self.base_closed or not self.product_closed:
            warnings.append("Open meshes limit solid-intersection checks.")
        if anchor_error > (0.25 if relation == "beside" else 0.15):
            failures.append("The placement is too far from the proposed target region.")
        if relation in {"on", "attached", "hanging", "beside"} and penetration is not None and penetration > 0.35:
            failures.append("Surface samples indicate substantial unintended solid overlap.")
        if relation == "inside":
            warnings.append("Cavity containment is sampled; complete enclosure in empty space is not proven.")
            if containment < 0.65:
                failures.append("Most of the product lies outside the base enclosure.")
            if b_fraction is not None and b_fraction > 0.05:
                failures.append("B is inside solid material in A, rather than an empty cavity.")
        if relation == "around":
            if self.plan.fit_mode != "aperture":
                warnings.append("No verified opening supports this surrounding relation; manual review is required.")
            if b_fraction is not None and b_fraction > 0.08:
                failures.append("B penetrates material in A; surrounding placement does not permit arbitrary penetration.")
        if self.plan.deform:
            warnings.append("This object requires deformation; the rigid solver does not simulate cloth or contact deformation.")
        hanging_score = 1.0
        if relation == "hanging":
            above = float((product_vertices.mean(axis=0)[1] - self.base_anchor[1]) / self.base_size)
            if above > 0.15:
                failures.append("The product is above the hanging point rather than suspended below it.")
                hanging_score = 0.0
        contact = float(np.exp(-((anchor_error / 0.08) ** 2)))
        collision = 1.0 if penetration is None else float(max(0, 1 - penetration * 2))
        scale_score = float(np.exp(-abs(np.log(transform.scale[0] / self.original_scale))))
        score = 0.55 * contact + 0.3 * collision + 0.15 * scale_score
        if relation == "inside":
            score = 0.35 * contact + 0.5 * containment + 0.15 * scale_score
        score *= hanging_score
        return {"geometry_score": round(float(score), 6), "eligible": not failures, "failures": failures, "warnings": warnings,
                "metrics": {"anchor_error_relative": round(anchor_error, 6), "surface_gap_estimate_relative": round(float(distance.min() / self.base_size), 6), "product_inside_base_fraction": b_fraction, "base_inside_product_fraction": a_fraction, "penetration_fraction": penetration, "containment_bounds_fraction": containment, "base_watertight": self.base_closed, "product_watertight": self.product_closed, "base_volume_test_available": self.base_volume_test, "product_volume_test_available": self.product_volume_test, "surface_samples_product": 256, "solid_samples_product": 192, "solid_samples_base": 96}}


def build_candidates(base_path, product_path, plan, captures, base_transform, count=5, hints=None, goal_contract=None, budget=None):
    from .constraint_solver import build_candidates as solve_program
    return solve_program(base_path, product_path, plan, captures, base_transform, count, hints,goal_contract=goal_contract,budget=budget)


def rank_candidates(candidates, evaluation=None, plan_confidence=1.0, error=None, semantic_required=False):
    result = copy.deepcopy(candidates)
    reviews = {review.candidate_id: review for review in evaluation.reviews} if evaluation else {}
    if evaluation and (len(reviews) != len(evaluation.reviews) or set(reviews) != {c["id"] for c in result}):
        raise ValueError("Reviews do not match the candidate list.")
    for candidate in result:
        review = reviews.get(candidate["id"])
        if review is not None and (not candidate["eligible"] or (semantic_required and review.intent_satisfied is not True)):
            review=review.model_copy(update={"acceptable":False})
        visual = float(np.mean([review.contact_score, review.placement_score, review.proportion_score, review.appearance_score])) if review else None
        candidate.update(visual_review=review.model_dump() if review else None, visual_score=visual,
                         combined_score=round(0.6 * candidate["geometry_score"] + 0.4 * visual, 6) if visual is not None else candidate["geometry_score"])
    result.sort(key=lambda c: (-int(c["eligible"]), -int(c["visual_review"]["acceptable"] if c["visual_review"] else True), -c["combined_score"], c["id"]))
    for rank, candidate in enumerate(result, 1):
        candidate["rank"] = rank
    eligible = [c for c in result if c["eligible"]]
    accepted = [c for c in eligible if c["visual_review"] and c["visual_review"]["acceptable"]]
    selected = accepted[0] if accepted else None
    suggested = selected or (eligible[0] if eligible else result[0])
    needs_review = selected is None or selected["geometry_score"] < 0.6 or selected["visual_score"] < 0.55 or plan_confidence < 0.6 or bool(selected["warnings"])
    warnings = list(suggested["warnings"])
    if not eligible:
        warnings.append("No candidate passed geometry checks. Adjust the target or edit manually.")
    elif evaluation and not accepted:
        warnings.append("The VLM accepted no candidate. Inspect the renders or edit manually.")
    if error:
        warnings.append(error)
    if plan_confidence < 0.6:
        warnings.append("The AI is uncertain about the relationship; inspect the target region.")
    baseline = next((c for c in result if c["baseline"]), None)
    return {"candidates": result, "selected_candidate_id": selected["id"] if selected else None, "suggested_candidate_id": suggested["id"], "auto_apply": not needs_review, "needs_review": needs_review,
            "selection_method": "geometry_and_visual" if evaluation else "geometry_only", "evaluation_status": "completed" if evaluation else "unavailable", "summary": ("Geometry unresolved: " + " ".join(suggested["failures"])) if not eligible else evaluation.summary if evaluation else "The geometric proposal needs inspection in 3D. No visual review has run.", "warnings": warnings,
            "comparison": {"baseline_id": baseline["id"], "compared_candidate_id": suggested["id"], "geometry_delta": round(suggested["geometry_score"] - baseline["geometry_score"], 6), "combined_delta": round(suggested["combined_score"] - baseline["combined_score"], 6)} if baseline else None,
            "score_weights": {"geometry": 0.6, "visual": 0.4}, "limitations": LIMITATIONS}
