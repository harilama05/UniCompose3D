"""Actual mesh tests: worn bands must encircle, not rest on skin."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from unicompose3d import geometry, aperture as rings, composition
from unicompose3d.schemas import Transform, PlacementPlan


class RingTests(unittest.TestCase):
    def setUp(self):
        self.finger = trimesh.creation.cylinder(radius=0.07, height=0.8, sections=64)
        self.ring = trimesh.creation.torus(major_radius=0.3, minor_radius=0.05, major_sections=64, minor_sections=24)

    def test_hole_is_verified_and_solid_sphere_is_rejected(self):
        hole = rings.hole_frame(self.ring)
        self.assertAlmostEqual(hole["inner_radius"], 0.25, delta=0.006)
        with self.assertRaisesRegex(ValueError, "through opening"):
            rings.hole_frame(trimesh.creation.icosphere(subdivisions=2))

    def test_fitting_recenters_from_skin_and_uses_aperture_scale(self):
        transform, diagnostics = rings.fit(self.finger, self.ring, Transform(), [0.07, 0, 0], [0.07, 0, 0.3], [0, 0, 0])
        metrics, failures = rings.check_pose(transform, diagnostics)
        self.assertFalse(failures)
        np.testing.assert_allclose(transform.position, [0, 0, 0], atol=0.001)
        self.assertGreater(transform.scale[0] * 0.25, 0.07)
        self.assertGreater(metrics["aperture_axis_alignment"], 0.999)
        # Exact closed-mesh volume samples verify gold stays outside the finger.
        points, _ = composition.surface_samples(self.ring, 400)
        points = trimesh.transform_points(points, geometry.transform_matrix(transform))
        self.assertFalse(composition.contains_points(self.finger, points).any())

    def test_arbitrary_object_orientation_and_base_nonuniform_scale(self):
        rotation = Rotation.from_euler("XYZ", [0.7, -0.4, 0.3])
        self.ring.apply_transform(trimesh.transformations.rotation_matrix(0.9, [1, 1, 0]))
        base = Transform(position=[1, 2, 3], rotation=rotation.as_euler("XYZ").tolist(), scale=[1.2, 0.9, 1.1])
        transform, diagnostics = rings.fit(self.finger, self.ring, base, [0.07, 0, 0], [0.07, 0, 0.3], [0.2, 0.3, 0.4])
        _, failures = rings.check_pose(transform, diagnostics)
        self.assertFalse(failures)
        expected = rotation.apply([0, 0, 1])
        self.assertGreater(abs(np.dot(expected, diagnostics["target_axis_world"])), 0.999)

    def test_surface_placement_is_rejected_by_ring_checks(self):
        transform, diagnostics = rings.fit(self.finger, self.ring, Transform(), [0.07, 0, 0], [0.07, 0, 0.3], [0, 0, 0])
        transform.position[0] += 0.07
        _, failures = rings.check_pose(transform, diagnostics)
        self.assertTrue(any("opening center" in failure.lower() for failure in failures))

    def test_degenerate_finger_direction_fails(self):
        with self.assertRaisesRegex(ValueError, "too close"):
            rings.fit(self.finger, self.ring, Transform(), [0.07, 0, 0], [0.07, 0, 0], [0, 0, 0])

    def test_same_solver_handles_wrist_bracelet_and_pipe_collar(self):
        for radius, length in ((0.16, 1.2), (0.035, 0.5)):
            with self.subTest(target_radius=radius):
                target = trimesh.creation.cylinder(radius=radius, height=length, sections=48)
                transform, diagnostics = rings.fit(target, self.ring, Transform(), [radius, 0, 0], [radius, 0, length * 0.3], [0, 0, 0])
                _, failures = rings.check_pose(transform, diagnostics)
                self.assertFalse(failures)
                self.assertGreater(transform.scale[0] * 0.25, radius)

    def test_candidate_search_rejects_surface_baseline_and_offers_fitted_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.glb", Path(tmp) / "b.glb"
            self.finger.export(a); self.ring.export(b)
            plan = PlacementPlan(relation="around", fit_mode="aperture", base_anchor=[1, 0.5, 0.5], base_axis_end=[1, 0.5, 0.9], product_anchor=[0.5, 0.5, 0.5], rotation_degrees=[0, 90, 0], relative_scale=0.3, confidence=0.9, explanation="Proposed band")
            candidates, diagnostics = composition.build_candidates(a, b, plan, [], Transform(), 3)
            self.assertFalse(candidates[0]["eligible"])
            self.assertTrue(any(candidate["eligible"] for candidate in candidates[1:]), candidates)
            self.assertEqual(diagnostics["resolved_features"]["b"]["kind"], "aperture")
            self.assertGreater(diagnostics["resolved_features"]["b"]["radius"], 0)


if __name__ == "__main__":
    unittest.main()
