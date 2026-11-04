"""Thinness is observed, not rejected; coordinates and object category are unchanged."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from unicompose3d import geometry
from unicompose3d.api import Engine
from unicompose3d.mesh_analysis import geometry_profile
from unicompose3d.storage import Store

class MeshAnalysisTests(unittest.TestCase):

    def test_thinness_is_rotation_invariant_and_does_not_modify_vertices(self):
        mesh = trimesh.creation.box(extents=[1, 0.6, 0.01])
        before = mesh.vertices.copy()
        first = geometry_profile(trimesh.Scene(mesh))
        rotated = mesh.copy()
        rotated.apply_transform(trimesh.transformations.rotation_matrix(0.8, [1, 2, 3]))
        second = geometry_profile(trimesh.Scene(rotated))
        self.assertTrue(first['thin_geometry'])
        self.assertAlmostEqual(first['thin_ratio'], second['thin_ratio'], places=6)
        np.testing.assert_array_equal(mesh.vertices, before)
        self.assertTrue(first['diagnostic_only'])

    def test_solid_and_through_opening_objects_are_not_classified_by_name(self):
        for mesh in (trimesh.creation.box(), trimesh.creation.icosphere(subdivisions=1), trimesh.creation.annulus(r_min=0.25, r_max=0.4, height=0.2)):
            self.assertFalse(geometry_profile(trimesh.Scene(mesh))['thin_geometry'])

    def test_normalization_preserves_flat_object_and_existing_asset_can_be_inspected_without_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'thin.glb'
            trimesh.creation.box(extents=[1, 1, 0.01]).export(source)
            llm = Mock()
            engine = Engine(Store(root / 'data'), gemini=llm)
            self.addCleanup(engine.pool.shutdown, wait=False, cancel_futures=True)
            self.addCleanup(engine.segments.close)
            asset = engine.asset(source, 'a genuinely thin object')
            self.assertEqual(asset['status'], 'ready')
            self.assertTrue(asset['geometry_profile']['thin_geometry'])
            normalized = geometry.flat_mesh(engine.model_path(asset['id']))
            self.assertAlmostEqual(min(normalized.extents) / max(normalized.extents), 0.01, places=5)
            engine.store.update('assets', asset['id'], geometry_profile=None)
            inspected = engine.inspect_asset(asset['id'])
            self.assertTrue(inspected['geometry_profile']['thin_geometry'])
            llm.plan.assert_not_called()

    def test_scene_node_transforms_are_included(self):
        scene = trimesh.Scene()
        transform = np.eye(4)
        transform[2, 2] = 0.01
        scene.add_geometry(trimesh.creation.box(), transform=transform)
        self.assertTrue(geometry_profile(scene)['thin_geometry'])
if __name__ == '__main__':
    unittest.main()
