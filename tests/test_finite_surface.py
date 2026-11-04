"""Unknown voxels must not create NaN vertices or close a genuine opening."""
import sys
import unittest
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hunyuan_compat import finite_marching_cubes

class FiniteSurfaceTests(unittest.TestCase):
    def test_sparse_sphere_and_torus_preserve_topology(self):
        axis = np.linspace(-1.01, 1.01, 65, dtype=np.float32)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        fields = [(np.sqrt(x*x+y*y+z*z)-.6, 2),
                  (np.sqrt((np.sqrt(x*x+y*y)-.5)**2+z*z)-.15, 0)]
        for field, euler in fields:
            with self.subTest(euler=euler):
                field[np.abs(field) > .12] = np.nan
                vertices, faces = finite_marching_cubes(field, 0)
                mesh = trimesh.Trimesh(vertices, faces, process=False)
                self.assertTrue(np.isfinite(vertices).all())
                self.assertTrue(mesh.is_watertight)
                self.assertEqual(mesh.euler_number, euler)

    def test_unknown_field_fails_instead_of_inventing_geometry(self):
        with self.assertRaisesRegex(RuntimeError, "finite"):
            finite_marching_cubes(np.full((8,8,8),np.nan), 0)

if __name__ == "__main__":
    unittest.main()
