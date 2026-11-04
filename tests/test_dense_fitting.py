"""Regression for real reconstruction-sized meshes and small attachment targets."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import trimesh
from unicompose3d.collision import contains, penetration_fraction
from unicompose3d.constraint_solver import ProgramSolver
from unicompose3d.schemas import Transform
from test_spatial_program import proposal


class DenseFittingTests(unittest.TestCase):
    def test_mesh_above_previous_face_limit_is_checked_without_decimation(self):
        sphere = trimesh.creation.icosphere(subdivisions=7)
        self.assertGreater(len(sphere.faces), 120000)
        np.testing.assert_array_equal(contains(sphere, [[0,0,0], [2,0,0], [.8,0,0]]), [True,False,True])
        self.assertEqual(len(sphere.faces), 327680)

    def test_all_deep_samples_count_and_hollow_center_is_not_solid(self):
        box = trimesh.creation.box()
        points = np.random.default_rng(42).uniform(-.2,.2,(128,3))
        self.assertEqual(penetration_fraction(box, points, .003), 1.)
        torus = trimesh.creation.torus(major_radius=.3,minor_radius=.05)
        np.testing.assert_array_equal(contains(torus, [[0,0,0],[.3,0,0],[0,0,.3]]), [False,True,False])

    def test_small_aperture_is_not_allowed_to_drift_by_whole_base_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"a.glb",Path(tmp)/"b.glb"
            trimesh.creation.cylinder(radius=.02,height=1,sections=64).export(a)
            trimesh.creation.torus(major_radius=.3,minor_radius=.05).export(b)
            program={"features":[{"id":"a","slot":"A","kind":"section","point":[1,.5,.5],"end":[1,.5,.8]},
                {"id":"b","slot":"B","kind":"aperture"}],
                "constraints":[{"kind":"coincident","a":"a","b":"b"},
                    {"kind":"align","a":"a","b":"b"},
                    {"kind":"clearance","a":"a","b":"b","target":1.05}]}
            solver=ProgramSolver(a,b,Transform(),proposal(program),[])
            seed=solver.seed()
            self.assertTrue(solver.check(seed)["eligible"])
            drift=seed.model_copy(deep=True); drift.position[0] += .012
            check=solver.check(drift)
            self.assertFalse(next(c for c in check["metrics"]["constraint_checks"] if c["kind"]=="clearance")["passed"])
            fit,_=solver.optimize(drift)
            self.assertTrue(solver.check(fit)["eligible"])

if __name__=="__main__": unittest.main()
