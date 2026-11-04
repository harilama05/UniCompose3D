"""UV layout may change; original triangle coordinates must not."""
import importlib.util,sys,tempfile,unittest
from pathlib import Path
import numpy as np,trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from uv_atlas import unwrap,rectangles
@unittest.skipUnless(importlib.util.find_spec("xatlas"),"xatlas is installed in the model runtime")
class AtlasTests(unittest.TestCase):
    def test_exact_geometry_survives_atlas_and_cache_reuse(self):
        mesh=trimesh.creation.icosphere(subdivisions=1)
        with tempfile.TemporaryDirectory() as tmp:
            output,report=unwrap(mesh,tmp)
            np.testing.assert_array_equal(output.triangles,mesh.triangles)
            self.assertEqual(len(output.faces),len(mesh.faces))
            self.assertTrue(np.isfinite(output.visual.uv).all())
            cached,again=unwrap(mesh,tmp)
            self.assertTrue(again["cache_reused"])
            np.testing.assert_array_equal(cached.triangles,output.triangles)
    def test_packed_regions_are_disjoint_and_cover_the_atlas(self):
        boxes=rectangles([1,2,3,4])
        self.assertAlmostEqual(sum(w*h for x,y,w,h in boxes.values()),1.)
        for i,a in boxes.items():
            for j,b in boxes.items():
                if i>=j:continue
                x,y,w,h=a;u,v,s,t=b
                self.assertLessEqual(min(x+w,u+s)-max(x,u),1e-12) if min(y+h,v+t)>max(y,v)+1e-12 else None
if __name__=="__main__":unittest.main()
