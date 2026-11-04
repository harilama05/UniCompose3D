"""Regression: integer index dtype must not quantize the voxel spacing to zero."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from hunyuan_compat import patch_source

SOURCE = """# Preserve upstream attribution
class HierarchicalVolumeDecoding:
    def map(self, nidx, resolution, bbox_min, device):
        next_points = torch.stack(nidx, dim=1)
        next_points = (next_points * torch.tensor(resolution, dtype=next_points.dtype, device=device) +
                       torch.tensor(bbox_min, dtype=next_points.dtype, device=device))
        return next_points
"""

class HierarchicalCoordinateTests(unittest.TestCase):
    def test_fractional_coordinates_cover_the_volume_instead_of_collapsing(self):
        nidx=tuple(torch.tensor([0,8,16]) for _ in range(3))
        ns={"torch":torch};exec(SOURCE,ns)
        before=ns["HierarchicalVolumeDecoding"]().map(nidx,[2.02/16]*3,[-1.01]*3,"cpu")
        self.assertEqual(len(torch.unique(before,dim=0)),1)
        exec(patch_source(SOURCE),ns)
        after=ns["HierarchicalVolumeDecoding"]().map(nidx,[2.02/16]*3,[-1.01]*3,"cpu")
        expected=torch.tensor([[-1.01]*3,[0.]*3,[1.01]*3])
        torch.testing.assert_close(after,expected,atol=1e-6,rtol=1e-6)

    def test_patch_is_idempotent_and_does_not_touch_other_classes(self):
        other="\nclass Other:\n    pass\n"
        fixed=patch_source(SOURCE+other)
        self.assertEqual(patch_source(fixed),fixed)
        self.assertTrue(fixed.startswith("# Preserve upstream attribution"))
        self.assertTrue(fixed.endswith(other))

    def test_unknown_source_fails_closed(self):
        with self.assertRaises(RuntimeError):patch_source("class Other: pass")

if __name__=="__main__":unittest.main()
