import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from unicompose3d.weight_storage import selected_files, HEAVY_FILES, shape_file
from unicompose3d.quality import QUALITY_SETTINGS

ROOT=Path(__file__).resolve().parents[1]
with patch.object(sys,"path",[str(ROOT/"tools"),*sys.path]):
    import setup_quality
    spec=importlib.util.spec_from_file_location("turbo_worker",ROOT/"tools/quality_worker.py")
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

class TurboProfileTests(unittest.TestCase):
    def test_downloads_turbo_only_and_avoids_delight_and_duplicate_unet(self):
        for multiview in (False,True):
            specs=setup_quality.download_spec(Path("/weights"),multiview)
            patterns=[v for spec in specs for v in spec[3]]
            self.assertFalse(any("delight" in p for p in patterns))
            self.assertTrue(any("-turbo/model.fp16.safetensors" in p for p in patterns))
            selected=selected_files(multiview)
            self.assertTrue(all("delight" not in p for p in selected))
            self.assertLess(sum(HEAVY_FILES[p][0] for p in selected),12*1024**3)
            self.assertIn(shape_file(multiview), selected)
        self.assertIn("turbo5", QUALITY_SETTINGS)

    def test_transparent_holes_never_leak_original_rgb_into_paint(self):
        image=Image.new("RGBA",(4,4),(255,0,0,255))
        image.putpixel((1,1),(0,255,0,0))
        result=worker.paint_input(image)
        self.assertEqual(result.getpixel((1,1)),(255,255,255))
        self.assertEqual(result.getpixel((0,0)),(255,0,0))
        self.assertEqual(image.getpixel((1,1)),(0,255,0,0))

if __name__=="__main__":unittest.main()
