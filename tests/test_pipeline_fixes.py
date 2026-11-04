"""Regression tests for prompt masks, brush edits, image accounting and failed fits."""
import io
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from sam_mask import select_mask
from unicompose3d.composition import build_candidates, rank_candidates
from unicompose3d.constraint_solver import compatibility_program
from unicompose3d.schemas import Transform
import test_composition as composition_tests
import test_input_quality as input_tests


class MaskSelectionTests(unittest.TestCase):
    def test_user_points_override_high_iou_background_fill(self):
        masks = np.ones((3, 12, 12), dtype=bool)
        masks[1, 6, 6] = False
        masks[2, 6, 6] = False
        masks[2, 2, 2] = False
        selected = select_mask(masks, [.99, .70, .90], [[.5, .5], [.2, .2]], [0, 1])
        self.assertEqual(selected[6, 6], 0)
        self.assertEqual(selected[2, 2], 255)

    def test_no_candidate_honors_points_requires_correction(self):
        with self.assertRaises(ValueError):
            select_mask(np.ones((3, 4, 4)), [1, .8, .7], [[1, 1]], [0])

    def test_boundary_points_are_clamped_to_last_pixel(self):
        masks = np.ones((1, 4, 4)); masks[0, -1, -1] = 0
        self.assertEqual(select_mask(masks, [1], [[1, 1]], [0])[-1, -1], 0)


class InputRegressionTests(unittest.TestCase):
    setUp = input_tests.InputQualityTests.setUp
    source = input_tests.InputQualityTests.source
    segment = input_tests.InputQualityTests.segment
    payload = input_tests.InputQualityTests.payload

    def tearDown(self):
        self.engine.segments.close()
        input_tests.InputQualityTests.tearDown(self)

    def edit(self, source, cutout, **action):
        return self.client.post(f"/api/sources/{source['id']}/mask-edit", json={
            "filename": source["images"][0]["file"], "cutout_file": cutout["file"], **action})

    def test_brush_erases_restores_and_undoes_without_model_calls(self):
        source = self.source(); first = self.segment(source).json()
        original = np.array(Image.open(self.engine.store.path("sources", source["id"]) / first["file"]))
        calls = self.engine.quality.segment.call_count
        erased = self.edit(source, first, strokes=[{"points": [[.5, .16]], "radius": .04}])
        self.assertEqual(erased.status_code, 200, erased.text)
        edited = erased.json()
        def rgba(item):
            data = self.client.get(f"/api/sources/{source['id']}/{item['file']}").content
            return np.array(Image.open(io.BytesIO(data)))
        changed = rgba(edited)
        np.testing.assert_array_equal(original[:, :, :3], changed[:, :, :3])
        self.assertEqual(changed[20, 64, 3], 0)
        self.assertEqual(original[20, 64, 3], 255)
        restored = self.edit(source, edited, strokes=[{"points": [[.5, .16]], "radius": .04, "keep": True}])
        self.assertEqual(restored.status_code, 200, restored.text)
        self.assertEqual(rgba(restored.json())[20, 64, 3], 255)
        undo = self.edit(source, restored.json(), undo=True)
        self.assertEqual(undo.json()["file"], edited["file"])
        self.assertEqual(self.engine.quality.segment.call_count, calls)
        self.assertEqual(self.gemini.locate.call_count, 1)

    def test_stale_mask_edit_cannot_replace_newer_alpha(self):
        source = self.source(); first = self.segment(source).json()
        stroke = {"strokes": [{"points": [[.5, .16]], "radius": .04}]}
        new = self.edit(source, first, **stroke).json()
        stale = self.edit(source, first, **stroke)
        self.assertEqual(stale.status_code, 400)
        self.assertEqual(self.engine.store.read("sources", source["id"])["cutouts"][source["images"][0]["file"]]["file"], new["file"])

    def test_brush_revision_blocks_late_sam_commit(self):
        source = self.source(); first = self.segment(source).json()
        entered, release = threading.Event(), threading.Event()
        original = self.engine.quality.segment.side_effect
        def slow(*args):
            entered.set(); release.wait(5)
            return original(*args)
        self.engine.quality.segment.side_effect = slow
        request = {"filename": source["images"][0]["file"], "target": "v\u1eadt th\u1ec3 b\u1ea5t k\u1ef3", "box": [.12, .1, .92, .9]}
        job = self.client.post(f"/api/sources/{source['id']}/segment-jobs", json=request).json()
        try:
            self.assertTrue(entered.wait(2))
            edited = self.edit(source, first, strokes=[{"points": [[.5, .16]], "radius": .04}])
            self.assertEqual(edited.status_code, 200, edited.text)
        finally:
            release.set()
        for _ in range(100):
            status = self.client.get("/api/segment-jobs/" + job["id"]).json()
            if status["status"] == "cancelled": break
            time.sleep(.01)
        self.assertEqual(status["status"], "cancelled")
        current = self.engine.store.read("sources", source["id"])["cutouts"][request["filename"]]
        self.assertEqual(current["file"], edited.json()["file"])

    def test_all_selected_photos_must_reach_backend(self):
        source = self.source(2); files = [image["file"] for image in source["images"]]
        for provider in ("hunyuan",):
            result = self.client.post("/api/generate", json={
                "source_id": source["id"], "provider": provider,
                "views": {"front": files[0]}, "selected_images": files})
            self.assertEqual(result.status_code, 400, result.text)
        self.engine.quality.generate.assert_not_called()

    def test_brush_paths_and_invalid_strokes_rejected(self):
        source = self.source(); cut = self.segment(source).json()
        for stroke in ({"points": [[2, 0]], "radius": .04}, {"points": [[.5]], "radius": .04}, {"points": [[.5, .5]], "radius": 1}):
            self.assertEqual(self.edit(source, cut, strokes=[stroke]).status_code, 422)
        result = self.client.post(f"/api/sources/{source['id']}/mask-edit", json={
            "filename": "../meta.json", "cutout_file": cut["file"], "undo": True})
        self.assertEqual(result.status_code, 400)


class SolverRegressionTests(unittest.TestCase):
    setUp = composition_tests.GeometrySearchTests.setUp
    tearDown = composition_tests.GeometrySearchTests.tearDown

    def test_one_candidate_runs_optimizer_and_needs_no_baseline(self):
        candidates, _ = build_candidates(self.a, self.b, self.plan, [], Transform(), 1)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["optimization"]["performed"])
        self.assertFalse(candidates[0]["baseline"])
        result = rank_candidates(candidates)
        self.assertIsNone(result["comparison"])
        self.assertIsNone(result["candidates"][0]["visual_score"])

    def test_missing_aperture_returns_one_diagnostic_not_five_copies(self):
        plan = self.plan.model_copy(deep=True)
        plan.relation = "around"; plan.fit_mode = "aperture"; plan.base_axis_end = [.5, .8, .5]
        plan.program = compatibility_program(plan)
        candidates, diagnostics = build_candidates(self.a, self.b, plan, [], Transform(), 5)
        self.assertIn("feature_error", diagnostics)
        self.assertEqual(len(candidates), 1)
        self.assertFalse(candidates[0]["eligible"])

    def test_required_deformation_is_not_accepted_by_rigid_solver(self):
        plan = self.plan.model_copy(deep=True); plan.deform = True
        candidates, _ = build_candidates(self.a, self.b, plan, [], Transform(), 1)
        self.assertFalse(candidates[0]["eligible"])


class EvaluationRegressionTests(unittest.TestCase):
    setUp = composition_tests.CompositionAPITests.setUp
    tearDown = composition_tests.CompositionAPITests.tearDown

    def test_failed_geometry_gets_visual_diagnosis_without_overriding_geometry(self):
        candidate = {"id": "c0", "label": "feature failure", "baseline": True,
                     "transform": Transform().model_dump(), "eligible": False,
                     "geometry_score": 0, "metrics": {}, "failures": ["missing aperture"], "warnings": []}
        with patch("unicompose3d.api.composition.build_candidates", return_value=([candidate], {"feature_error": "missing aperture"})):
            prepared = self.client.post("/api/compositions", json=self.request)
        self.assertEqual(prepared.status_code, 200, prepared.text)
        run = prepared.json()
        self.assertEqual(run["evaluation_status"], "pending")
        self.gemini.evaluate.return_value=composition_tests.review(run["candidates"],winner="c0")
        evaluated = self.client.post(f"/api/compositions/{run['id']}/evaluate", json={"renders":[{"candidate_id":"c0","image":composition_tests.image_data()}]})
        self.assertEqual(evaluated.status_code, 200, evaluated.text)
        self.assertEqual(evaluated.json()["api_calls"]["evaluation"], 1)
        self.assertIsNone(evaluated.json()["selected_candidate_id"])
        self.assertFalse(evaluated.json()["auto_apply"])
        self.gemini.evaluate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
