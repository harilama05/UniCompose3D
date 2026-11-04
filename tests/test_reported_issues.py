"""Regression tests for the reported mask, grounding and slow request paths."""
import io
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
from unicompose3d import aperture
from unicompose3d.composition import solid_mesh
from unicompose3d.constraint_solver import ProgramSolver
from unicompose3d.schemas import MaskStroke, PlacementPlan, SpatialProgram, Transform, RegionHints, Capture
from unicompose3d.input_processing import brush_mask
from unicompose3d.api import create_app
from unicompose3d.clients import Gemini
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from quality_progress import gpu_vertex_normals

def camera(slot='B'):
    f = 1 / np.tan(np.deg2rad(35) / 2)
    p = np.array([[f, 0, 0, 0], [0, f, 0, 0], [0, 0, -1.0002, -0.020002], [0, 0, -1, 0]])
    w = np.eye(4)
    w[2, 3] = 2.2
    return Capture(slot=slot, image='', projection=p.flatten(order='F').tolist(), world=w.flatten(order='F').tolist())

def proposal():
    return PlacementPlan(relation='around', base_anchor=[0.5, 0.5, 0.5], product_anchor=[0.5, 0.5, 0.5], rotation_degrees=[0, 0, 0], relative_scale=0.3, confidence=0.9, explanation='generic opening', program=SpatialProgram(features=[{'id': 'a_section', 'slot': 'A', 'kind': 'section', 'point': [1, 0.5, 0.5], 'end': [1, 0.5, 0.9]}, {'id': 'b_opening', 'slot': 'B', 'kind': 'aperture', 'uv': [0.5, 0.5], 'view': 0}], constraints=[{'kind': 'coincident', 'a': 'a_section', 'b': 'b_opening'}, {'kind': 'align', 'a': 'a_section', 'b': 'b_opening'}, {'kind': 'clearance', 'a': 'a_section', 'b': 'b_opening', 'target': 1.05}]))

def partial_band(angle=5 * np.pi / 3):
    a = np.linspace(-angle / 2, angle / 2, 81)
    vertices = np.array([[r * np.cos(t), r * np.sin(t), z] for t in a for r, z in ((0.25, -0.08), (0.35, -0.08), (0.35, 0.08), (0.25, 0.08))])
    faces = []
    for i in range(len(a) - 1):
        for j in range(4):
            p = i * 4 + j
            q = i * 4 + (j + 1) % 4
            s = (i + 1) * 4 + j
            t = (i + 1) * 4 + (j + 1) % 4
            faces.extend([[p, q, t], [p, t, s]])
    faces.extend([[0, 2, 1], [0, 3, 2]])
    k = (len(a) - 1) * 4
    faces.extend([[k, k + 1, k + 2], [k, k + 2, k + 3]])
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.fix_normals()
    return mesh

class ReportedIssueTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_one_background_seed_removes_the_whole_hole_and_preserves_band(self):
        original = self.root / 'original.png'
        cutout = self.root / 'cut.png'
        result = self.root / 'result.png'
        rgb = Image.new('RGB', (160, 160), (165, 135, 100))
        d = ImageDraw.Draw(rgb)
        d.ellipse((20, 20, 140, 140), fill=(210, 215, 220))
        d.ellipse((40, 40, 120, 120), fill=(165, 135, 100))
        rgb.save(original)
        alpha = Image.new('L', rgb.size)
        ImageDraw.Draw(alpha).ellipse((20, 20, 140, 140), fill=255)
        rgba = rgb.convert('RGBA')
        rgba.putalpha(alpha)
        rgba.save(cutout)
        info = brush_mask(original, cutout, [MaskStroke(points=[[0.5, 0.5]], radius=0.01, mode='region', tolerance=12)], result)
        with Image.open(result) as image:
            self.assertEqual(image.getpixel((80, 80))[3], 0)
            self.assertEqual(image.getpixel((102, 80))[3], 0)
            self.assertEqual(image.getpixel((130, 80))[3], 255)
        self.assertGreater(info['region_pixels_removed'], 4000)
        with Image.open(cutout) as image:
            self.assertEqual(image.getpixel((80, 80))[3], 255)

    def test_region_does_not_erase_similar_disconnected_background(self):
        original = self.root / 'original.png'
        cutout = self.root / 'cut.png'
        result = self.root / 'result.png'
        image = Image.new('RGBA', (100, 100), (150, 130, 100, 255))
        d = ImageDraw.Draw(image)
        d.rectangle((45, 0, 55, 99), fill=(0, 0, 255, 255))
        image.convert('RGB').save(original)
        image.save(cutout)
        brush_mask(original, cutout, [MaskStroke(points=[[0.2, 0.5]], radius=0.01, mode='region', tolerance=10)], result)
        with Image.open(result) as final:
            self.assertEqual(final.getpixel((20, 20))[3], 0)
            self.assertEqual(final.getpixel((80, 20))[3], 255)

    def test_region_overflow_is_rejected_without_mutating_current_mask(self):
        original = self.root / 'original.png'
        cutout = self.root / 'cut.png'
        Image.new('RGB', (50, 50), 'white').save(original)
        Image.new('RGBA', (50, 50), 'white').save(cutout)
        with self.assertRaisesRegex(ValueError, 'too many objects'):
            brush_mask(original, cutout, [MaskStroke(points=[[0.5, 0.5]], radius=0.01, mode='region')], self.root / 'out.png')
        self.assertFalse((self.root / 'out.png').exists())

    def test_empty_hole_uv_does_not_require_hitting_a_triangle(self):
        a = self.root / 'a.glb'
        b = self.root / 'b.glb'
        trimesh.creation.cylinder(radius=0.07, height=0.8, sections=48).export(a)
        trimesh.creation.torus(major_radius=0.3, minor_radius=0.05).export(b)
        solver = ProgramSolver(a, b, Transform(), proposal(), [camera()])
        result, _ = solver.optimize(solver.seed())
        self.assertTrue(solver.check(result)['eligible'])

    def test_user_point_is_used_before_invalid_model_uv_is_considered(self):
        a = self.root / 'a.glb'
        b = self.root / 'b.glb'
        trimesh.creation.box().export(a)
        trimesh.creation.box().export(b)
        plan = proposal()
        plan.program = SpatialProgram(features=[{'id': 'a', 'slot': 'A', 'kind': 'point', 'source': 'user', 'uv': [0.99, 0.99], 'view': 7}, {'id': 'b', 'slot': 'B', 'kind': 'point'}], constraints=[{'kind': 'coincident', 'a': 'a', 'b': 'b'}])
        solver = ProgramSolver(a, b, Transform(), plan, [], RegionHints(A_point=[0.2, 0.3, 0.4]))
        np.testing.assert_allclose(solver.features['a']['center'], [0.2, 0.3, 0.4])

    def test_c_shaped_opening_verified_without_requiring_nested_closed_loops(self):
        mesh = partial_band()
        self.assertTrue(mesh.is_watertight)
        frame = aperture.hole_frame(mesh)
        self.assertGreater(frame['angular_coverage'], 0.7)
        self.assertLess(frame['angular_coverage'], 1)
        self.assertAlmostEqual(frame['inner_radius'], 0.25, delta=0.025)
        self.assertGreater(abs(frame['axis'][2]), 0.99)

    def test_large_gap_and_solid_object_are_not_called_verified_openings(self):
        for mesh in (partial_band(np.pi), trimesh.creation.cylinder(radius=0.3, height=0.2, sections=48)):
            with self.assertRaises(ValueError):
                aperture.hole_frame(mesh)

    def test_texture_seams_do_not_make_closed_geometry_seem_open(self):
        source = trimesh.creation.torus(major_radius=0.3, minor_radius=0.05)
        mesh = trimesh.Trimesh(source.triangles.reshape(-1, 3), np.arange(len(source.faces) * 3).reshape(-1, 3), process=False)
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=np.random.default_rng(1).random((len(mesh.vertices), 2)))
        self.assertFalse(mesh.is_watertight)
        solid, closed = solid_mesh(mesh)
        self.assertTrue(closed)
        self.assertEqual(len(solid.faces), len(source.faces))

    def test_device_normals_match_original_unweighted_mean(self):
        import torch
        mesh = trimesh.creation.icosphere(subdivisions=2)
        actual = gpu_vertex_normals(torch.tensor(mesh.vertices, dtype=torch.float32), torch.tensor(mesh.faces), torch.tensor(mesh.face_normals, dtype=torch.float32)).numpy()
        expected = trimesh.geometry.mean_vertex_normals(len(mesh.vertices), mesh.faces, mesh.face_normals)
        np.testing.assert_allclose(actual, expected, atol=2e-06)

    def test_background_composition_request_returns_before_llm_finishes(self):
        planner = Mock(spec=Gemini)
        planner.key = 'fixture'
        planner.model = 'fixture'
        app = create_app(self.root / 'data', 'test-token', planner)
        engine = app.state.engine
        entered, release = (threading.Event(), threading.Event())
        a = self.root / 'a.glb'
        trimesh.creation.box().export(a)
        asset = engine.asset(a, 'generic')

        def delayed(request):
            entered.set()
            release.wait(5)
            ident, _ = engine.store.create('compositions', {'status': 'prepared'})
            return engine.store.read('compositions', ident)
        engine.prepare_composition = delayed
        client = TestClient(app)
        client.headers['Authorization'] = 'Bearer test-token'
        try:
            response = client.post('/api/composition-jobs', json={'base_id': asset['id'], 'product_id': asset['id']})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(entered.wait(1))
            self.assertFalse(release.is_set())
            task = response.json()
            self.assertIn(task['status'], ['queued', 'running'])
            release.set()
            for _ in range(100):
                status = client.get('/api/composition-jobs/' + task['id']).json()
                if status['status'] == 'success':
                    break
                time.sleep(0.01)
            self.assertIn('result', status)
        finally:
            release.set()
            engine.pool.shutdown(wait=True)
            engine.segments.close()

    def test_smart_region_crosses_a_two_colour_background_inside_an_opening(self):
        original = self.root / 'two.png'
        cutout = self.root / 'cut.png'
        result = self.root / 'result.png'
        rgb = Image.new('RGB', (160, 160), (20, 20, 25))
        d = ImageDraw.Draw(rgb)
        d.rectangle((0, 90, 159, 159), fill=(235, 230, 220))
        d.ellipse((20, 20, 140, 140), outline=(90, 160, 210), width=17)
        rgb.save(original)
        alpha = Image.new('L', rgb.size)
        ImageDraw.Draw(alpha).ellipse((20, 20, 140, 140), fill=255)
        rgba = rgb.convert('RGBA')
        rgba.putalpha(alpha)
        rgba.save(cutout)
        brush_mask(original, cutout, [MaskStroke(points=[[0.5, 0.4]], radius=0.005, mode='region', tolerance=10)], result)
        with Image.open(result) as image:
            self.assertEqual(image.getpixel((80, 65))[3], 0)
            self.assertEqual(image.getpixel((80, 105))[3], 0)
            self.assertEqual(image.getpixel((130, 80))[3], 255)

    def test_near_silhouette_uv_can_snap_but_a_distant_point_cannot(self):
        from unicompose3d.grounding import snap_camera_anchor
        mesh = trimesh.creation.box()
        capture = camera()
        p = np.asarray(capture.projection).reshape(4, 4, order='F')
        world = np.asarray(capture.world).reshape(4, 4, order='F')
        vertex = np.r_[0.5, 0.5, 0.5, 1.0]
        clip = p @ np.linalg.inv(world) @ vertex
        uv = np.array([(clip[0] / clip[3] + 1) / 2, (1 - clip[1] / clip[3]) / 2])
        self.assertIsNotNone(snap_camera_anchor(mesh, capture, (uv + [0.004, -0.004]).tolist()))
        self.assertIsNone(snap_camera_anchor(mesh, capture, [0.99, 0.99]))

    def test_geometry_cache_returns_isolated_copies_and_tracks_file_changes(self):
        from unicompose3d.geometry import flat_mesh
        path = self.root / 'cached.glb'
        trimesh.creation.box().export(path)
        one = flat_mesh(path)
        one.vertices += 100
        two = flat_mesh(path)
        self.assertLess(abs(two.bounds).max(), 1)
        trimesh.creation.box(extents=[2, 2, 2]).export(path)
        self.assertAlmostEqual(flat_mesh(path).extents[0], 2)

    def test_invalid_grounding_gets_feedback_and_successful_retry_is_cached(self):
        planner = Mock(spec=Gemini)
        planner.key = 'fixture'
        planner.model = 'fixture'
        invalid = proposal()
        invalid.program = SpatialProgram(features=[{'id': 'a', 'slot': 'A', 'kind': 'point'}, {'id': 'b', 'slot': 'B', 'kind': 'point'}], constraints=[{'kind': 'align', 'a': 'a', 'b': 'b'}])
        valid = proposal()
        valid.program = SpatialProgram(features=[{'id': 'a', 'slot': 'A', 'kind': 'surface', 'point': [0.5, 1, 0.5]}, {'id': 'b', 'slot': 'B', 'kind': 'surface', 'point': [0.5, 0, 0.5]}], constraints=[{'kind': 'contact', 'a': 'a', 'b': 'b', 'target': 0.003}, {'kind': 'proportion', 'a': 'a', 'b': 'b', 'target': 0.3}])
        planner.plan.side_effect = [invalid, valid]
        app = create_app(self.root / 'store', 'test', planner)
        engine = app.state.engine
        raw = self.root / 'cube.glb'
        trimesh.creation.box().export(raw)
        asset = engine.asset(raw, 'arbitrary object')
        client = TestClient(app)
        client.headers['Authorization'] = 'Bearer test'
        body = {'base_id': asset['id'], 'product_id': asset['id'], 'candidate_count': 1, 'captures': [camera('A').model_dump(), camera('B').model_dump()]}
        try:
            response = client.post('/api/compositions', json=body)
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertTrue(result['diagnostics']['grounding_repair_attempted'])
            self.assertTrue(result['candidates'][0]['eligible'], result)
            self.assertEqual(result['api_calls']['plan'], 2)
            self.assertIn('could not resolve', planner.plan.call_args.args[2])
            again = client.post('/api/compositions', json=body).json()
            self.assertEqual(again['id'], result['id'])
            self.assertEqual(planner.plan.call_count, 2)
            (engine.store.path('compositions', result['id']) / 'meta.json').unlink()
            planner.plan.side_effect = None
            planner.plan.return_value = valid
            rebuilt = client.post('/api/compositions', json=body)
            self.assertEqual(rebuilt.status_code, 200, rebuilt.text)
        finally:
            engine.pool.shutdown(wait=True)
            engine.segments.close()
if __name__ == '__main__':
    unittest.main()
