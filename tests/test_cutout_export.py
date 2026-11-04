"""Export segmented images without inference or API calls."""
import io, json, tempfile, unittest, zipfile
from pathlib import Path
from PIL import Image
import test_input_quality as helpers

class CutoutExportTests(unittest.TestCase):
    setUp = helpers.InputQualityTests.setUp
    source = helpers.InputQualityTests.source
    segment = helpers.InputQualityTests.segment

    def tearDown(self):
        self.engine.segments.close()
        helpers.InputQualityTests.tearDown(self)

    def payload(self, source, cutouts):
        return {'files': [i['file'] for i in source['images']], 'target': 'an arbitrary object', 'cutout_files': {i['file']: c['file'] for i, c in zip(source['images'], cutouts)}, 'confirmed': True}

    def test_export_all_five_selected_images_preserves_alpha_and_order_without_api(self):
        source = self.source(5)
        cuts = [self.segment(source, i).json() for i in range(5)]
        body = self.payload(source, cuts)
        body['files'].reverse()
        calls = (self.gemini.locate.call_count, self.engine.quality.segment.call_count)
        response = self.client.post(f"/api/sources/{source['id']}/export-cutouts", json=body)
        self.assertEqual(response.status_code, 200, response.text[:200] if response.headers.get('content-type') != 'application/zip' else 'zip')
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = archive.namelist()
            self.assertEqual(sum((p.endswith('.png') for p in names)), 5)
            manifest = json.loads(archive.read(f"object_{source['id']}/manifest.json"))
            self.assertEqual([item['original'] for item in manifest['images']], body['files'])
            for name in names:
                if name.endswith('.png'):
                    with Image.open(io.BytesIO(archive.read(name))) as image:
                        self.assertEqual(image.getpixel((60, 60))[3], 0)
                        self.assertEqual(image.size, (128, 128))
        self.assertEqual((self.gemini.locate.call_count, self.engine.quality.segment.call_count), calls)

    def test_stale_unconfirmed_unknown_or_duplicate_selection_is_rejected(self):
        source = self.source()
        cut = self.segment(source).json()
        body = self.payload(source, [cut])
        for changes in ({'confirmed': False}, {'cutout_files': {}}, {'files': ['../meta.json']}, {'files': body['files'] * 2}):
            response = self.client.post(f"/api/sources/{source['id']}/export-cutouts", json={**body, **changes})
            self.assertEqual(response.status_code, 400, response.text)
