"""End-to-end browser test with real geometry/rendering and an offline VLM.

The fixture exists only in this QA script, never in the production launcher.
No external model calls or Tripo credits are used.
"""
from __future__ import annotations
import base64
import io
import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
import numpy as np
import requests
import uvicorn
from PIL import Image
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unicompose3d.api import create_app
from unicompose3d.clients import ProviderError
from unicompose3d.schemas import CandidateReview, PlacementPlan, VisualEvaluation, SpatialProgram
from unicompose3d.constraint_solver import compatibility_program

class OfflineVLM:
    key = 'offline-fixture-only'
    model = 'offline-fixture-only'

    def __init__(self):
        self.plan_calls = 0
        self.evaluation_calls = 0
        self.mode = 'normal'
        self.block = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.captured_sheets = []

    def plan(self, a, b, instruction, captures):
        self.plan_calls += 1
        assert len(captures) == 8
        result = PlacementPlan(relation='on', base_anchor=[0.5, 1, 0.5], product_anchor=[0.5, 0, 0.5], rotation_degrees=[0, 0, 0], relative_scale=0.3, confidence=0.9, explanation='Fixture: place a vase on a table; not real API reasoning.')
        result.program = compatibility_program(result)
        from unicompose3d.schemas import SpatialConstraint
        result.program.constraints.append(SpatialConstraint(kind='proportion', a='a', b='b', target=0.3))
        if self.mode == 'feature':
            result.program = SpatialProgram.model_validate({'features': [{'id': 'a', 'slot': 'A', 'kind': 'point'}, {'id': 'b', 'slot': 'B', 'kind': 'point'}], 'constraints': [{'kind': 'align', 'a': 'a', 'b': 'b'}]})
        return result

    def evaluate(self, plan, candidates, renders, instruction):
        self.evaluation_calls += 1
        self.entered.set()
        if self.block:
            if not self.release.wait(10):
                raise ProviderError('Fixture timeout')
        self.captured_sheets = renders
        for render in renders:
            image = Image.open(io.BytesIO(base64.b64decode(render.image.split(',', 1)[-1])))
            assert image.size == ((960, 1492) if render.layout.endswith('contact-v2') else (960, 1012))
            for x, y in ((16, 74), (496, 74), (16, 554), (496, 554)) + (((16, 1034), (496, 1034)) if render.layout.endswith('contact-v2') else ()):
                pixels = np.asarray(image.convert('RGB').crop((x, y, x + 448, y + 448)))
                assert float(pixels.std()) > 2, 'Blank quadrant: expected actual rendered geometry'
        if self.mode == 'quota':
            raise ProviderError('Fixture: Gemini quota exhausted')
        return VisualEvaluation(reviews=[CandidateReview(candidate_id=c['id'], acceptable=self.mode != 'reject', contact_score=0.96 if c['id'] == 'c1' else 0.55, placement_score=0.96 if c['id'] == 'c1' else 0.55, proportion_score=0.96 if c['id'] == 'c1' else 0.55, appearance_score=0.96 if c['id'] == 'c1' else 0.55, explanation='Fixture evaluation for QA; not evidence of real VLM quality.') for c in candidates], summary='Fixture candidate review; not a real API result.')

def main():
    output = Path(__file__).resolve().parents[1] / 'test-results'
    output.mkdir(exist_ok=True)
    fixture = OfflineVLM()
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        app = create_app(temporary, 'offline-browser-test', gemini=fixture)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning', access_log=False))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        origin = f'http://127.0.0.1:{port}'
        for _ in range(100):
            try:
                if requests.get(origin + '/health', timeout=0.2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.05)
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True, args=['--enable-webgl', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
                page = browser.new_page(viewport={'width': 1440, 'height': 1000}, accept_downloads=True)
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(origin + '/#token=offline-browser-test')
                page.locator('#demoBtn').click()
                page.locator('#nameB').wait_for(state='visible')
                page.locator('#composeMode').select_option('geometry', force=True)
                assert page.locator('#composeMode').input_value() == 'geometry'
                page.locator('#composeBtn').click()
                page.wait_for_function("!document.querySelector('#composeBtn').disabled && !document.querySelector('#compositionActions').classList.contains('hidden')")
                assert fixture.plan_calls == 1 and fixture.evaluation_calls == 0
                page.locator('#candidatesBtn').click()
                assert page.locator('.candidate-card').count() == 1
                page.locator('#candidatesDialog [data-close]').click()
                page.locator('#composeMode').select_option('evaluated', force=True)
                page.locator('#candidateCount').select_option('3', force=True)
                page.locator('#composeBtn').click()
                page.wait_for_function("!document.querySelector('#compositionActions').classList.contains('hidden')")
                page.wait_for_function("!document.querySelector('#composeBtn').disabled")
                assert fixture.plan_calls == 1 and fixture.evaluation_calls == 1
                page.locator('#candidatesBtn').click()
                assert page.locator('.candidate-card').count() == 3
                assert page.locator('.candidate-card.recommended').get_attribute('data-candidate') == 'c1'
                with page.expect_download() as event:
                    page.locator('#dialogReportBtn').click()
                event.value.save_as(output / 'composition-report.json')
                report = json.loads((output / 'composition-report.json').read_text(encoding='utf-8'))
                assert report['evaluation_status'] == 'completed'
                assert len(report['renders']) == 3 and all((r['layout'] == 'front-side-back-oblique-contact-v2' for r in report['renders']))
                assert report['api_calls'] == {'plan': 0, 'evaluation': 1}, report['api_calls']
                first_image = fixture.captured_sheets[0].image.split(',', 1)[-1]
                (output / 'composition-contact-sheet.png').write_bytes(base64.b64decode(first_image))
                page.screenshot(path=str(output / 'composition-candidates.png'), full_page=True)
                page.locator('[data-candidate="c0"] button').click()
                page.locator('#position0').fill('0.2')
                page.locator('#position0').press('Tab')
                assert page.locator('#compositionDirty').is_visible()
                page.locator('#composeMode').select_option('quick', force=True)
                page.locator('#composeBtn').click()
                page.wait_for_function("document.querySelector('#planInfo').textContent.includes('Single-pass composition')")
                assert fixture.plan_calls == 1 and fixture.evaluation_calls == 1
                page.locator('#composeMode').select_option('evaluated', force=True)
                page.locator('#composeBtn').click()
                page.wait_for_function("document.querySelector('#planInfo').textContent.includes('Reused')")
                page.wait_for_function("!document.querySelector('#composeBtn').disabled")
                assert fixture.plan_calls == 1 and fixture.evaluation_calls == 1
                page.locator('#candidateCount').select_option('5', force=True)
                page.locator('#composeBtn').click()
                page.wait_for_function("!document.querySelector('#composeBtn').disabled")
                page.locator('#candidatesBtn').click()
                assert page.locator('.candidate-card').count() == 5
                page.locator('#candidatesDialog [data-close]').click()
                assert fixture.plan_calls == 1 and fixture.evaluation_calls == 2
                fixture.mode = 'quota'
                page.locator('#instruction').fill('Test quota handling')
                previous = page.locator('#position1').input_value()
                page.locator('#composeBtn').click()
                page.wait_for_function("!document.querySelector('#composeBtn').disabled")
                assert page.locator('#position1').input_value() == previous
                assert 'not applied automatically' in page.locator('#planInfo').inner_text()
                page.locator('#candidatesBtn').click()
                assert 'quota' in page.locator('#evaluationWarnings').inner_text().lower()
                page.locator('#candidatesDialog [data-close]').click()
                calls_before_retry = fixture.plan_calls
                fixture.mode = 'normal'
                page.locator('#composeBtn').click()
                page.wait_for_function("!document.querySelector('#composeBtn').disabled")
                assert fixture.plan_calls == calls_before_retry
                fixture.mode = 'reject'
                page.locator('#instruction').fill('Test rejection of all candidates')
                previous = page.locator('#position1').input_value()
                page.locator('#composeBtn').click()
                page.wait_for_function("!document.querySelector('#composeBtn').disabled")
                assert page.locator('#position1').input_value() == previous
                page.locator('#candidatesBtn').click()
                assert page.locator('.candidate-card.recommended').count() == 0
                page.locator('#candidatesDialog [data-close]').click()
                fixture.mode = 'normal'
                fixture.block = True
                fixture.entered.clear()
                fixture.release.clear()
                page.locator('#instruction').fill('Test manual editing while waiting')
                page.locator('#composeBtn').click()
                page.wait_for_function("document.querySelector('#loadingText').textContent.includes('Step 3/3')", timeout=60000)
                assert fixture.entered.wait(10)
                page.locator('#position0').fill('0.456')
                page.locator('#position0').press('Tab')
                fixture.release.set()
                page.wait_for_function("!document.querySelector('#composeBtn').disabled")
                assert page.locator('#position0').input_value() == '0.456'
                assert 'changed' in page.locator('#toast').inner_text()
                fixture.block = False
                page.locator('#composeBtn').click()
                page.wait_for_function("!document.querySelector('#composeBtn').disabled")
                page.set_viewport_size({'width': 390, 'height': 844})
                page.locator('#candidatesBtn').click()
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1')
                page.screenshot(path=str(output / 'composition-mobile.png'), full_page=True)
                page.locator('#candidatesDialog [data-close]').click()
                fixture.mode = 'feature'
                page.locator('#instruction').fill('missing geometry feature')
                previous_calls = fixture.evaluation_calls
                page.locator('#composeBtn').click()
                page.wait_for_function("!document.querySelector('#composeBtn').disabled")
                assert fixture.evaluation_calls == previous_calls
                assert page.locator('.candidate-card').count() == 1
                assert page.locator('.candidate-card.rejected').count() == 1
                assert not errors, errors
                browser.close()
                print('PASS: real candidate geometry + four-view renders + VLM fixture + ranking + cache + quota/manual fallback + reject-all + stale-edit protection + report + mobile UI.')
                print('Offline fixture only. No external VLM or Tripo calls.')
        except Exception:
            print('QA JS errors:', errors if 'errors' in locals() else [], flush=True)
            print('QA jobs:', app.state.engine.store.list('jobs'), flush=True)
            raise
        finally:
            fixture.release.set()
            app.state.engine.stopping.set()
            server.should_exit = True
            thread.join(5)
            app.state.engine.pool.shutdown(wait=True)
if __name__ == '__main__':
    main()
