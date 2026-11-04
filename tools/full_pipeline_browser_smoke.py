"""Hunyuan-only Studio, segmentation and clarification UI; GPU/LLM fixtures."""
import io, json, socket, sys, tempfile, threading, time, shutil
from pathlib import Path
from unittest.mock import Mock
import requests, trimesh, uvicorn
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unicompose3d.api import create_app
from unicompose3d.schemas import TargetLocation, AgentQuestion
from composition_browser_smoke import OfflineVLM

class PlainVLM(OfflineVLM):

    def plan(self, a, b, instruction, captures):
        plan = super().plan(a, b, instruction, captures)
        if instruction == 'Try it' and (not a.get('user_answers')):
            plan.program = None
            plan.clarification = AgentQuestion(question='Where would you like the product?', options=['On top', 'Beside'])
        return plan

def main():
    output = Path(__file__).resolve().parents[1] / 'test-results'
    output.mkdir(exist_ok=True)
    llm = PlainVLM()
    llm.locate = lambda *a: TargetLocation(found=True, confidence=0.9, box=[0.1, 0.1, 0.9, 0.9], explanation='Fixture')
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        app = create_app(Path(tmp) / 'data', 'full-qa', llm)
        engine = app.state.engine
        engine.quality = Mock()
        engine.quality.ready = True
        engine.quality.segment_ready = True

        def mask(image, box, points, labels, directory, stopping, **kwargs):
            with Image.open(image) as photo:
                out = Image.new('L', photo.size, 0)
                ImageDraw.Draw(out).rectangle((20, 20, 235, 235), fill=255)
                out.save(directory / 'mask.png')
            return directory / 'mask.png'
        engine.quality.segment.side_effect = mask
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning', access_log=False))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        origin = 'http://127.0.0.1:' + str(port)
        try:
            for _ in range(100):
                try:
                    if requests.get(origin + '/health', timeout=0.2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.05)
            demo = requests.post(origin + '/api/demo', headers={'Authorization': 'Bearer full-qa'}).json()
            generated = []

            def reconstruct(views, directory, stopping, texture=True, progress=None):
                generated.append(list(views))
                if progress:
                    progress('shape', {'shape_s': 0.01})
                src = engine.model_path(demo['A' if len(generated) == 1 else 'B']['id'])
                target = directory / 'result.glb'
                shutil.copyfile(src, target)
                (directory / 'hunyuan-diagnostics.json').write_text(json.dumps({'timings': {'shape_s': 0.01}, 'used_images': len(views)}))
                return target
            engine.quality.generate.side_effect = reconstruct
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=['--enable-webgl', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
                page = browser.new_page(viewport={'width': 1440, 'height': 1100}, accept_downloads=True)
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(origin + '/#token=full-qa')

                def add(slot, count):
                    page.locator('#upload' + slot).click()
                    if count > 1:
                        page.locator('[data-kind="multiview"]').click()
                    page.locator('#objectName').fill('Object ' + slot)
                    uploads = []
                    for i in range(count):
                        blob = io.BytesIO()
                        Image.new('RGB', (256, 256), (80 + i * 20, 120, 100)).save(blob, 'PNG')
                        uploads.append({'name': 'photo' + str(i) + '.png', 'mimeType': 'image/png', 'buffer': blob.getvalue()})
                    page.locator('#sourceFiles').set_input_files(uploads)
                    page.locator('#prepareBtn').click()
                    page.locator('#inputPreparation').wait_for(state='visible')
                    assert page.locator('#reconstructionProvider').input_value() == 'hunyuan'
                    assert page.locator('#reconstructionProvider option').count() == 1
                    if count > 1:
                        page.locator('#advancedViewAssignments summary').click()
                        page.locator('#assign_left').select_option(index=2)
                    page.locator('#segmentSelected').click()
                    page.wait_for_function("!document.querySelector('#cutoutPreview').hidden && !document.querySelector('#segmentCurrent').disabled", timeout=60000)
                    page.locator('#confirmCutouts').check()
                    page.locator('#generateBtn').click()
                    page.locator('#reconstructionReview').wait_for(state='visible', timeout=60000)
                    page.wait_for_function("!document.querySelector('#reviewAccept').disabled", timeout=60000)
                    page.locator('#reviewAccept').click()
                    page.locator('#name' + slot).wait_for(state='visible')
                add('A', 1)
                add('B', 2)
                assert list(map(len, generated)) == [1, 2]
                assert engine.quality.segment.call_count == 3
                page.locator('#instruction').fill('Try it')
                page.locator('#composeBtn').click()
                page.locator('#agentQuestion').wait_for(state='visible', timeout=90000)
                assert llm.evaluation_calls == 0
                page.screenshot(path=str(output / 'agent-question.png'), full_page=True)
                page.locator('#agentQuestion button').first.click()
                page.wait_for_function("!document.querySelector('#composeBtn').disabled && !document.querySelector('#compositionActions').classList.contains('hidden')", timeout=90000)
                assert llm.plan_calls == 2
                record = next((c for c in engine.store.list('compositions') if c['status'] != 'needs_input'))
                assert record['clarification_answers'][0]['answer'] == 'On top'
                assert record['api_calls']['plan'] == 2
                page.locator('#saveBtn').click()
                with page.expect_download() as download:
                    page.locator('#exportBtn').click()
                target = output / 'full-studio-scene.glb'
                download.value.save_as(target)
                assert len(trimesh.load(target, force='scene', process=False).graph.nodes_geometry) >= 2
                page.locator('#helpBtn').click()
                text = page.locator('#helpDialog').inner_text().lower()
                assert all((word not in text for word in ('trellis', 'triposr', 'sf3d', 'tripo')))
                page.locator('#helpDialog [data-close]').click()
                assert not errors, errors
                page.screenshot(path=str(output / 'full-studio-flow.png'), full_page=True)
                browser.close()
                print('PASS: Hunyuan-only upload/SAM/preview/A+B -> short prompt/question/options -> geometry/render/evaluation -> save/export. GPU/LLM are fixtures.')
        finally:
            engine.stopping.set()
            server.should_exit = True
            thread.join(timeout=5)
            engine.segments.close()
            engine.pool.shutdown(wait=True)
if __name__ == '__main__':
    main()
