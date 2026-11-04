"""Real UI/HTTP/GLB agent loop, with an intentionally fallible offline VLM."""
import os, sys, socket, tempfile, threading, time, json, zipfile
from pathlib import Path
import requests, uvicorn
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unicompose3d.api import create_app
from composition_browser_smoke import OfflineVLM

class RevisingVLM(OfflineVLM):

    def __init__(self):
        super().__init__()
        self.revise_calls = 0
        self.previous = []
        self.reject_all = False

    def revise(self, a, b, instruction, captures, feedback, renders):
        self.revise_calls += 1
        assert feedback['instruction'] == instruction
        assert renders and renders[0]['image']
        assert feedback['candidates'][0]['metrics']['scale_evidence']['verified']
        self.previous.append(feedback)
        result = self.plan(a, b, instruction, captures)
        next((c for c in result.program.constraints if c.kind == 'proportion')).target = 0.3 - 0.02 * self.revise_calls
        return result

    def evaluate(self, plan, candidates, renders, instruction):
        self.mode = 'reject' if self.reject_all or self.evaluation_calls == 0 else 'normal'
        result = super().evaluate(plan, candidates, renders, instruction)
        for review in result.reviews:
            review.contact_score = review.placement_score = review.proportion_score = review.appearance_score = 0.95
            review.explanation = 'Revise requested part' if self.mode == 'reject' else 'Accepted fixture'
            review.issues = ['Move to requested part'] if self.mode == 'reject' else []
        result.summary = 'Needs another agent round' if self.mode == 'reject' else 'Fixture accepted after revision'
        return result

def main():
    output = Path(__file__).resolve().parents[1] / 'test-results'
    output.mkdir(exist_ok=True)
    fixture = RevisingVLM()
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        app = create_app(tmp, 'agent-qa', fixture)
        engine = app.state.engine
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
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=['--enable-webgl', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
                page = browser.new_page(viewport={'width': 1440, 'height': 1100})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(origin + '/#token=agent-qa')
                page.locator('#demoBtn').click()
                page.locator('#nameB').wait_for(state='visible')
                assert page.locator('#composeMode').input_value() == 'agent'
                page.locator('#instruction').fill('Put the product on the requested support surface')
                page.locator('#composeBtn').click()
                page.wait_for_function("!document.querySelector('#composeBtn').disabled && !document.querySelector('#compositionActions').classList.contains('hidden')", timeout=90000)
                records = engine.store.list('compositions')
                assert len(records) == 2, records
                assert fixture.revise_calls == 1 and fixture.evaluation_calls == 2
                final = next((record for record in records if record['agent_round'] == 2))
                assert final['auto_apply']
                first = next((record for record in records if record['agent_round'] == 1))
                assert first['candidates'][0]['transform']['scale'] != final['candidates'][0]['transform']['scale']
                assert 'Agent: 2' in page.locator('#planInfo').text_content()
                assert len(final['agent_history']) == 1
                with page.expect_download() as download_info:
                    page.locator('#compositionDiagnostics').click()
                download = download_info.value
                with zipfile.ZipFile(download.path()) as archive:
                    assert {'A.glb', 'B.glb', 'round_1/report.json', 'round_2/report.json'} <= set(archive.namelist())
                    assert json.loads(archive.read('round_2/report.json'))['id'] == final['id']
                fields = ['#position0', '#position1', '#position2', '#scale0', '#scale1', '#scale2']
                before = [page.locator(field).input_value() for field in fields]
                fixture.reject_all = True
                page.locator('#instruction').fill('A different goal that the fixture keeps rejecting')
                page.locator('#composeBtn').click()
                page.wait_for_function("!document.querySelector('#composeBtn').disabled && document.querySelector('#planInfo').textContent.includes('Agent: 3')", timeout=90000)
                assert [page.locator(field).input_value() for field in fields] == before
                latest = engine.store.list('compositions')[0]
                assert latest['agent_round'] == 3 and (not latest['auto_apply'])
                assert not errors, errors
                page.screenshot(path=str(output / 'agent-composition.png'), full_page=True)
                browser.close()
                print('PASS: agent rejects round one, uses real render/geometry feedback, accepts round two; capped failure at three preserves scene. VLM is an offline fixture.')
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            engine.segments.close()
            engine.pool.shutdown(wait=True)
if __name__ == '__main__':
    main()
