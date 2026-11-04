"""Real browser check: ineligible geometry still receives visual diagnosis."""
import sys, tempfile, time, socket, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests, uvicorn
from playwright.sync_api import sync_playwright
from unicompose3d.api import create_app
from composition_browser_smoke import OfflineVLM
fixture = OfflineVLM()
fixture.mode = 'feature'
output = Path(__file__).resolve().parents[1] / 'test-results'
output.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    app = create_app(tmp, 'diagnostic', fixture)
    engine = app.state.engine
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning', access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        origin = 'http://127.0.0.1:' + str(port)
        for _ in range(100):
            try:
                if requests.get(origin + '/health', timeout=0.2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.05)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=['--enable-webgl', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
            page = browser.new_page(viewport={'width': 1440, 'height': 1100})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(origin + '/#token=diagnostic')
            page.locator('#demoBtn').click()
            page.locator('#nameB').wait_for(state='visible')
            fields = ['#position0', '#position1', '#position2', '#scale0', '#scale1', '#scale2']
            before = [page.locator(x).input_value() for x in fields]
            page.locator('#agentRounds').select_option('1', force=True)
            page.locator('#instruction').fill('Place B on the intended part of A.')
            page.locator('#composeBtn').click()
            page.wait_for_function("!document.querySelector('#composeBtn').disabled && !document.querySelector('#compositionActions').classList.contains('hidden')", timeout=90000)
            record = engine.store.list('compositions')[0]
            assert not any((c['eligible'] for c in record['candidates']))
            assert record['evaluation_status'] == 'completed'
            assert fixture.evaluation_calls == 1
            assert len(record['renders']) == len(record['candidates'])
            assert not record['auto_apply'] and record['selected_candidate_id'] is None
            assert [page.locator(x).input_value() for x in fields] == before
            page.locator('#candidatesBtn').click()
            page.locator('#candidatesDialog').wait_for(state='visible')
            assert [page.locator(x).input_value() for x in fields] == before
            assert not errors, errors
            page.screenshot(path=str(output / 'diagnostic-visual-review.png'), full_page=True)
            browser.close()
            print('PASS: failed geometry rendered and visually diagnosed; acceptance blocked; review button preserves scene. Offline VLM fixture.')
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        engine.stopping.set()
        engine.segments.close()
        engine.pool.shutdown(wait=True)
