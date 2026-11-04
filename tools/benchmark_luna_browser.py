"""Opt-in live Luna benchmark. Uses real UI/HTTP/render/LLM, not a fixture VLM."""
import os, sys, json, time, tempfile, threading, socket, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests, uvicorn, trimesh
from playwright.sync_api import sync_playwright
from unicompose3d.api import create_app
from unicompose3d.clients import ProviderError
from unicompose3d.openai_client import OpenAIClient
from composition_cases import cases
parser = argparse.ArgumentParser()
parser.add_argument('--live', action='store_true')
parser.add_argument('--max-requests', type=int, default=12)
parser.add_argument('--output', type=Path, default=Path('test-results/luna-live'))
parser.add_argument('--cases', default='support,beside,side_attachment')
args = parser.parse_args()
if not args.live:
    parser.error('Add --live to authorize a paid API benchmark; no calls have been made.')
if not os.environ.get('OPENAI_API_KEY'):
    parser.error('Set OPENAI_API_KEY privately; no calls have been made.')
if not 1 <= args.max_requests <= 30:
    parser.error('max-requests must be 1..30')
instructions = {'support': 'Place B on top of A.', 'beside': 'Place B to the right of A.', 'side_attachment': 'Attach B to the right side of A.'}
selected = args.cases.split(',')
if set(selected) - set(instructions):
    parser.error('Available cases: ' + ','.join(instructions))
args.output.mkdir(parents=True, exist_ok=True)

class Ledger(requests.Session):

    def __init__(self):
        super().__init__()
        self.calls = []

    def post(self, url, **kwargs):
        if len(self.calls) >= args.max_requests:
            raise ProviderError('Live benchmark request budget reached.')
        entry = {'request': len(self.calls) + 1, 'model': kwargs['json']['model']}
        self.calls.append(entry)
        started = time.monotonic()
        try:
            response = super().post(url, **kwargs)
            entry.update(status=response.status_code, seconds=round(time.monotonic() - started, 3))
            try:
                entry['usage'] = response.json().get('usage')
            except ValueError:
                pass
            return response
        finally:
            (args.output / 'requests.json').write_text(json.dumps(self.calls, indent=2), encoding='utf-8')
ledger = Ledger()
llm = OpenAIClient(transport=ledger)
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    app = create_app(root / 'data', 'bench', llm)
    engine = app.state.engine
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning', access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    results = []
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
            for name, a, b, plan, base, expected in cases():
                if name not in selected:
                    continue
                page.goto(origin + '/#token=bench')
                for slot, mesh in (('A', a), ('B', b)):
                    path = root / (slot + '.glb')
                    mesh.export(path)
                    page.locator('#upload' + slot).click()
                    page.locator('[data-kind="model"]').click()
                    page.locator('#objectName').fill(slot + '_' + name)
                    page.locator('#sourceFiles').set_input_files(str(path))
                    page.locator('#prepareBtn').click()
                    page.locator('#name' + slot).wait_for(state='visible', timeout=90000)
                page.locator('#agentRounds').select_option('1', force=True)
                page.locator('#instruction').fill(instructions[name])
                page.locator('#composeBtn').click()
                page.wait_for_function("!document.querySelector('#composeBtn').disabled", timeout=360000)
                record = next((r for r in engine.store.list('compositions') if r['instruction'] == instructions[name]), None)
                if record:
                    renders = engine.store.path('compositions', record['id'])
                    for path in renders.glob('c*.png'):
                        (args.output / (name + '-' + path.name)).write_bytes(path.read_bytes())
                    (args.output / (name + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
                    results.append({'case': name, 'prompt': instructions[name], 'geometry_pass': any((c['eligible'] for c in record['candidates'])), 'selected': record.get('selected_candidate_id'), 'evaluation_status': record['evaluation_status'], 'candidates': len(record['candidates'])})
                else:
                    results.append({'case': name, 'error': page.locator('#sceneStatus').text_content()})
                page.screenshot(path=str(args.output / (name + '.png')), full_page=True)
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        engine.stopping.set()
        engine.segments.close()
        engine.pool.shutdown(wait=True)
    (args.output / 'summary.json').write_text(json.dumps({'mode': 'live Luna; self-evaluation is not independent human ground truth', 'model': llm.model, 'request_budget': args.max_requests, 'actual_requests': len(ledger.calls), 'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved live benchmark; actual requests:', len(ledger.calls))
