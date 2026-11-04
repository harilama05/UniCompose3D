import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Reconstructed assets are previewed independently; the accepted scene stays intact.
export function createReconstructionReview({ api, post, safe, loadAsset, watchJob, toast, download, loader, environment, addLights }) {
  const $ = id => document.getElementById(id), dialog = $('reconstructionReview');
  const history = new Map();
  let current = null, renderer, scene, camera, controls, root, resizeObserver, serial = 0, pendingJob = null;
  function disposeRoot(object = root) {
    if (!object) return;
    scene?.remove(object);
    const geometries = new Set(), materials = new Set(), textures = new Set();
    object.traverse(node => {
      if (node.geometry) geometries.add(node.geometry);
      for (const material of Array.isArray(node.material) ? node.material : node.material ? [node.material] : []) {
        materials.add(material);
        for (const value of Object.values(material)) if (value?.isTexture && value !== environment) textures.add(value);
      }
    });
    geometries.forEach(value => value.dispose()); materials.forEach(value => value.dispose()); textures.forEach(value => value.dispose());
    if (root === object) root = null;
  }
  function frame(view = 'front') {
    if (!root) return;
    const container = $('reconstructionViewport'), width = Math.max(container.clientWidth, 200), height = Math.max(container.clientHeight, 280);
    renderer.setSize(width, height, false); camera.aspect = width / height;
    const bounds = new THREE.Box3().setFromObject(root), center = bounds.getCenter(new THREE.Vector3());
    const radius = Math.max(...bounds.getSize(new THREE.Vector3()).toArray(), .01);
    const direction = { front: [0,.05,1], side: [1,.05,0], back: [0,.05,-1], top: [0,1,.001] }[view];
    const distance = radius / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.4 * Math.max(1, 1/camera.aspect);
    controls.target.copy(center); camera.position.copy(center).addScaledVector(new THREE.Vector3(...direction).normalize(), distance);
    camera.near = Math.max(.0001, distance/1000); camera.far = Math.max(20, distance*10); camera.updateProjectionMatrix(); controls.update();
  }
  function ensureRenderer() {
    if (renderer) return;
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.toneMapping = THREE.ACESFilmicToneMapping;
    scene = new THREE.Scene(); scene.background = new THREE.Color(0xf4f5f0); scene.environment = environment; addLights(scene);
    camera = new THREE.PerspectiveCamera(35, 1, .001, 20);
    controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
    $('reconstructionViewport').append(renderer.domElement);
    resizeObserver = new ResizeObserver(() => { if (dialog.open) frame(); }); resizeObserver.observe($('reconstructionViewport'));
  }
  async function open(job, slot, asset) {
    const ticket = ++serial; pendingJob = null; current = { job, slot, asset };
    ensureRenderer(); disposeRoot();
    $('reviewTitle').textContent = asset.name + ' · object ' + slot;
    $('reviewMeta').textContent = (job.provider === 'hunyuan' ? ('Hunyuan3D' + ' ? ' + job.used_images + ' images ? ') : '') + asset.faces.toLocaleString('en-US') + ' faces';
    $('reviewHistory').disabled = false;
    $('reviewStatus').textContent = 'Loading preview…';
    $('reviewAccept').disabled = true; $('reviewRetry').disabled = true;
    const key = slot + ':' + job.source_id, items = history.get(key) || [];
    if (!items.some(item => item.job.id === job.id)) items.push({ job, slot, asset });
    history.set(key, items);
    $('reviewHistory').replaceChildren(...items.map((item, index) => new Option('Variant ' + (index+1) + ' · ' + (item.job.provider === 'hunyuan' ? 'Hunyuan3D' : item.asset.name), item.job.id)));
    $('reviewHistory').value = job.id;
    if (!dialog.open) dialog.showModal();
    const response = await api('assets/' + asset.id + '/model.glb');
    const parsed = await loader.parseAsync(await response.arrayBuffer(), '');
    if (ticket !== serial || !dialog.open) { disposeRoot(parsed.scene); return; }
    root = parsed.scene; scene.add(root); frame();
    renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
    $('reviewAccept').textContent = 'Use as ' + slot;
    $('reviewAccept').disabled = false; $('reviewRetry').disabled = true;
    $('reviewHistory').disabled = false;
    $('reviewStatus').textContent = 'Check the shape, openings, and all views before using the model. A valid file does not guarantee an accurate shape.';
  }
  $('reviewAccept').onclick = safe(async () => {
    if (!current || !root || pendingJob) return;
    const selected = current;
    $('reviewAccept').disabled = true;
    try { await loadAsset(selected.slot, selected.asset, null, true, false, root); root = null; dialog.close(); toast('Selected model assigned to ' + selected.slot + '.'); }
    finally { $('reviewAccept').disabled = false; }
  });
  $('reviewRetry').disabled = true;
  $('reviewHistory').onchange = safe(async () => {
    if (!current || pendingJob) return;
    const item = history.get(current.slot + ':' + current.job.source_id)?.find(value => value.job.id === $('reviewHistory').value);
    if (item) await open(item.job, item.slot, item.asset);
  });
  $('reviewDiagnostics').onclick = safe(() => current && download('jobs/' + current.job.id + '/reconstruction.zip', 'reconstruction-' + current.job.id + '.zip'));
  $('reviewDownload').onclick = safe(() => current && download('assets/' + current.asset.id + '/model.glb', current.asset.name + '.glb'));
  $('reviewKeepScene').onclick = () => dialog.close();
  dialog.querySelectorAll('[data-review-view]').forEach(button => button.onclick = () => frame(button.dataset.reviewView));
  dialog.addEventListener('close', () => {
    ++serial; current = null;
    if (renderer) renderer.setAnimationLoop(null);
    disposeRoot();
    if (renderer) {
      controls.dispose(); resizeObserver.disconnect();
      renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove();
      renderer = null; scene = null;
    }
  });
  function failed(job) {
    if (pendingJob !== job.id) return;
    pendingJob = null;
    if (!dialog.open) return;
    $('reviewHistory').disabled = false;
    $('reviewAccept').disabled = !root; $('reviewRetry').disabled = true;
    $('reviewHistory').disabled = false;
    $('reviewStatus').textContent = 'Could not create a new variant: ' + job.message + '. The preview and current scene are preserved.';
  }
  return { open, failed };
}
