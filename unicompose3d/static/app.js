import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { createCompositionController } from './composition-ui.js';
import { createInputPreparation } from './input-ui.js';
import { createReconstructionReview } from './reconstruction-ui.js';

const $ = id => document.getElementById(id);
const $$ = selector => [...document.querySelectorAll(selector)];
const state = { A: null, B: null, objects: {}, previews: {}, selected: 'B', view: 'all', source: null, uploadSlot: 'A', kind: 'image', anchors: { A: null, B: null }, hints: { A_end: null, B_end: null }, pick: null, sculptPoint: null, sculptHistory: [], jobTimers: new Map(), busy: false, selectionVersion: 0 };
let compositionUI, providerInitialized = false;
let token = new URLSearchParams(location.hash.slice(1)).get('token') || sessionStorage.getItem('tryon_token') || '';
if (token) { sessionStorage.setItem('tryon_token', token); history.replaceState(null, '', location.pathname); }
let toastTimer;
function toast(message, error = false) { $('toast').textContent = message; $('toast').classList.remove('hidden'); $('toast').classList.toggle('error', error); clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').classList.add('hidden'), error ? 13000 : 6000); }
function safe(callback) { return async (...args) => { try { await callback(...args); } catch (error) { console.error(error); toast(error.message || 'The operation did not complete.', true); } }; }
async function api(path, options = {}) {
  const headers = { Authorization: `Bearer ${token}`, ...options.headers };
  if (options.body && !(options.body instanceof FormData) && typeof options.body !== 'string') { headers['Content-Type'] = 'application/json'; options.body = JSON.stringify(options.body); }
  const response = await fetch(`/api/${path}`, { ...options, headers });
  if (!response.ok) {
    let detail; try { detail = (await response.json()).detail; } catch { detail = `HTTP ${response.status}`; }
    if (response.status === 401) $('settingsDialog').showModal();
    if (response.status === 404 && (!detail || detail === 'Not Found' || detail === 'HTTP 404')) {
      detail = 'Studio HTTP 404 at ' + path.split('/')[0] + ': the route or session data is no longer available. Reopen Studio using the matching notebook/source.';
    }
    throw new Error(typeof detail === 'string' ? detail : 'Invalid parameters. Check the input data.');
  }
  return response;
}
async function json(path, options) { return (await api(path, options)).json(); }
function post(path, body) { return json(path, { method: 'POST', body }); }
async function download(path, filename, body) {
  const response = await api(path, body ? { method: 'POST', body } : {});
  const blob = await response.blob(), url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 30000);
}
function timingSummary(t = {}) {
  const items = [];
  const loads = (t.shape_load_s || 0) + (t.paint_load_s || 0);
  if (loads) items.push('model loading ' + Math.round(loads) + 's');
  if (t.shape_s != null) items.push('shape ' + Math.round(t.shape_s) + 's');
  if (t.paint_total_s != null) items.push('texture ' + Math.round(t.paint_total_s) + 's');
  if (t.uv_unwrap_s != null) items.push('UV unwrap ' + Math.round(t.uv_unwrap_s) + 's');
  if (t.studio_normalize_s != null) items.push('Studio preparation ' + Math.round(t.studio_normalize_s) + 's');
  return items.length ? '\n' + items.join(' · ') : '';
}
function loading(show, text = 'Processing…') { $('loadingText').textContent = text; $('loadingOverlay').classList.toggle('hidden', !show); }
function setStatus(text) { $('sceneStatus').textContent = text; }
function modal(id) { if (!$(id).open) $(id).showModal(); }
$$('[data-close]').forEach(button => button.onclick = () => button.closest('dialog').close());
$$('dialog').forEach(dialog => dialog.addEventListener('click', event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close(); } }));

// Main viewer: actual mesh geometry and independently editable scene nodes.
const viewport = $('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.15; viewport.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.005, 200);
camera.position.set(1.7, 1.15, 2.1);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.dampingFactor = 0.08; controls.target.set(0, 0.35, 0); controls.minDistance = 0.05; controls.maxDistance = 40;
const environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture; scene.environment = environment;
function addLights(target) {
  target.add(new THREE.HemisphereLight(0xffffff, 0x77796c, 2.2));
  const key = new THREE.DirectionalLight(0xfff7e9, 3.5); key.position.set(3, 5, 4); key.castShadow = true; key.shadow.mapSize.set(1024, 1024); key.shadow.camera.left = -3; key.shadow.camera.right = 3; key.shadow.camera.top = 3; key.shadow.camera.bottom = -3; key.shadow.normalBias = 0.01; key.shadow.bias = -0.0001; target.add(key);
  const fill = new THREE.DirectionalLight(0xe7f1ff, 1.5); fill.position.set(-3, 2, -2); target.add(fill);
}
addLights(scene);
const grid = new THREE.GridHelper(12, 60, 0xc0c8b6, 0xd7dece); grid.material.transparent = true; grid.material.opacity = 0.55; scene.add(grid);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(50, 50), new THREE.ShadowMaterial({ opacity: 0.10 })); ground.rotation.x = -Math.PI / 2; ground.position.y = -0.002; ground.receiveShadow = true; scene.add(ground);
const gizmo = new TransformControls(camera, renderer.domElement); gizmo.setSize(0.8); scene.add(gizmo.getHelper());
gizmo.addEventListener('dragging-changed', event => controls.enabled = !event.value);
gizmo.addEventListener('objectChange', () => { if (gizmo.object) gizmo.object.scale.clampScalar(0.001, 1000); updateFields(); updateMarkers(); compositionUI?.markEdited(); });
const draco = new DRACOLoader(); draco.setDecoderPath('/vendor/draco/');
const loader = new GLTFLoader(); loader.setDRACOLoader(draco); loader.setMeshoptDecoder(MeshoptDecoder);
const markers = {};
function resize() { const width = viewport.clientWidth, height = viewport.clientHeight; renderer.setSize(width, height, false); camera.aspect = width / Math.max(height, 1); camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(viewport);
function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); for (const preview of Object.values(state.previews)) { preview.root.rotation.y += 0.003; preview.renderer.render(preview.scene, preview.camera); } }
animate();

function objectTransform(slot) { const object = state.objects[slot]; return object ? { position: object.position.toArray(), rotation: [object.rotation.x, object.rotation.y, object.rotation.z], scale: object.scale.toArray() } : { position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] }; }
function applyTransform(slot, transform) { const object = state.objects[slot]; if (!object) return; object.position.fromArray(transform.position); object.rotation.set(...transform.rotation, 'XYZ'); object.scale.fromArray(transform.scale); object.updateMatrixWorld(true); updateFields(); updateMarkers(); }
function initialTransform(slot, asset) { return { position: slot === 'A' ? [0, asset.extents[1] / 2, 0] : [0.65, asset.extents[1] * 0.175, 0], rotation: [0, 0, 0], scale: slot === 'A' ? [1, 1, 1] : [0.35, 0.35, 0.35] }; }
async function showGeometryInfo(slot, asset) {
  const element = $('geometryNote' + slot);
  element.textContent = ''; element.classList.add('hidden');
  try {
    const info = asset.geometry_profile ? asset : await (await api(`assets/${asset.id}/inspect`)).json();
    if (state[slot]?.id !== asset.id) return;
    asset.geometry_profile = info.geometry_profile;
    const observations = info.geometry_profile?.observations || [];
    element.textContent = observations.join(' ');
    element.classList.toggle('hidden', !observations.length);
  } catch { /* Optional local analysis must not prevent loading an existing mesh. */ }
}

async function loadAsset(slot, asset, transform = null, reframe = true, preserveSculptHistory = false, preparedScene = null) {
  loading(true, `Loading ${slot === 'A' ? 'base object' : 'product'}…`);
  try {
    let modelScene = preparedScene;
    if (!modelScene) {
      const response = await api(`assets/${asset.id}/model.glb`);
      modelScene = (await loader.parseAsync(await response.arrayBuffer(), '')).scene;
    }
    if (state.objects[slot]) { if (gizmo.object === state.objects[slot]) gizmo.detach(); scene.remove(state.objects[slot]); disposeObject(state.objects[slot]); }
    const root = new THREE.Group(); root.add(modelScene); root.name = slot; root.userData.slot = slot;
    root.traverse(child => { if (child.isMesh) { child.castShadow = true; child.receiveShadow = true; } });
    state.objects[slot] = root; state[slot] = asset; scene.add(root); state.selectionVersion++; compositionUI?.reset(); $('planInfo').classList.add('hidden');
    if (slot === 'B' && !preserveSculptHistory) { state.sculptHistory = []; $('sculptUndoBtn').disabled = true; }
    state.anchors[slot] = null; if (slot === 'B') { state.sculptPoint = null; $('sculptBtn').disabled = true; }
    state.hints[slot + '_end'] = null; updateHintStatus();
    if (markers[slot]) { scene.remove(markers[slot]); markers[slot].geometry.dispose(); markers[slot].material.dispose(); delete markers[slot]; }
    applyTransform(slot, transform || initialTransform(slot, asset));
    $('upload' + slot).classList.add('hidden'); $('summary' + slot).classList.remove('hidden'); $('badge' + slot).textContent = 'Ready'; $('badge' + slot).classList.add('ready'); $('name' + slot).textContent = asset.name; $('stats' + slot).textContent = `${asset.faces.toLocaleString('en-US')} faces · ${asset.has_texture ? 'có texture' : 'vertex colors'}`;
    void showGeometryInfo(slot, asset);
    createPreview(slot, root); updateAvailability(); setView(state.view); select(state.selected); updateAnchorStatus(); if (reframe) frameScene();
    setStatus('Interactive 3D scene ready');
  } finally { loading(false); }
}
function disposeObject(object) { const geometries = new Set(), materials = new Set(), textures = new Set(); object.traverse(child => { if (child.geometry) geometries.add(child.geometry); for (const material of Array.isArray(child.material) ? child.material : child.material ? [child.material] : []) { materials.add(material); for (const v of Object.values(material)) if (v?.isTexture) textures.add(v); } }); geometries.forEach(g => g.dispose()); materials.forEach(m => m.dispose()); textures.forEach(t => t.dispose()); }
function createPreview(slot, root) {
  const old = state.previews[slot]; if (old) { old.renderer.dispose(); old.renderer.forceContextLoss(); old.renderer.domElement.remove(); }
  const r = new THREE.WebGLRenderer({ antialias: true, alpha: true }); r.setSize(68, 68); r.setPixelRatio(Math.min(devicePixelRatio, 2)); r.toneMapping = THREE.ACESFilmicToneMapping;
  const s = new THREE.Scene(); s.environment = environment; s.add(new THREE.HemisphereLight(0xffffff, 0x777777, 3)); const light = new THREE.DirectionalLight(0xffffff, 3); light.position.set(2, 3, 4); s.add(light);
  const clone = root.children[0].clone(true); s.add(clone); const c = new THREE.PerspectiveCamera(35, 1, 0.01, 20); c.position.set(1.3, 0.8, 1.7); c.lookAt(0, 0, 0); $('preview' + slot).replaceChildren(r.domElement); state.previews[slot] = { renderer: r, scene: s, camera: c, root: clone };
}
function updateAvailability() { const both = !!(state.A && state.B); for (const id of ['composeBtn', 'saveBtn', 'exportBtn']) $(id).disabled = !both || state.busy; $('composeMode').disabled = state.busy; $('agentRounds').disabled = state.busy; $('candidateCount').disabled = state.busy || $('composeMode').value !== 'evaluated'; $('candidatesBtn').disabled = state.busy; $('projectExportBtn').disabled = !both; $('emptyState').classList.toggle('hidden', !!(state.A || state.B)); $('viewerLabel').classList.toggle('hidden', !(state.A || state.B)); }
function select(slot) { state.selected = slot; $$('[data-select]').forEach(button => button.classList.toggle('selected', button.dataset.select === slot)); if (state.objects[slot] && state.objects[slot].visible && !state.pick) gizmo.attach(state.objects[slot]); else gizmo.detach(); updateFields(); }
function setView(view) { state.view = view; for (const slot of ['A', 'B']) if (state.objects[slot]) state.objects[slot].visible = view === 'all' || view === slot; $$('[data-view]').forEach(button => button.classList.toggle('selected', button.dataset.view === view)); $('viewerLabel').textContent = view === 'all' ? 'Showing both objects' : `Showing object ${view} only`; select(state.selected); updateMarkers(); }
function visibleBounds() { const box = new THREE.Box3(); for (const root of Object.values(state.objects)) if (root.visible) box.union(new THREE.Box3().setFromObject(root)); return box; }
function frameScene(direction) { const box = visibleBounds(); if (box.isEmpty()) return; const center = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3()), radius = Math.max(size.x, size.y, size.z, 0.2); const aspectFactor = Math.max(1, 1 / camera.aspect); const distance = radius / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.6 * aspectFactor; const dir = direction || new THREE.Vector3(1.1, 0.65, 1.6).normalize(); controls.target.copy(center); camera.position.copy(center).addScaledVector(dir, distance); camera.near = Math.max(0.001, distance / 1000); camera.far = Math.max(200, distance * 10); camera.updateProjectionMatrix(); controls.update(); camera.updateMatrixWorld(true); }

for (const [key, label, step] of [['position', 'Position', 0.01], ['rotation', 'Rotation (°)', 1], ['scale', 'Scale', 0.01]]) {
  const group = document.createElement('div'); group.className = 'transform-group'; const heading = document.createElement('label'); heading.textContent = label; group.append(heading); const inputs = document.createElement('div'); inputs.className = 'xyz-inputs';
  ['X', 'Y', 'Z'].forEach((axis, index) => { const l = document.createElement('label'); l.textContent = axis; const input = document.createElement('input'); input.type = 'number'; input.step = step; input.id = `${key}${index}`; input.setAttribute('aria-label', `${label} ${axis}`); if (key === 'scale') { input.min = '0.001'; input.max = '1000'; }
    input.addEventListener('change', () => { const root = state.objects[state.selected]; if (!root) return; const value = Number(input.value); if (!Number.isFinite(value) || Math.abs(value) > 10000 || (key === 'scale' && (value < 0.001 || value > 1000))) { updateFields(); return; } root[key].setComponent ? root[key].setComponent(index, value) : (root.rotation[axis.toLowerCase()] = THREE.MathUtils.degToRad(value)); root.updateMatrixWorld(true); updateMarkers(); compositionUI?.markEdited(); });
    l.append(input); inputs.append(l); }); group.append(inputs); $('transformFields').append(group);
}
function updateFields() { const transform = objectTransform(state.selected); for (const key of ['position', 'rotation', 'scale']) transform[key].forEach((v, i) => { const input = $(`${key}${i}`); if (document.activeElement !== input) input.value = (key === 'rotation' ? THREE.MathUtils.radToDeg(v) : v).toFixed(key === 'rotation' ? 1 : 3); input.disabled = !state.objects[state.selected]; }); }
$$('[data-select]').forEach(button => button.onclick = () => select(button.dataset.select));
$$('[data-mode]').forEach(button => button.onclick = () => { gizmo.setMode(button.dataset.mode); $$('[data-mode]').forEach(b => b.classList.toggle('selected', b === button)); });
$$('[data-view]').forEach(button => button.onclick = () => { cancelPick(); setView(button.dataset.view); frameScene(); });
$$('[data-camera]').forEach(button => button.onclick = () => frameScene({ front: new THREE.Vector3(0, 0.05, 1).normalize(), side: new THREE.Vector3(1, 0.05, 0).normalize(), top: new THREE.Vector3(0, 1, 0.001).normalize() }[button.dataset.camera]));
$('frameBtn').onclick = () => frameScene(); $('gridBtn').onclick = () => { grid.visible = !grid.visible; $('gridBtn').classList.toggle('selected', grid.visible); }; $('rotateBtn').onclick = () => { controls.autoRotate = !controls.autoRotate; $('rotateBtn').classList.toggle('selected', controls.autoRotate); };
$('resetBtn').onclick = () => { if (state[state.selected]) { applyTransform(state.selected, initialTransform(state.selected, state[state.selected])); compositionUI?.markEdited(); } };
window.addEventListener('keydown', event => { if (/INPUT|TEXTAREA|SELECT/.test(event.target.tagName) || $$('dialog[open]').length) return; const mode = { w: 'translate', e: 'rotate', r: 'scale' }[event.key.toLowerCase()]; if (mode) document.querySelector(`[data-mode="${mode}"]`).click(); if (event.key.toLowerCase() === 'f') frameScene(); if (event.key === 'Escape') cancelPick(); });

// Hunyuan3D is the single reconstruction backend.
const providerSelect=document.createElement('select');providerSelect.id='reconstructionProvider';providerSelect.hidden=true;
providerSelect.add(new Option('Hunyuan3D','hunyuan'));$('paidSection').prepend(providerSelect);
const reconstructionNote=document.createElement('p');reconstructionNote.id='reconstructionNote';reconstructionNote.className='small-note';providerSelect.after(reconstructionNote);
const textureLabel=document.createElement('label');textureLabel.className='check-row';
textureLabel.innerHTML='<input id="qualityTexture" type="checkbox" checked>Generate texture';
reconstructionNote.after(textureLabel);
const reconstructionReview=createReconstructionReview({api,post,safe,loadAsset,watchJob,toast,download,loader,environment,addLights});
function selectedInputViews() {
 return state.kind==='image'?(state.source?{front:state.source.images[0].file}:{}):
 Object.fromEntries(['front','left','back','right'].map(view=>[view,$('assign_'+view)?.value]).filter(([,value])=>value));
}
function selectedInputFiles() {
 return state.source?(state.kind==='image'?[state.source.images[0].file]:
 [...document.querySelectorAll('#sourceGallery input:checked')].map(input=>input.value)):[];
}
const inputPreparation=createInputPreparation({state,api,post,safe,onChange:updateGenerationUI,getViews:selectedInputViews,getSelectedFiles:selectedInputFiles});
function updateGenerationUI() {
 inputPreparation.update('hunyuan');
 const files=selectedInputFiles(),views=selectedInputViews();
 let ready=true;try{if(state.source)inputPreparation.options('hunyuan');}catch{ready=false;}
 const valid=files.length>0&&files.length<=4&&!!views.front&&new Set(Object.values(views)).size===files.length&&files.every(file=>Object.values(views).includes(file));
 const installed=state.backendReady?.hunyuan!==false;
 reconstructionNote.textContent=state.kind==='image'?'Segment and review the object before reconstruction.':'Select up to four images and assign the view of each image.';
 if(!installed)reconstructionNote.textContent+=' Hunyuan3D is not installed; run the notebook setup cell.';
 $('generateBtn').textContent='Generate 3D model';
 $('generateBtn').disabled=!installed||!state.source||inputPreparation.busy||!ready||!valid;
 $('advancedViewAssignments').classList.toggle('hidden',state.kind==='image');
 for(const view of ['left','back','right'])if($('assign_'+view))$('assign_'+view).disabled=inputPreparation.busy;
}
providerSelect.onchange=updateGenerationUI;

function openUpload(slot) { state.uploadSlot = slot; state.source = null; $('uploadEyebrow').textContent = `OBJECT ${slot} · ${slot === 'A' ? 'BASE' : 'PRODUCT'}`; $('objectName').value = ''; $('sourceFiles').value = ''; $('sourceStatus').textContent = ''; $('sourceStatus').classList.remove('error'); $('viewPicker').classList.add('hidden'); $('paidSection').classList.add('hidden'); $('jobProgress').classList.add('hidden'); $('forceNew').checked = false; $('generateBtn').disabled = true; $('prepareBtn').disabled = false; $('prepareBtn').textContent = 'Prepare input'; setKind('image'); updateGenerationUI(); modal('uploadDialog'); }
function setKind(kind) { state.kind = kind; state.source = null; $('viewPicker').classList.add('hidden'); $('paidSection').classList.add('hidden'); $('sourceFiles').value = ''; $('sourceFiles').multiple = kind === 'multiview'; $('sourceFiles').accept = { image: '.png,.jpg,.jpeg,.webp', multiview: '.png,.jpg,.jpeg,.webp', video: '.mp4,.mov,.webm,.avi', model: '.glb,.gltf,.obj,.stl,.ply' }[kind]; $('filePrompt').textContent = { image: 'Select a clear object image', multiview: 'Select multiple views of the same object', video: 'Select a video of a stationary object', model: 'Select an existing 3D model' }[kind]; $('fileHint').textContent = { image: 'PNG, JPG, WEBP · up to 20 MB', multiview: 'Up to four views · same object, different angles', video: 'Up to 150 MB, three minutes · choose views after frame extraction', model: 'GLB recommended · embedded GLTF, OBJ, STL, PLY · 150 MB' }[kind]; $$('[data-kind]').forEach(b => b.classList.toggle('selected', b.dataset.kind === kind)); }
$$('[data-kind]').forEach(button => button.onclick = () => setKind(button.dataset.kind));
$('sourceFiles').onchange = () => { state.source = null; $('viewPicker').classList.add('hidden'); $('paidSection').classList.add('hidden'); $('jobProgress').classList.add('hidden'); $('sourceStatus').textContent = 'Click Prepare input to validate the selected files.'; $('generateBtn').disabled = true; };
for (const slot of ['A', 'B']) $('upload' + slot).onclick = () => openUpload(slot);
$$('[data-change]').forEach(button => button.onclick = () => openUpload(button.dataset.change));
$$('[data-preview]').forEach(button => button.onclick = () => { cancelPick(); setView(button.dataset.preview); frameScene(); });
$$('[data-download]').forEach(button => button.onclick = safe(async () => { const a = state[button.dataset.download]; if (a) await download(`assets/${a.id}/model.glb`, `${button.dataset.download}.glb`); }));
let preparedBlobUrls = [];
$('prepareBtn').onclick = safe(async () => {
  const files = [...$('sourceFiles').files]; if (!files.length) throw new Error('Select an input file.');
  state.source = null; $('viewPicker').classList.add('hidden'); $('paidSection').classList.add('hidden');
  $('prepareBtn').disabled = true; $('sourceFiles').disabled = true; $$('[data-kind]').forEach(b => b.disabled = true); $('sourceStatus').textContent = 'Loading and preparing data…'; $('sourceStatus').classList.remove('error');
  try {
    const form = new FormData(); form.append('kind', state.kind); form.append('name', $('objectName').value.trim() || (state.uploadSlot === 'A' ? 'Base object' : 'Product')); files.forEach(file => form.append('files', file));
    const source = await json('sources', { method: 'POST', body: form }); state.source = source;
    if (source.asset) { await loadAsset(state.uploadSlot, source.asset); $('uploadDialog').close(); toast('3D model imported without API credits.'); return; }
    preparedBlobUrls.forEach(URL.revokeObjectURL); preparedBlobUrls = []; $('sourceGallery').replaceChildren(); $('viewAssignments').replaceChildren();
    for (const [i, item] of source.images.entries()) { const response = await api(`sources/${source.id}/${item.file}`); const url = URL.createObjectURL(await response.blob()); preparedBlobUrls.push(url); const figure = document.createElement('div'); figure.className = 'source-thumb'; const image = document.createElement('img'); image.src = url; image.alt = `Image ${i + 1}`; const label = document.createElement('span'); label.textContent = `#${i + 1} · ${item.time !== undefined ? item.time + 's' : item.original_name || item.file}`; figure.append(image, label); if (state.kind !== 'image') { const check = document.createElement('input'); check.type = 'checkbox'; check.checked = true; check.value = item.file; check.setAttribute('aria-label', `Use image ${i + 1}`); check.onclick = event => event.stopPropagation(); check.onchange = () => inputPreparation.invalidate(); figure.append(check); } $('sourceGallery').append(figure); }
    if (state.kind !== 'image') {
      for (const [view, label] of [['front', 'Front *'], ['left', 'Left'], ['back', 'Back'], ['right', 'Right']]) { const wrapper = document.createElement('label'); wrapper.textContent = label; const select = document.createElement('select'); select.id = `assign_${view}`; select.add(new Option('— Unused —', '')); source.images.forEach((item, i) => select.add(new Option(`#${i + 1}${item.time !== undefined ? ' · ' + item.time + 's' : ''}`, item.file))); if (view === 'front') select.value = source.images[0].file; wrapper.append(select); $('viewAssignments').append(wrapper); }
    }
    $('viewPicker').classList.remove('hidden'); $('paidSection').classList.remove('hidden'); $('sourceStatus').textContent = 'Images received. Segment and inspect each selected view before reconstruction.';
    await inputPreparation.render(source);
    for (const select of $('viewAssignments').querySelectorAll('select')) select.onchange = () => inputPreparation.invalidate();
    updateGenerationUI();
  } catch (error) { $('sourceStatus').textContent = error.message; $('sourceStatus').classList.add('error'); throw error; } finally { $('prepareBtn').disabled = false; $('sourceFiles').disabled = false; $$('[data-kind]').forEach(b => b.disabled = false); }
});
$('generateBtn').onclick=safe(async()=>{
 if(!state.source)return;
 const views=selectedInputViews(),selectedImages=selectedInputFiles();
 if(!views.front||selectedImages.length>4||selectedImages.length!==Object.keys(views).length||
 !selectedImages.every(file=>Object.values(views).includes(file)))throw new Error('Select up to four images and review their view assignments.');
 $('generateBtn').disabled=true;$('jobProgress').classList.remove('hidden');$('jobMessage').textContent='Starting reconstruction…';
 try{
  const job=await post('generate',{source_id:state.source.id,name:$('objectName').value.trim()||state.source.name,
   views,selected_images:selectedImages,provider:'hunyuan',force_new:$('forceNew').checked,
   texture:$('qualityTexture').checked,...inputPreparation.options('hunyuan')});
  watchJob(job.id,state.uploadSlot,true);toast(job.reused?'Reopened the existing result.':'Reconstruction is running. Progress is available below.');
 }catch(error){updateGenerationUI();throw error;}
});
function watchJob(id, slot, inUpload = false) {
  if (state.jobTimers.has(id)) return;
  let consecutiveErrors = 0;
  const tick = async () => {
    try {
      const job = await json(`jobs/${id}`); consecutiveErrors = 0;
      if (inUpload) { $('jobBar').value = job.progress || 0; $('jobMessage').textContent = job.message + timingSummary(job.timings); $('jobTaskId').textContent = job.task_id ? `Task ID: ${job.task_id}` : ''; }
      setStatus('Hunyuan3D: '+(job.name||slot)+' ? '+(job.progress||0)+'%');
      if (job.status === 'success') {
        state.jobTimers.delete(id);
        const assets = await json('assets'), asset = assets.find(a => a.id === job.asset_id);
        if (inUpload && $('uploadDialog').open) $('uploadDialog').close();
        if (slot && asset) {
          if (job.provider === 'hunyuan') await reconstructionReview.open(job, slot, asset);
          else await loadAsset(slot, asset);
        }
        toast('The 3D model is ready in the library.');
        if ($('jobsDialog').open) await openJobs(false);
        return;
      }
      if (['error', 'paused', 'uncertain'].includes(job.status)) { reconstructionReview.failed(job); state.jobTimers.delete(id); toast(job.message, true); if (inUpload) updateGenerationUI(); setStatus('Job needs review; existing results are retained'); if ($('jobsDialog').open) await openJobs(false); return; }
    } catch (error) { consecutiveErrors++; if (consecutiveErrors >= 5) { state.jobTimers.delete(id); toast('Connection lost. After reconnecting, open 3D jobs instead of creating another job.', true); return; } }
    state.jobTimers.set(id, setTimeout(tick, 3500));
  };
  state.jobTimers.set(id, true); tick();
}

// Image-conditioned placement. Capture cameras use the same canonical coordinates
// as Python's raycasting, regardless of the editable world transforms.
const captureRenderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true }); captureRenderer.setSize(640, 640); captureRenderer.toneMapping = THREE.ACESFilmicToneMapping;
async function captureModels() {
  captureRenderer.setSize(640, 640);
  const result = [];
  for (const slot of ['A', 'B']) {
    const s = new THREE.Scene(); s.background = new THREE.Color(0xf4f5f0); s.environment = environment; addLights(s); const clone = state.objects[slot].children[0].clone(true); s.add(clone);
    for (const point of [[0, 0.12, 2.2], [2.1, 0.2, 0.05], [0, 0.12, -2.2], [1.5, 1.4, 1.5]]) { const c = new THREE.PerspectiveCamera(35, 1, 0.01, 30); c.position.set(...point); c.lookAt(0, 0, 0); c.updateProjectionMatrix(); c.updateMatrixWorld(true); captureRenderer.render(s, c); result.push({ slot, image: captureRenderer.domElement.toDataURL('image/png'), projection: c.projectionMatrix.toArray(), world: c.matrixWorld.toArray() }); }
  }
  return result;
}
function getHints() { return { A_point: state.anchors.A, B_point: state.anchors.B, A_end: state.hints.A_end, B_end: state.hints.B_end }; }
compositionUI = createCompositionController({ state, post, api, download, captureModels, captureRenderer, environment, addLights, objectTransform, applyTransform, loading, toast, setStatus, updateAvailability, cancelPick, setView, select, frameScene, getHints });
$('composeBtn').onclick = safe(() => compositionUI.run());
$('candidatesBtn').onclick = () => compositionUI.open();
$('reportBtn').onclick = safe(() => compositionUI.report());
$('dialogReportBtn').onclick = safe(() => compositionUI.report());
$('instruction').addEventListener('input', () => compositionUI.markEdited());

function startPick(slot, sculpt = false, kind = null) { if (!state.objects[slot]) { toast(`Add object ${slot} first.`, true); return; } state.pick = { slot, sculpt, kind }; controls.autoRotate = false; $('rotateBtn').classList.remove('selected'); gizmo.detach(); setView(slot); gizmo.detach(); frameScene(); viewport.classList.add('picking'); $('viewerLabel').textContent = `Click the surface of ${slot} to select ${kind ? 'a second point defining direction from the selected region' : sculpt ? 'sculpt region' : 'target region'} · Esc to cancel`; toast($('viewerLabel').textContent); }
function cancelPick() { state.pick = null; viewport.classList.remove('picking'); select(state.selected); }
$('pickABtn').onclick = () => startPick('A'); $('pickBBtn').onclick = () => startPick('B'); $('sculptPickBtn').onclick = () => startPick('B', true);
let pointerStart;
renderer.domElement.addEventListener('pointerdown', event => { pointerStart = [event.clientX, event.clientY]; });
renderer.domElement.addEventListener('pointerup', event => {
  if (!state.pick || !pointerStart || Math.hypot(event.clientX - pointerStart[0], event.clientY - pointerStart[1]) > 5) return;
  const rect = renderer.domElement.getBoundingClientRect(), mouse = new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1), ray = new THREE.Raycaster(); ray.setFromCamera(mouse, camera); const slot = state.pick.slot, root = state.objects[slot]; const hit = ray.intersectObject(root, true).find(h => h.object.isMesh); if (!hit) return;
  const local = root.worldToLocal(hit.point.clone());
  if (state.pick.kind) { state.hints[state.pick.kind] = local.toArray(); updateHintStatus(); toast('Direction selected. Compose again to use it.'); }
  else if (state.pick.sculpt) { state.sculptPoint = local.toArray(); $('sculptBtn').disabled = false; toast('Selected a sculpt region on B.'); }
  else { state.anchors[slot] = local.toArray(); if (markers[slot]) scene.remove(markers[slot]); const marker = new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 12), new THREE.MeshBasicMaterial({ color: slot === 'A' ? 0x29a582 : 0xe26d43, depthTest: false })); marker.renderOrder = 999; markers[slot] = marker; scene.add(marker); updateMarkers(); updateAnchorStatus(); toast(`Selected anchor ${slot}.`); }
  compositionUI?.markEdited(); cancelPick();
});
function updateMarkers() { for (const slot of ['A', 'B']) if (markers[slot] && state.anchors[slot] && state.objects[slot]) { state.objects[slot].updateMatrixWorld(true); markers[slot].position.copy(state.objects[slot].localToWorld(new THREE.Vector3(...state.anchors[slot]))); markers[slot].visible = state.objects[slot].visible; } }
function updateAnchorStatus() { $('anchorStatus').textContent = `A: ${state.anchors.A ? 'selected ✓' : 'not selected'} · B: ${state.anchors.B ? 'selected ✓' : 'not selected'}`; $('alignBtn').disabled = !(state.anchors.A && state.anchors.B); }
function updateHintStatus() { $('hintStatus').textContent = `A direction: ${state.hints.A_end ? 'specified ✓' : 'AI inference'} · B: ${state.hints.B_end ? 'specified ✓' : 'AI inference'}`; }
$('axisABtn').onclick = () => startPick('A', false, 'A_end');
$('axisBBtn').onclick = () => startPick('B', false, 'B_end');
$('clearHintsBtn').onclick = () => { state.anchors = { A: null, B: null }; state.hints = { A_end: null, B_end: null }; for (const marker of Object.values(markers)) marker.visible = false; updateAnchorStatus(); updateHintStatus(); compositionUI.markEdited(); toast('Cleared target hints. The AI will infer the placement.'); };
$('alignBtn').onclick = safe(async () => { const result = await post('align', { base_id: state.A.id, product_id: state.B.id, base_point: state.anchors.A, product_point: state.anchors.B, base_transform: objectTransform('A'), product_transform: objectTransform('B') }); applyTransform('B', result.transform); compositionUI.markEdited(); cancelPick(); setView('all'); frameScene(); toast('Aligned the two points. Rotate or scale to refine the placement.'); });
$('sculptBtn').onclick = safe(async () => { if (!state.sculptPoint || !state.B) return; const transform = objectTransform('B'), before = state.B; loading(true, 'Sculpting the product…'); try { const asset = await post('deform', { asset_id: before.id, point: state.sculptPoint, delta: [Number($('sculptX').value), Number($('sculptY').value), Number($('sculptZ').value)], radius: Number($('sculptRadius').value), name: before.name.replace(/ · sculpted$/, '') + ' · sculpted' }); state.sculptHistory.push({ asset: before, transform }); await loadAsset('B', asset, transform, false, true); $('sculptUndoBtn').disabled = false; toast('Sculpted model saved; the previous model remains in the library.'); } finally { loading(false); } });
$('sculptUndoBtn').onclick = safe(async () => { const previous = state.sculptHistory.pop(); if (previous) await loadAsset('B', previous.asset, previous.transform, false, true); $('sculptUndoBtn').disabled = !state.sculptHistory.length; });

function sceneRequest() { if (!state.A || !state.B) throw new Error('Both A and B are required to save or export a scene.'); return { base_id: state.A.id, product_id: state.B.id, base_transform: objectTransform('A'), product_transform: objectTransform('B'), title: $('sceneTitle').value.trim() || 'Composition scene', notes: $('instruction').value, hints: getHints() }; }
$('saveBtn').onclick = safe(async () => { await post('scenes', sceneRequest()); toast('Scene saved to the library. Download a project ZIP before ending the session.'); });
$('exportBtn').onclick = safe(async () => { loading(true, 'Exporting the textured GLB scene…'); try { await download('export', 'unicompose3d-scene.glb', sceneRequest()); toast('3D scene exported.'); } finally { loading(false); } });
$('projectExportBtn').onclick = safe(async () => download('project/export', 'unicompose3d-project.zip', sceneRequest()));
$('projectImportBtn').onclick = () => $('projectFile').click();
$('projectFile').onchange = safe(async () => { const file = $('projectFile').files[0]; if (!file) return; const form = new FormData(); form.append('file', file); const value = await json('project/import', { method: 'POST', body: form }); await openScene(value); $('projectFile').value = ''; toast('Project reopened without reconstruction API calls.'); });
async function openScene(value) { const assets = await json('assets'); const a = assets.find(asset => asset.id === value.base_id), b = assets.find(asset => asset.id === value.product_id); if (!a || !b) throw new Error('The scene is missing source models. Open an exported project ZIP.'); await loadAsset('A', a, value.base_transform, false); await loadAsset('B', b, value.product_transform, false); $('sceneTitle').value = value.title; $('instruction').value = value.notes || ''; const hints = value.hints || {}; state.anchors = { A: hints.A_point || null, B: hints.B_point || null }; state.hints = { A_end: hints.A_end || null, B_end: hints.B_end || null }; updateAnchorStatus(); updateHintStatus(); setView('all'); frameScene(); }
function row(title, subtitle, buttons) { const div = document.createElement('div'); div.className = 'library-row'; const info = document.createElement('div'); info.className = 'row-info'; const strong = document.createElement('strong'); strong.textContent = title; const small = document.createElement('small'); small.textContent = subtitle; info.append(strong, small); div.append(info); for (const [label, callback] of buttons) { const button = document.createElement('button'); button.className = 'button outline'; button.textContent = label; button.onclick = safe(callback); div.append(button); } return div; }
$('libraryBtn').onclick = safe(async () => { const [assets, scenes] = await Promise.all([json('assets'), json('scenes')]); $('assetList').replaceChildren(); $('sceneList').replaceChildren(); for (const asset of assets) $('assetList').append(row(asset.name, `${asset.faces.toLocaleString()} faces · ${asset.origin === 'demo' ? 'Procedural demo' : asset.origin === 'hunyuan' ? 'Hunyuan3D' : 'Imported or edited'}`, [['Use as A', async () => { await loadAsset('A', asset); $('libraryDialog').close(); }], ['Use as B', async () => { state.sculptHistory = []; await loadAsset('B', asset); $('libraryDialog').close(); }]])); for (const value of scenes) $('sceneList').append(row(value.title, new Date(value.created_at).toLocaleString('en-US'), [['Open', async () => { await openScene(value); $('libraryDialog').close(); }]])); if (!assets.length) $('assetList').textContent = 'No models yet. Add an object or open the demo.'; if (!scenes.length) $('sceneList').textContent = 'No saved scenes.'; modal('libraryDialog'); });
async function openJobs(show = true) {
  const jobs = await json('jobs'); $('jobList').replaceChildren();
  if (!jobs.length) $('jobList').textContent = 'No reconstruction jobs in this session.';
  for (const job of jobs) {
    const buttons = [];
    if (job.asset_id) {
      for (const slot of ['A', 'B']) buttons.push(['Use as ' + slot, async () => {
        const assets = await json('assets');
        await loadAsset(slot, assets.find(asset => asset.id === job.asset_id));
        $('jobsDialog').close();
      }]);
    } else if (['queued', 'running'].includes(job.status)) {
      buttons.push(['Follow progress', () => { watchJob(job.id, null); toast('Following the existing job.'); }]);
    }
    $('jobList').append(row((job.name || 'Reconstruct 3D') + ' ? ' + job.status,
      (job.message || '') + timingSummary(job.timings), buttons));
  }
  if (show) modal('jobsDialog');
}

async function demo() { const assets = await post('demo', {}); await loadAsset('A', assets.A, null, false); await loadAsset('B', assets.B, { position: [0.12, assets.A.extents[1] + assets.B.extents[1] * 0.15 + 0.002, 0.04], rotation: [0, 0, 0], scale: [0.3, 0.3, 0.3] }, false); setView('all'); frameScene(); toast('Demo scene opened without API calls.'); }
$('demoBtn').onclick = safe(demo); $('emptyDemoBtn').onclick = safe(demo); $('helpBtn').onclick = () => modal('helpDialog');
async function loadConfig(){
 const config=await json('config');state.searchBudget=config.composition_search_budget;state.backendReady={hunyuan:config.hunyuan_ready};providerSelect.value='hunyuan';updateGenerationUI();
 $('connectionDot').classList.toggle('ready',config.hunyuan_ready&&config.llm_ready);
 $('connectionInfo').replaceChildren();
 for(const [label,text]of [['3D reconstruction · Hunyuan3D',config.hunyuan_ready?'Ready':'Not installed'],
 ['Segmentation',config.segment_ready?'Ready':'Not installed'],
 ['Composition agent',config.llm_ready?'Connected · '+config.llm_model:'API key missing'],
 ['Version',config.version]]){
  const row=document.createElement('div');row.className='connection-row';
  const strong=document.createElement('strong');strong.textContent=label;
  const span=document.createElement('span');span.textContent=text;row.append(strong,span);$('connectionInfo').append(row);
 }
 return config;
}
$('settingsBtn').onclick = safe(async () => { await loadConfig(); modal('settingsDialog'); });
$('accessTokenBtn').onclick = safe(async () => { const candidate = $('accessTokenInput').value.trim(); if (!candidate) throw new Error('Enter the token from the notebook link or reopen the full URL.'); token = candidate; sessionStorage.setItem('tryon_token', token); await loadConfig(); $('accessTokenInput').value = ''; $('settingsDialog').close(); toast('Reconnected.'); });
window.addEventListener('error', event => { console.error(event.error); if (/WebGL|context|fetch|module/i.test(event.message)) toast('Viewer error. Check WebGL and the application connection.', true); });
updateFields(); updateAvailability();
safe(async () => { const config = await loadConfig(); if (!config.segment_ready || !state.backendReady[config.default_provider]) toast('Enable a GPU and install SAM/Hunyuan3D to segment and reconstruct. Imported GLBs and the demo remain available.'); const jobs = await json('jobs'); for (const job of jobs) if (['queued', 'running', 'uploading', 'submitting'].includes(job.status)) watchJob(job.id, null); })();






