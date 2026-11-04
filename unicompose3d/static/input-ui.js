// Common target selection for all object categories; server retains originals.
export function createInputPreparation({ state, api, post, safe, onChange, getViews, getSelectedFiles }) {
  const $ = id => document.getElementById(id);
  const section = document.createElement('section'); section.id = 'inputPreparation'; section.className = 'input-preparation hidden';
  section.innerHTML = `<h3>Select the object and remove its background</h3>
    <label class="field-label">Object to keep<input id="targetDescription" maxlength="200" placeholder="Example: my hand, the red chair"></label>
    <p class="small-note">Draw a box for fast segmentation. Optionally describe the object for AI localization, which uses the API. Keep all parts needed for composition.</p>
    <label class="field-label">Current image<select id="editSourceImage"></select></label>
    <div class="mask-tools"><button type="button" id="maskBox" class="button ghost">Box selection</button><button type="button" id="maskKeep" class="button ghost">Include point +</button><button type="button" id="maskRemove" class="button ghost">Exclude point −</button><button type="button" id="maskEraseBrush" class="button ghost">Remove background region</button><button type="button" id="maskErasePixels" class="button ghost">Erase brush</button><button type="button" id="maskRestoreBrush" class="button ghost">Restore brush</button><button type="button" id="maskUndo" class="button ghost">Undo brush</button><label>Region sensitivity <input id="maskRegionTolerance" type="range" min="3" max="45" value="18"></label><label>Brush size <input id="maskBrushSize" type="range" min="1" max="12" value="2"></label><button type="button" id="maskReset" class="button ghost">Clear guidance / let AI select again</button></div>
    <div class="mask-preview"><div><p>Original image · drag a box or click to add points</p><canvas id="maskCanvas"></canvas></div><div><p>Segmentation result · check details and openings</p><img id="cutoutPreview" alt="Segmented object image" hidden></div></div>
    <p id="maskStatus" class="small-note" role="status"></p>
    <div class="mask-tools"><button type="button" id="segmentCurrent" class="button dark">Segment current image</button><button type="button" id="segmentSelected" class="button dark">Segment all selected views</button><button type="button" id="segmentCancel" class="button ghost" hidden>Cancel AI / select manually</button></div>
    <label class="check-row"><input id="useCutouts" type="checkbox" checked>Use segmented images for reconstruction</label>
    <label class="check-row"><input id="confirmCutouts" type="checkbox">I reviewed every selected view and the masks preserve the intended object.</label><button id="exportCutouts" type="button" class="button outline full" disabled>Download segmented images (ZIP)</button><p class="small-note">PNG images preserve alpha and openings for reuse; this is not a 3D result.</p>`;
  $('viewPicker').append(section);
  const canvas = $('maskCanvas'), context = canvas.getContext('2d');
  let sourceImage, currentFile, mode = 'box', start, busy = false;
  let revision = 0, urls = [], hints = {}, stroke = [];
  let runToken = 0, activeJob = null, restoreControls = null;
  function unlock() { busy = false; if (restoreControls) restoreControls(); restoreControls = null; $('segmentCancel').hidden = true; onChange(); }
  function cancelActive(show = true) {
    runToken++; revision++; const previous = activeJob; activeJob = null; unlock();
    if (show) { mode = 'box'; $('maskStatus').textContent = 'Cancelled. Draw a box and click Segment current image.'; }
    if (previous) api(`segment-jobs/${previous}`, { method: 'DELETE' }).catch(() => {});
  }
  $('segmentCancel').onclick = () => cancelActive();
  const invalidate = () => { $('confirmCutouts').checked = false; onChange(); };
  const changedHints = () => { revision++; if (state.source?.cutouts) delete state.source.cutouts[currentFile]; $('cutoutPreview').hidden = true; invalidate(); };
  const position = event => { const rect = canvas.getBoundingClientRect(); return [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))]; };
  function draw(temporary) {
    if (!sourceImage) return;
    context.clearRect(0, 0, canvas.width, canvas.height); context.drawImage(sourceImage, 0, 0);
    const item = hints[currentFile] || {}; const box = temporary || item.box;
    context.lineWidth = Math.max(2, canvas.width / 300);
    if (box) { context.strokeStyle = '#266eff'; context.strokeRect(box[0] * canvas.width, box[1] * canvas.height, (box[2] - box[0]) * canvas.width, (box[3] - box[1]) * canvas.height); }
    (item.points || []).forEach((p, i) => { context.beginPath(); context.fillStyle = item.labels[i] ? '#21b855' : '#f33e42'; context.arc(p[0] * canvas.width, p[1] * canvas.height, Math.max(4, canvas.width / 130), 0, 2 * Math.PI); context.fill(); context.strokeStyle = '#fff'; context.stroke(); });
  }
  async function blobUrl(file) { const response = await api(`sources/${state.source.id}/${file}`); const url = URL.createObjectURL(await response.blob()); urls.push(url); return url; }
  async function preview() {
    const previewRevision = revision, previewFile = currentFile;
    const item = state.source?.cutouts?.[currentFile]; $('cutoutPreview').hidden = true;
    const input = state.source.images.find(i => i.file === currentFile);
    const warnings = [...(input?.quality?.warnings || []), ...(item?.warnings || [])];
    $('maskStatus').textContent = warnings.join(' ') || 'Draw a box or describe the object for AI selection.';
    if (item && item.target === $('targetDescription').value.trim()) {
      const url = await blobUrl(item.file);
      if (previewRevision !== revision || previewFile !== currentFile || state.source?.cutouts?.[currentFile] !== item) return;
      $('cutoutPreview').src = url; $('cutoutPreview').hidden = false;
      $('maskStatus').textContent = [item.explanation, ...warnings].join(' ');
    }
  }
  async function edit(file) {
    const activeRevision = ++revision; currentFile = file; $('editSourceImage').value = file;
    const image = new Image(); image.src = await blobUrl(file); await image.decode();
    if (activeRevision !== revision) return;
    sourceImage = image; canvas.width = image.naturalWidth; canvas.height = image.naturalHeight; draw(); await preview();
  }
  function drawStroke() {
    draw();
    if (!stroke.length) return;
    const radius = Number($('maskBrushSize').value) / 100 * Math.min(canvas.width, canvas.height);
    context.strokeStyle = context.fillStyle = mode !== 'restoreBrush' ? '#ef454588' : '#21b85588';
    context.lineWidth = radius * 2; context.lineCap = 'round'; context.lineJoin = 'round';
    context.beginPath(); stroke.forEach((p, i) => context[i ? 'lineTo' : 'moveTo'](p[0] * canvas.width, p[1] * canvas.height)); context.stroke();
    const p = stroke[stroke.length - 1]; context.beginPath(); context.arc(p[0] * canvas.width, p[1] * canvas.height, radius, 0, 2 * Math.PI); context.fill();
  }
  async function paint(strokes, undo = false) {
    const source = state.source, file = currentFile, item = source?.cutouts?.[file];
    if (busy) return;
    if (!item || item.target !== $('targetDescription').value.trim()) throw new Error('Segment first, then refine the mask with the brush.');
    busy = true; revision++; invalidate();
    $('maskStatus').textContent = 'Editing alpha on CPU…';
    try {
      const result = await post(`sources/${source.id}/mask-edit`, { filename: file, cutout_file: item.file, strokes, undo });
      if (state.source !== source) return;
      source.cutouts[file] = result;
      if (file === currentFile) { await preview(); $('maskStatus').textContent += undo ? ' Undone.' : ' Edited; review and confirm again.'; }
    } finally { busy = false; draw(); onChange(); }
  }
  canvas.onpointerdown = event => { if (busy || !currentFile || !sourceImage) return; canvas.setPointerCapture(event.pointerId); start = position(event); if (mode.endsWith('Brush')) { stroke = [start]; drawStroke(); return; } if (mode !== 'box') { const h = hints[currentFile] ||= { points: [], labels: [] }; if (h.points.length >= 32) return; h.points.push(start); h.labels.push(mode === 'keep' ? 1 : 0); changedHints(); draw(); } };
  canvas.onpointermove = event => { if (start && mode.endsWith('Brush')) { if (stroke.length < 511) stroke.push(position(event)); drawStroke(); return; } if (start && mode === 'box') { const end = position(event); draw([Math.min(start[0], end[0]), Math.min(start[1], end[1]), Math.max(start[0], end[0]), Math.max(start[1], end[1])]); } };
  canvas.onpointerup = safe(async event => { if (start && mode.endsWith('Brush')) { const points = [...stroke, position(event)]; start = null; stroke = []; await paint([{ points, radius: Number($('maskBrushSize').value) / 100, keep: mode === 'restoreBrush', mode: mode === 'regionBrush' ? 'region' : 'pixels', tolerance: Number($('maskRegionTolerance').value) }]); return; } if (start && mode === 'box') { const end = position(event); const box = [Math.min(start[0], end[0]), Math.min(start[1], end[1]), Math.max(start[0], end[0]), Math.max(start[1], end[1])]; if (box[2] - box[0] > 0.005 && box[3] - box[1] > 0.005) { hints[currentFile] = { box, points: [], labels: [] }; changedHints(); } } start = null; draw(); });
  $('cutoutPreview').onpointerup = safe(async event => {
    if (mode !== 'regionBrush' || busy) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const point = [(event.clientX-rect.left)/rect.width, (event.clientY-rect.top)/rect.height];
    await paint([{points:[point], radius:.005, keep:false, mode:'region',
                  tolerance:Number($('maskRegionTolerance').value)}]);
  });
  canvas.onpointercancel = () => { start = null; stroke = []; draw(); };
  for (const [id, value] of [['maskBox', 'box'], ['maskKeep', 'keep'], ['maskRemove', 'remove']]) $(id).onclick = () => { mode = value; $('maskStatus').textContent = value === 'box' ? 'Draw a box around the entire object to keep.' : 'Click the image to add a point to ' + (value === 'keep' ? 'keep.' : 'remove unwanted background or details.'); };
  for (const [id, value] of [['maskEraseBrush', 'regionBrush'], ['maskErasePixels', 'eraseBrush'], ['maskRestoreBrush', 'restoreBrush']]) $(id).onclick = () => {
    mode = value; if (mode === 'regionBrush') { $('maskStatus').textContent = 'Click a background area in either image to expand removal through the region, including multicolored openings. Lower sensitivity if it spreads too far. Undo is available.'; return; } $('maskStatus').textContent = 'Drag on the original image to ' + (value === 'eraseBrush' ? 'remove background, including inside openings.' : 'restore missing object parts.') + ' Applied directly without SAM or LLM.';
  };
  $('maskUndo').onclick = safe(() => paint([], true));
  $('maskReset').onclick = () => { hints[currentFile] = { points: [], labels: [] }; changedHints(); draw(); };
  $('editSourceImage').onchange = safe(() => edit($('editSourceImage').value));
  $('targetDescription').oninput = () => { hints = {}; invalidate(); draw(); $('cutoutPreview').hidden = true; };
  $('confirmCutouts').onchange = onChange;
  $('useCutouts').onchange = invalidate;
  $('exportCutouts').onclick = safe(async () => {
    const files = getSelectedFiles();
    const settings = options('hunyuan');
    const response = await api(`sources/${state.source.id}/export-cutouts`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ files, target: settings.target, cutout_files: settings.cutout_files,
                             confirmed: $('confirmCutouts').checked }) });
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement('a'); link.href = url;
    link.download = 'unicompose3d-cutouts-' + state.source.id + '.zip'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  });
  async function segment(files) {
    if (busy) return;
    const target = $('targetDescription').value.trim(); if (!target) throw new Error('Describe the object to keep.');
    const source = state.source, token = ++runToken;
    busy = true; invalidate();
    const controls = [...section.querySelectorAll('input, button, select'), ...document.querySelectorAll('#viewAssignments select, #sourceGallery input'), $('objectName'), $('sourceFiles'), $('prepareBtn'), ...document.querySelectorAll('[data-kind]')].filter(c => c.id !== 'segmentCancel');
    const disabled = controls.map(c => c.disabled); controls.forEach(c => c.disabled = true);
    restoreControls = () => controls.forEach((c, i) => c.disabled = disabled[i]);
    $('segmentCancel').hidden = false; $('segmentCancel').disabled = false;
    try {
      for (const [index, file] of files.entries()) {
        const h = hints[file] || { points: [], labels: [] };
        $('maskStatus').textContent = `Sending request · ${index + 1}/${files.length}`;
        const job = await post(`sources/${source.id}/segment-jobs`, { filename: file, target, box: h.box || (h.points.length ? [0, 0, 1, 1] : null), points: h.points, labels: h.labels });
        if (token !== runToken) { api(`segment-jobs/${job.id}`, { method: 'DELETE' }).catch(() => {}); return; }
        activeJob = job.id; let status = job;
        while (true) {
          if (token !== runToken) return;
          $('maskStatus').textContent = `${index + 1}/${files.length} · ${status.message} · ${status.elapsed_s || 0}s`;
          if (status.status === 'success') { $('maskStatus').textContent = 'Opening segmented image…'; break; }
          if (['error', 'cancelled'].includes(status.status)) throw new Error(status.message);
          await new Promise(resolve => setTimeout(resolve, 400));
          if (token !== runToken) return;
          status = await (await api(`segment-jobs/${job.id}`)).json();
        }
        if (token !== runToken || state.source !== source) return;
        const result = status.result;
        source.cutouts ||= {}; source.cutouts[file] = result;
        hints[file] = { box: result.box, points: result.points, labels: result.labels };
        activeJob = null;
      }
      if (token !== runToken) return;
      await edit(files[0]);
      const t = source.cutouts[files[0]]?.timings || {};
      $('maskStatus').textContent += ` Segmentation complete · locate ${t.locate_s ?? 0}s · encode ${t.image_encode_s ?? 0}s · mask ${t.sam_mask_s ?? 0}s. Review every view and confirm.`;
    } finally { if (token === runToken) { activeJob = null; unlock(); } }
  }
  $('segmentCurrent').onclick = safe(() => segment([currentFile]));
  $('segmentSelected').onclick = safe(() => { const files = getSelectedFiles(); if (!files.length) throw new Error('Select at least one image.'); return segment(files); });
  function options(provider) {
    const use = provider === 'hunyuan' || $('useCutouts').checked;
    const views = getViews(), files = getSelectedFiles();
    if (use) {
      if (busy || !$('confirmCutouts').checked) throw new Error('Segment and confirm all selected views before reconstruction.');
      for (const file of files) if (state.source?.cutouts?.[file]?.target !== $('targetDescription').value.trim()) throw new Error('Not all selected views have a valid segmentation.');
    }
    return { use_cutouts: use, confirm_cutouts: use && $('confirmCutouts').checked, target: $('targetDescription').value.trim(),
      cutout_files: Object.fromEntries(files.map(file => [file, state.source?.cutouts?.[file]?.file || ''])) };
  }
  return {
    options, invalidate,
    get busy() { return busy; },
    update(provider) { $('exportCutouts').disabled = busy || !$('confirmCutouts').checked || !getSelectedFiles().length; const required = provider === 'hunyuan'; if (required) $('useCutouts').checked = true; $('useCutouts').disabled = required || busy; },
    async render(source) {
      if (busy) cancelActive(false);
      revision++; hints = {}; start = null; sourceImage = null; urls.forEach(URL.revokeObjectURL); urls = [];
      $('targetDescription').value = source.name; $('confirmCutouts').checked = false; $('useCutouts').checked = true;
      $('editSourceImage').replaceChildren();
      source.images.forEach((item, i) => $('editSourceImage').add(new Option(`#${i + 1} · ${item.original_name || item.file}`, item.file)));
      section.classList.remove('hidden');
      await edit(source.images[0].file);
      document.querySelectorAll('#sourceGallery .source-thumb').forEach((card, i) => { card.title = 'Click to edit the selection for this image'; card.onclick = safe(event => event.target.closest('input, label') ? undefined : edit(source.images[i].file)); });
    },
  };
}



