import * as THREE from 'three';

const $ = id => document.getElementById(id);
const point = value => value == null ? 'Not available' : `${Math.round(value * 100)}/100`;
const relationNames = { on: 'on', attached: 'attached', around: 'bao quanh', inside: 'inside', beside: 'beside', hanging: 'hanging' };

export function createCompositionController(context) {
  const { state, post, api, download, captureModels, captureRenderer, environment, addLights, objectTransform, applyTransform, loading, toast, setStatus, updateAvailability, cancelPick, setView, select, frameScene, getHints } = context;
  let run = null, images = {}, objectUrls = [], editSerial = 0;
  const questionBox=document.createElement('section');questionBox.id='agentQuestion';
  questionBox.className='agent-question hidden';
  $('composeBtn').after(questionBox);
  function askUser(result,request,snapshot,serial) {
    questionBox.replaceChildren();questionBox.classList.remove('hidden');
    const heading=document.createElement('strong');heading.textContent=result.question.question;
    questionBox.append(heading);
    const choices=document.createElement('div');choices.className='inline-actions';questionBox.append(choices);
    const answer=document.createElement('input');answer.maxLength=500;answer.placeholder='Or type your own answer';
    const resume=async value=>{
      if(state.busy)return;
      try {
        assertCurrent(snapshot,serial);
        state.busy=true;updateAvailability();questionBox.classList.add('hidden');
        await runAgent({...request,clarification_of:result.id,clarification_answer:value},snapshot,serial);
      } catch(error){toast(error.message,true);}
      finally{state.busy=false;loading(false);updateAvailability();}
    };
    for(const label of result.question.options) {
      const button=document.createElement('button');button.type='button';button.className='button outline';
      button.textContent=label;button.onclick=()=>resume(label);choices.append(button);
    }
    const submit=document.createElement('button');submit.className='button dark';submit.textContent='Continue';
    submit.onclick=()=>{if(answer.value.trim())resume(answer.value.trim());};
    questionBox.append(answer,submit);
    setStatus('Choose an answer to continue. The current scene is retained.');
  }


  function reset() {
    run = null; images = {};$('compositionDetails').classList.add('hidden');$('compositionStatus').classList.add('hidden');questionBox.classList.add("hidden");questionBox.replaceChildren();
    objectUrls.forEach(url => URL.revokeObjectURL(url)); objectUrls = [];
    $('compositionActions').classList.add('hidden'); $('planInfo').classList.add('hidden'); $('compositionDirty').classList.add('hidden');
    $('candidateGrid').replaceChildren();
    if ($('candidatesDialog').open) $('candidatesDialog').close();
  }

  function markEdited() {
    editSerial++;
    if (run) $('compositionDirty').classList.remove('hidden');
  }

  function identity() { return JSON.stringify([state.A?.id, state.B?.id, objectTransform('A'), $('instruction').value, getHints()]); }

  function assertCurrent(snapshot, serial) {
    if (identity() !== snapshot || editSerial !== serial) {
      reset();
      throw new Error('Objects, instructions, or the base pose changed during processing. Run again for the current scene.');
    }
  }

  function applyToRoot(root, transform) {
    root.position.fromArray(transform.position); root.rotation.set(...transform.rotation, 'XYZ'); root.scale.fromArray(transform.scale); root.updateMatrixWorld(true);
  }

  async function renderCandidates(prepared) {
    const scene = new THREE.Scene(); scene.background = new THREE.Color(0xf4f5f0); scene.environment = environment; addLights(scene);
    const a = new THREE.Group(), b = new THREE.Group();
    a.add(state.objects.A.children[0].clone(true)); b.add(state.objects.B.children[0].clone(true));
    applyToRoot(a, prepared.base_transform); scene.add(a, b);
    const bounds = new THREE.Box3().setFromObject(a);
    for (const candidate of prepared.candidates) { applyToRoot(b, candidate.transform); bounds.union(new THREE.Box3().setFromObject(b)); }
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z, 0.1);
    const camera = new THREE.PerspectiveCamera(35, 1, 0.001, Math.max(100, radius * 30));
    const distance = radius / (2 * Math.tan(THREE.MathUtils.degToRad(35 / 2))) * 1.6;
    const views = [['FRONT', [0, 0.06, 1]], ['SIDE', [1, 0.06, 0]], ['BACK', [0, 0.06, -1]], ['OBLIQUE', [1, 0.8, 1]]];
    const renders = []; captureRenderer.setSize(448, 448);
    for (const candidate of prepared.candidates) {
      applyToRoot(b, candidate.transform);
      const features=candidate.resolved_features||prepared.diagnostics?.resolved_features||{};
      const constraints=candidate.executed_program?.constraints||prepared.plan.program?.constraints||[];
      const pair=constraints.find(item=>['clearance','contact','containment'].includes(item.kind));
      const fa=pair&&features[pair.a], fb=pair&&features[pair.b];
      let focus=null,focusRadius=radius,contactDirections=[];
      if(fa?.center && fb?.center) {
        const ca=new THREE.Vector3(...fa.center);
        const cb=new THREE.Vector3(...fb.center).applyMatrix4(b.matrixWorld);
        focus=ca.clone().add(cb).multiplyScalar(.5);
        focusRadius=Math.max(.025,Math.min(radius*.65,Math.max(fa.extent||.05,
          (fb.extent||.05)*Math.max(...candidate.transform.scale))*1.55));
        const axis=new THREE.Vector3(...(fa.axis||[0,1,0])).normalize();
        let normal=new THREE.Vector3(0,0,1).addScaledVector(axis,-axis.z);
        if(normal.length()<.1) normal=new THREE.Vector3(1,0,0).addScaledVector(axis,-axis.x);
        normal.normalize();
        contactDirections=[normal.clone().addScaledVector(axis,.35).toArray(),
          normal.clone().negate().addScaledVector(axis,-.25).toArray()];
      }
      const renderViews=focus?[...views,['CONTACT CLOSE-UP',contactDirections[0],true],
        ['CONTACT REVERSE',contactDirections[1],true]]:views;
      const sheet = document.createElement('canvas'); sheet.width = 960; sheet.height = focus?1492:1012;
      const canvas = sheet.getContext('2d'); canvas.fillStyle = '#f4f5f0'; canvas.fillRect(0, 0, sheet.width, sheet.height);
      canvas.fillStyle = '#242a2b'; canvas.font = 'bold 22px sans-serif'; canvas.fillText(`Candidate ${candidate.id} | A + B`, 20, 32);
      for (let i = 0; i < renderViews.length; i++) {
        const [label, direction,close] = renderViews[i];
        const target=close?focus:center;
        const cameraDistance=close?focusRadius/(2*Math.tan(THREE.MathUtils.degToRad(35/2)))*1.6:distance;
        camera.position.copy(target).addScaledVector(new THREE.Vector3(...direction).normalize(), cameraDistance); camera.lookAt(target); camera.updateProjectionMatrix(); camera.updateMatrixWorld(true);
        captureRenderer.render(scene, camera);
        const x = (i % 2) * 480 + 16, y = Math.floor(i / 2) * 480 + 74;
        canvas.font = '15px sans-serif'; canvas.fillStyle = '#64705f'; canvas.fillText(label, x, y - 9);
        canvas.drawImage(captureRenderer.domElement, x, y, 448, 448);
      }
      const image = sheet.toDataURL('image/png'); images[candidate.id] = image;
      renders.push({ candidate_id: candidate.id, image, layout: focus?'front-side-back-oblique-contact-v2':'front-side-back-oblique-v1' });
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
    return renders;
  }

  function applyCandidate(candidate, manual = false) {
    if (!run || state.A?.id !== run.base_id || state.B?.id !== run.product_id || JSON.stringify(objectTransform('A')) !== JSON.stringify(run.base_transform)) {
      throw new Error('This candidate belongs to older objects or a different base pose. Run the agent again.');
    }
    cancelPick(); applyTransform('B', candidate.transform); setView('all'); select('B'); frameScene();
    $('compositionDirty').classList.add('hidden');
    setStatus(`${manual ? 'User selected' : 'AI selected'} ${candidate.id.toUpperCase()} · ${candidate.label}`);
    if ($('candidatesDialog').open) $('candidatesDialog').close();
  }

  async function showResults(result) {
    run = result;
    $('compositionDetails').classList.remove('hidden');
    $('compositionStatus').classList.remove('hidden');
    $('compositionStatus').textContent=result.selected_candidate_id?'A composition is ready. Rotate the scene to inspect it.':
      result.candidates.some(item=>item.eligible)?'A composition proposal is ready for inspection.':'The agent has not met the goal. Open details or revise the target.';
    for (const candidate of result.candidates) {
      if (!images[candidate.id] && result.renders?.some(r => r.candidate_id === candidate.id)) {
        try { const response = await api(`compositions/${result.id}/renders/${candidate.id}.png`); const url = URL.createObjectURL(await response.blob()); objectUrls.push(url); images[candidate.id] = url; } catch { /* Scores still usable if thumbnail download fails. */ }
      }
    }
    const chosen = result.candidates.find(c => c.id === (result.selected_candidate_id || result.suggested_candidate_id));
    const status = result.evaluation_status === 'completed' ? 'Reviewed the scene and available interface close-ups.' : result.evaluation_status === 'skipped_geometry' ? 'Stopped at geometry checks; visual evaluation was not requested.' : 'Visual evaluation was not requested.';
    $('planInfo').textContent = `AI proposal, not a confirmation:\n${result.plan.explanation}\nRelationship: ${relationNames[result.plan.relation] || result.plan.relation}. ${status}\n${result.needs_review ? 'Review and select a candidate; it has not been applied automatically.' : `Proposal ${chosen.id.toUpperCase()} · geometry ${point(chosen.geometry_score)} · visual ${point(chosen.visual_score)}.`}${result.reused ? '\nReused the saved result; completed evaluations are not called again.' : ''}${result.plan.deform ? '\nSoft objects may need additional mesh editing.' : ''}`;
    $('planInfo').classList.remove('hidden'); $('compositionActions').classList.remove('hidden');
    $('evaluationSummary').textContent = result.summary;
    if (result.api_calls) $('planInfo').textContent += `\nLLM calls: planning ${result.api_calls.plan || 0}, evaluation ${result.api_calls.evaluation || 0}.`;
    if (result.plan.program) $('planInfo').textContent += `\nSpatial program: ${result.plan.program.features.length} features · ${result.plan.program.constraints.length} constraints · AI-selected tools.`;
    if (result.needs_review && result.warnings?.length) $('planInfo').textContent += '\nReview required: ' + result.warnings.join(' ');
    const warnings = result.warnings || [];
    $('evaluationWarnings').textContent = warnings.join('\n'); $('evaluationWarnings').classList.toggle('hidden', !warnings.length);
    $('candidateGrid').replaceChildren();
    for (const candidate of result.candidates) {
      const card = document.createElement('article'); card.className = 'candidate-card'; card.dataset.candidate = candidate.id;
      card.classList.toggle('recommended', candidate.id === result.selected_candidate_id); card.classList.toggle('rejected', !candidate.eligible);
      const title = document.createElement('strong'); title.textContent = `${candidate.id.toUpperCase()} · ${candidate.label}`; card.append(title);
      const badge = document.createElement('span'); badge.className = 'candidate-badge'; badge.textContent = !candidate.eligible ? 'Geometry failed' : candidate.visual_review && !candidate.visual_review.acceptable ? 'VLM rejected' : candidate.id === result.selected_candidate_id ? 'Proposal' : candidate.baseline ? 'Reference · single pass' : 'Alternative'; card.append(badge);
      if (images[candidate.id]) { const image = document.createElement('img'); image.src = images[candidate.id]; image.alt = `Views of ${candidate.id}`; image.loading = 'lazy'; card.append(image); }
      const scores = document.createElement('div'); scores.className = 'candidate-scores';
      for (const [label, score] of [['Geometry', candidate.geometry_score], ['Visual', candidate.visual_score], ['Combined', candidate.combined_score]]) { const span = document.createElement('span'); span.textContent = `${label}: ${point(score)}`; scores.append(span); } card.append(scores);
      const explanation = document.createElement('p'); explanation.textContent = candidate.visual_review?.explanation || 'No VLM review yet.'; card.append(explanation);
      const issues = [...(candidate.failures || []), ...(candidate.warnings || []), ...(candidate.visual_review?.issues || [])];
      if (issues.length) { const notes = document.createElement('p'); notes.className = 'candidate-issues'; notes.textContent = issues.join(' '); card.append(notes); }
      const button = document.createElement('button'); button.className = 'button outline full'; button.textContent = candidate.eligible ? 'View this candidate in 3D' : 'Open for manual editing';
      button.onclick = () => { try { applyCandidate(candidate, true); toast(candidate.eligible ? 'Opened the candidate for inspection and editing.' : 'This candidate failed checks. Adjust the anchors or geometry.', !candidate.eligible); } catch (error) { toast(error.message, true); } }; card.append(button);
      $('candidateGrid').append(card);
    }
    const fit=chosen?.metrics?.interface_fit?.[0];
    if(fit) $('planInfo').textContent+='\n'+'Median interface gap, relative to the target radius: ' +(fit.gap_median_ratio*100).toFixed(1)+'%'+' · Axis-scale difference: ' +(((chosen.metrics.axis_scale_ratio||1)-1)*100).toFixed(1)+'%';
    const delta = result.comparison?.combined_delta;
    $('baselineComparison').hidden = !result.comparison;
    $('baselineComparison').textContent = result.evaluation_status === 'completed' ? `C0 is the single-pass reference. Combined-score difference: ${delta >= 0 ? '+' : ''}${Math.round((delta || 0) * 100)} points. This is a heuristic comparison for this run, not dataset-wide evidence.` : 'C0 is the single-pass reference. No visual comparison has completed.';
  }


  async function waitTask(task, snapshot, serial, prefix='') {
    while (!['success','error','paused'].includes(task.status)) {
      assertCurrent(snapshot,serial);
      loading(true,prefix+task.message);
      await new Promise(resolve=>setTimeout(resolve,900));
      task=await (await api('composition-jobs/'+task.id)).json();
    }
    if (task.status!=='success') throw new Error(task.message);
    return task.result;
  }

  async function runAgent(request,snapshot,serial) {
    const limit=Number($('agentRounds').value);
    let previous=request.refinement_of||null, result=null;
    for(let round=1;round<=limit;round++) {
      images={};
      const prefix='Agent · round '+round+'/'+limit+' · ';
      loading(true,prefix+(round===1?'reading the instruction and measuring target features…':'revising from the previous checks…'));
      result=await waitTask(await post('composition-jobs',{
        ...request,candidate_count:Number($("candidateCount").value),refinement_of:previous
      }),snapshot,serial,prefix);
      assertCurrent(snapshot,serial);
      if(result.status==='needs_input') {
        askUser(result,{...request,refinement_of:previous},snapshot,serial);
        return;
      }
      if(result.agent_no_change) break;
      if(result.candidates.length && result.evaluation_status!=='completed') {
        loading(true,prefix+'rendering candidates to check placement, orientation, and scale…');
        const renders=await renderCandidates(result);
        result=await waitTask(await post('compositions/'+result.id+'/evaluation-jobs',{renders}),
                              snapshot,serial,prefix);
      }
      assertCurrent(snapshot,serial);
      setStatus(prefix+result.summary);
      const accepted=result.candidates.find(candidate=>candidate.id===result.selected_candidate_id);
      if(accepted?.eligible && accepted.visual_review?.acceptable &&
          accepted.geometry_score>=.6 && accepted.visual_score>=.55 && result.plan.confidence>=.6) break;
      if(result.status==='evaluation_unavailable' || result.diagnostics?.grounding_repair_error) break;
      if(result.plan.deform || result.plan.program?.deformation_required) break;
      if(result.agent_round>=limit) break;
      previous=result.id;
    }
    result=await (await api('compositions/'+result.id+'/agent-result')).json();
    await showResults(result);
    const history=result.agent_attempts||[...(result.agent_history||[]),{round:result.agent_round,api_calls:result.api_calls,summary:result.summary}];
    if(result.agent_selected_round!==result.agent_total_rounds)
      $('planInfo').textContent+='\nRetained the accepted proposal from round '+result.agent_selected_round+'.';
    const calls=history.reduce((sum,item)=>sum+(item.api_calls?.plan||0)+(item.api_calls?.evaluation||0),0);
    $('planInfo').textContent+='\nAgent: '+history.length+' rounds · '+calls+' LLM calls.\n'+
      history.map(item=>'Round '+item.round+': '+item.summary).join('\n');
    assertCurrent(snapshot,serial);
    if(result.auto_apply && result.selected_candidate_id) {
      applyCandidate(result.candidates.find(candidate=>candidate.id===result.selected_candidate_id));
      toast('The agent opened a proposal after geometry and visual checks. Inspect other views.');
    } else if(result.selected_candidate_id) {
      setStatus('A checked candidate was retained; review its remaining limitations.');
      toast('The AI accepted a candidate. Inspect it before applying.');
    } else {
      setStatus('Agent stopped without an accepted candidate.');
      toast('The agent did not meet the goal within its budget. Inspect the candidates; the current scene is retained.',true);
    }
  }


  async function runComposition() {
    if (!state.A || !state.B || state.busy) return;
    state.busy = true; updateAvailability();
    const snapshot = identity(), serial = editSerial;
    reset();
    try {
      loading(true, 'Step 1/3 · Analyzing the relationship and generating spatial candidates…');
      const request = { base_id: state.A.id, product_id: state.B.id, base_transform: objectTransform('A'), instruction: $('instruction').value, hints: getHints(), captures: await captureModels(), candidate_count: ['geometry','agent'].includes($('composeMode').value) ? 1 : Number($('candidateCount').value), use_cache: true,search_budget:state.searchBudget };
      if ($('composeMode').value === 'agent') { await runAgent(request,snapshot,serial); return; }
      if ($('composeMode').value === 'quick') {
        const result = await post('compose', request); assertCurrent(snapshot, serial);
        if(result.status==='needs_input'){askUser(result,request,snapshot,serial);return;}
        if (result.eligible === false) throw new Error(result.failures.join(' '));
        cancelPick(); applyTransform('B', result.transform); setView('all'); select('B'); frameScene();
        $('planInfo').textContent = `${result.plan.explanation}\nSingle-pass composition: no candidate comparison or visual review.`; $('planInfo').classList.remove('hidden');
        setStatus('Single-pass composition; compare it with agent review'); return;
      }
      let task = await post('composition-jobs', request);
      while (!['success', 'error', 'paused'].includes(task.status)) {
        assertCurrent(snapshot, serial);
        loading(true, task.message + ' Large meshes may take longer.');
        await new Promise(resolve => setTimeout(resolve, 900));
        task = await (await api('composition-jobs/' + task.id)).json();
      }
      if (task.status !== 'success') throw new Error(task.message);
      let prepared = task.result; assertCurrent(snapshot, serial);
      if(prepared.status==='needs_input'){askUser(prepared,request,snapshot,serial);return;}
      if ($('composeMode').value === 'geometry' || prepared.evaluation_status === 'skipped_geometry' || prepared.evaluation_status === 'completed') {
        await showResults(prepared);
      } else {
        loading(true, `Step 2/3 · Rendering ${prepared.candidates.length} candidates from multiple views…`);
        const renders = await renderCandidates(prepared); assertCurrent(snapshot, serial);
        loading(true, 'Step 3/3 · Reviewing rendered candidates against geometry and the goal…');
        try { prepared = await post(`compositions/${prepared.id}/evaluate`, { renders }); }
        catch (error) { prepared = { ...prepared, needs_review: true, auto_apply: false, summary: 'Visual evaluation is unavailable. Inspect and choose a candidate manually.', warnings: [...(prepared.warnings || []), error.message] }; }
        assertCurrent(snapshot, serial); await showResults(prepared);
      }
      assertCurrent(snapshot, serial);
      if ($('composeMode').value === 'geometry' && run.candidates.some(c => c.eligible)) {
        const candidate = run.candidates.find(c => c.id === run.suggested_candidate_id);
        applyCandidate(candidate); setStatus('Geometric proposal; inspect it from multiple views');
        $('planInfo').textContent = $('planInfo').textContent.replace('Review and select a candidate; it has not been applied automatically.', 'Opened the geometric proposal in 3D; inspect all views.');
        toast('Opened a geometric proposal. Visual LLM review has not run.');
      } else if (run.auto_apply && run.selected_candidate_id) {
        applyCandidate(run.candidates.find(c => c.id === run.selected_candidate_id)); toast('Selected a candidate after geometry and multi-view review.');
      } else {
        setStatus('Review candidates or select a different target'); toast('Review required. Click View 3D results; the current scene is retained.');
      }
    } finally { state.busy = false; updateAvailability(); loading(false); }
  }

  function open() { if(!run)return; if(run.candidates.length===1 && run.candidates[0].eligible){applyCandidate(run.candidates[0],true);return;} if(!$('candidatesDialog').open)$('candidatesDialog').showModal(); }
  async function report() { if (run) await download(`compositions/${run.agent_run_id||run.id}/${run.agent_run_id?'agent-result':'report'}`, 'unicompose3d-composition-report.json'); }
  $('composeMode').onchange = () => {
    const agent = $('composeMode').value === 'agent';
    $('agentRoundsLabel').hidden = !agent;
    if(agent) {
      $('candidateCount').disabled=false;
      $('composeBtn').innerHTML='<span>✦</span> Compose with agent';
      $('compositionCostNote').textContent='Up to '+$('agentRounds').value+' rounds; each proposes, measures, and reviews candidates. API usage follows actual calls; meshes are not reconstructed.';
      return;
    }
    const quick = $('composeMode').value === 'quick'; $('candidateCount').disabled = $('composeMode').value !== 'evaluated';
    $('composeBtn').innerHTML = '<span>✦</span> ' + ($('composeMode').value === 'geometry' ? 'Propose a 3D composition' : quick ? 'Single-pass composition' : 'Compose and review');
    $('compositionCostNote').textContent = quick ? 'Up to three plan/repair calls without cache · comparison mode' : $('composeMode').value === 'geometry' ? 'Up to three plan/repair calls · no visual review or reconstruction' : 'Up to four LLM calls · rendered candidates are reviewed against geometry';
  };
  $('compositionDiagnostics').onclick=()=>run && download('compositions/'+(run.agent_run_id||run.id)+'/diagnostics.zip','unicompose3d-composition-diagnostics.zip').catch(error=>toast(error.message,true));
  $('agentRounds').onchange=()=>$('composeMode').onchange();
  return { run: runComposition, reset, markEdited, open, report };
}



