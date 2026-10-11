// DroneVision: work session — what is opened, marked, drawn and shown.
// Loaded after tools.js and before app.js; uses app.js globals (form, video, viewer, selectedFrames…) only
// inside functions, i.e. after app.js has run.
//  * The last session is saved automatically on the server (~/.dronevision/last_session.json, see
//    dronevision/session.py) and restored at the next start, in any browser.
//  * «Сохранить в файл…» / «Открыть из файла…» write and read the same JSON anywhere on disk.
// Results (projection, comparison, mosaic, average, comparison of two averages) are restored from the PNG files in the project folder;
// the session keeps their description, not the pixels.
DV.session = (() => {
  const FIELDS = ['project_dir', 'yaw_offset_deg', 'takeoff_height_m', 'mosaic_max_distance_m', 'projection_resolution_m'];
  const results = {projection: null, mosaic: null, waves: null};
  let dirty = false, resultsDirty = false, restoring = false, busy = false, lastLight = '';
  const ready = () => {try {return !!viewer} catch (e) {return false}};   // app.js not loaded yet -> not ready
  const statusNode = () => document.querySelector('#sessionStatus');
  function say(text, bad = false) {const node = statusNode(); if (node) {node.textContent = text; node.classList.toggle('bad', bad)}}
  function when(iso) {const d = iso ? new Date(iso) : new Date(); return isNaN(d) ? '' : d.toLocaleString('ru-RU', {day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit'})}

  function snapshot(withResults = true) {
    const fields = {video_path: viewer.video_path, log_path: form.elements.log_path.value};
    FIELDS.forEach(key => {if (form.elements[key]) fields[key] = form.elements[key].value});
    if (form.elements.refine_alignment) fields.refine_alignment = form.elements.refine_alignment.checked;
    const annotations = Object.fromEntries(Object.entries(DV.annotationStore.images).filter(([, entry]) => entry.shapes && entry.shapes.length));
    const session = {
      form: fields,
      view: {time_s: video.currentTime || 0, horizon_offset_percent: Number(horizonOffset.value) || 0, bottom_crop_percent: Number(bottomCropPercent.value) || 0},
      selected_frames: selectedFrames, selected_intervals: selectedIntervals, interval_start: intervalStart,
      adjustments: DV.adjust.params(), overlays: {...DV.overlays}, annotations,
    };
    if (withResults) session.results = Object.fromEntries(Object.entries(results).filter(([, value]) => value));
    return session;
  }

  // Autosave: cheap state goes on every change, the bulky results only when a new one was built.
  function touch() {if (!restoring) dirty = true}
  function setResult(slot, value) {results[slot] = value; resultsDirty = true; touch()}
  async function flush() {
    if (!dirty || restoring || busy || !ready()) return;
    const sendResults = resultsDirty, light = JSON.stringify(snapshot(false));
    dirty = false;
    if (!sendResults && light === lastLight) return;
    busy = true; resultsDirty = false;
    try {
      const body = sendResults ? {session: snapshot(true)} : {session: JSON.parse(light), keep_results: true};
      const saved = await api('/api/session/save', {method: 'POST', body: JSON.stringify(body)});
      lastLight = light; say(`сохранена автоматически · ${when(saved.saved_at)}`);
    } catch (error) {dirty = true; resultsDirty = resultsDirty || sendResults; say(`не удалось сохранить: ${error.message}`, true)}
    finally {busy = false}
  }
  ['click', 'change', 'input', 'pointerup', 'keyup'].forEach(name => document.addEventListener(name, touch, true));
  window.addEventListener('dv:annotations-changed', touch);
  setInterval(flush, 2000);
  // Closing the tab right after a change: send what is pending without waiting for the timer.
  window.addEventListener('pagehide', () => {
    if (!dirty || restoring || !ready()) return;
    const body = resultsDirty ? {session: snapshot(true)} : {session: snapshot(false), keep_results: true};
    navigator.sendBeacon('/api/session/save', new Blob([JSON.stringify(body)], {type: 'application/json'}));
  });

  // Another video was opened: the results on screen belong to the previous one (the files stay in the project folder).
  function clearResults() {
    document.querySelector('#projectionResult').hidden = true; document.querySelector('#mosaicResult').hidden = true; DV.waves.hide();
    results.projection = null; results.mosaic = null; results.waves = null; resultsDirty = true; touch();
  }

  const rasterUrl = paths => (index, saved) => `/api/raster?path=${encodeURIComponent(paths[index])}${saved ? '&saved=1' : ''}&t=${Date.now()}`;
  function showResults(saved) {
    const notes = [];
    document.querySelector('#projectionResult').hidden = true; document.querySelector('#mosaicResult').hidden = true;
    results.projection = null; results.mosaic = null; results.waves = null; DV.waves.hide();
    const projection = saved.projection, mosaic = saved.mosaic;
    try {
      if (projection && projection.result) {
        const urlFor = rasterUrl(projection.result.variants.map(v => v.path));
        if (projection.kind === 'pair') showPairResult(projection.result, urlFor, true); else showProjectionResult(projection.result, urlFor, true);
        results.projection = projection;
      }
    } catch (error) {notes.push(`проекция не восстановлена: ${error.message}`)}
    try {
      if (mosaic && mosaic.result) {
        const urlFor = rasterUrl(mosaic.kind === 'average-pair' ? mosaic.result.variants.map(v => v.path) : [mosaic.result.path]);
        if (mosaic.kind === 'average-pair') showAveragePairResult(mosaic.result, urlFor, true);
        else if (mosaic.kind === 'average') showAverageResult(mosaic.result, urlFor, true); else showMosaicResult(mosaic.result, mosaic.info, urlFor, true);
        results.mosaic = mosaic;
      }
    } catch (error) {notes.push(`мозаика не восстановлена: ${error.message}`)}
    if (saved.waves && saved.waves.path) {
      results.waves = saved.waves;
      DV.waves.restore(saved.waves).catch(error => {results.waves = null; say(`течения не восстановлены: ${error.message}`, true)});
    }
    return notes;
  }

  // Put a stored session on screen; `loaded` is the answer of /api/session/load.
  async function apply(loaded) {
    const session = loaded.session, fields = session.form || {}, view = session.view || {};
    if (!fields.video_path) throw Error('в сессии не указано видео');
    if (!loaded.video_exists) throw Error(`видео недоступно: ${fields.video_path}`);
    restoring = true;
    try {
      FIELDS.forEach(key => {if (fields[key] != null && form.elements[key]) form.elements[key].value = fields[key]});
      // Sessions saved before the distance limit became optional carry its old default; it was not a choice.
      if (fields.mosaic_max_distance_m === '3000') form.elements.mosaic_max_distance_m.value = '';
      if (form.elements.refine_alignment && fields.refine_alignment != null) form.elements.refine_alignment.checked = !!fields.refine_alignment;
      if (session.adjustments) DV.adjust.setAll(session.adjustments);
      Object.entries(session.overlays || {}).forEach(([key, value]) => DV.setOverlay(key, !!value));
      // A session is an exact snapshot.  Clear annotations from the currently open session first;
      // otherwise shapes absent from the loaded file can leak into it when raster paths coincide.
      Object.keys(DV.annotationStore.images).forEach(path => delete DV.annotationStore.images[path]);
      Object.assign(DV.annotationStore.images, session.annotations || {}); DV.annotationStore.save();
      await openViewer(fields.video_path);   // clears the marks, so they are restored after it
      selectedFrames = (session.selected_frames || []).map(item => ({...item}));
      selectedIntervals = (session.selected_intervals || []).map(range => [...range]);
      intervalStart = session.interval_start ?? null;
      setHorizonOffset(Number(view.horizon_offset_percent) || 0); setBottomCrop(Number(view.bottom_crop_percent) || 0);
      renderSelections();
      const time = Number(view.time_s) || 0;
      if (time > 0) afterMetadata(() => {video.currentTime = Math.min(time, video.duration || time)});
      const notes = showResults(session.results || {});
      if (loaded.missing_results && loaded.missing_results.length) notes.push(`файлы результатов не найдены (${loaded.missing_results.length}) — постройте заново`);
      lastLight = ''; dirty = false; resultsDirty = false;
      return notes;
    } finally {restoring = false}
  }

  // Start of the page: the last session, otherwise the default video as before.
  async function start() {
    let loaded = null;
    try {loaded = await api('/api/session/load', {method: 'POST', body: '{}'})} catch (error) {say(`последняя сессия не прочитана: ${error.message}`, true)}
    if (loaded && loaded.session) {
      try {
        const notes = await apply(loaded);
        say(`восстановлена последняя сессия от ${when(loaded.saved_at)}${notes.length ? ' · ' + notes.join(' · ') : ''}`, notes.length > 0);
        return;
      } catch (error) {say(`последняя сессия не восстановлена: ${error.message}`, true)}
    } else if (loaded) say('новая сессия · сохраняется автоматически');
    await openViewer();
  }

  function sessionFolder() {return form.elements.project_dir.value || folderOf(form.elements.video_path.value)}
  async function saveToFile() {
    if (!ready()) {say('Сначала откройте видео.', true); return}
    const stamp = new Date(Date.now() - new Date().getTimezoneOffset() * 60000).toISOString().slice(0, 16).replace(/[-:]/g, '').replace('T', '_');
    const name = `${baseName(viewer.video_path).replace(/\.[^.]+$/, '')}_${stamp}.dvsession.json`;
    const chosen = await api('/api/dialog', {method: 'POST', body: JSON.stringify({kind: 'save', filter: 'session', start: sessionFolder(), name})});
    if (!chosen.path) return;
    const saved = await api('/api/session/save', {method: 'POST', body: JSON.stringify({session: snapshot(true), path: chosen.path})});
    say(`сохранена в файл ${saved.path}`);
  }
  async function openFromFile() {
    let start = '';
    try {start = sessionFolder()} catch (e) {}
    const chosen = await api('/api/dialog', {method: 'POST', body: JSON.stringify({kind: 'file', filter: 'session', start})});
    if (!chosen.path) return;
    const loaded = await api('/api/session/load', {method: 'POST', body: JSON.stringify({path: chosen.path})});
    const notes = await apply(loaded);
    dirty = true; resultsDirty = true;   // the opened session becomes the last one
    say(`открыта сессия ${baseName(loaded.path)} от ${when(loaded.saved_at)}${notes.length ? ' · ' + notes.join(' · ') : ''}`, notes.length > 0);
  }
  async function startNew() {
    if (!confirm('Начать новую сессию? Отмеченные кадры и показанные результаты будут убраны с экрана; файлы в папке проекта останутся.')) return;
    restoring = true;   // nothing may be autosaved between clearing and reloading
    await api('/api/session/clear', {method: 'POST', body: '{}'});
    location.reload();
  }
  function bind() {
    const guard = handler => () => handler().catch(error => {restoring = false; say(error.message, true)});
    [['#sessionSave', saveToFile], ['#sessionOpen', openFromFile], ['#sessionNew', startNew]].forEach(([selector, handler]) => {
      const button = document.querySelector(selector); if (button) button.onclick = guard(handler);
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bind); else bind();

  return {start, touch, setResult, clearResults, snapshot, apply, saveToFile, openFromFile};
})();
