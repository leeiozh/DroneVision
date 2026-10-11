// DroneVision: coastline over the video frame (dronevision/coastview.py).
// Loaded after tools.js and before app.js; app.js globals (viewer, video, form, api, seriesAt…) are used
// only inside functions.
//  * After a video is opened (and after every seek) the server says whether land should be in the frame.
//    If so, the collapsible panel «Берег и выравнивание по нему» says it in its heading and offers to draw
//    the coastline over the picture.
//  * The server sends the outlines as viewing directions (azimuth, depression); they are projected here
//    with the pose of the current frame, so the overlay follows the playback and can be dragged:
//    left/right changes «Поправка курса», up/down changes the gimbal pitch offset of this video.
//  * «Выровнять автоматически»: pitch and roll from the sea horizon, heading from the land in view.
//  * «Сохранить выравнивание» writes pitch/roll to <video>.pose.json — every tool then uses them.
DV.coast = (() => {
  const st = {info: null, shown: false, pitch: 0, roll: 0, saved: {pitch: 0, roll: 0}, video: '', timer: null,
    drag: null, lastDraw: '', offered: false, requested: 0, sticky: 0};
  const q = selector => document.querySelector(selector);
  const ready = () => {try {return !!viewer} catch (e) {return false}};
  const rad = Math.PI / 180;
  const signed = (value, digits = 2) => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(digits)}`;
  const yawOffset = () => Number(form.elements.yaw_offset_deg.value) || 0;
  // A result message (alignment found, saved) stays for a while: routine re-checks do not overwrite it.
  function say(text, state = 'idle', keep = false) {if (!keep && Date.now() < st.sticky) return; st.sticky = keep ? Date.now() + 12000 : 0; activity(q('#coastStatus'), text, state)}
  function params() {
    return {video_path: viewer.video_path, time_s: video.currentTime || 0, yaw_offset_deg: yawOffset(),
      takeoff_height_m: Number(form.elements.takeoff_height_m.value) || 0, horizontal_fov_deg: Number(form.elements.horizontal_fov_deg.value) || 0};
  }

  // Camera of the frame on screen: log series + heading offset + pitch/roll offsets being tried.
  function camera() {
    const c = st.info.camera, t = video.currentTime || 0;
    const focal = c.focal_px * seriesAt('focal_length_mm', t) / st.focalMm;
    return {yaw: seriesAt('drone_yaw', t, true) + yawOffset(), pitch: seriesAt('gimbal_pitch', t) + st.pitch,
      roll: seriesAt('gimbal_roll', t) + st.roll, focal, width: c.width, height: c.height};
  }
  function basis(cam) {
    const y = cam.yaw * rad, p = cam.pitch * rad, r = cam.roll * rad;
    const f = [Math.cos(p) * Math.sin(y), Math.cos(p) * Math.cos(y), Math.sin(p)], right = [Math.cos(y), -Math.sin(y), 0];
    const down = [f[1] * right[2] - f[2] * right[1], f[2] * right[0] - f[0] * right[2], f[0] * right[1] - f[1] * right[0]];
    const mix = (a, b, ka, kb) => [a[0] * ka + b[0] * kb, a[1] * ka + b[1] * kb, a[2] * ka + b[2] * kb];
    return [f, mix(right, down, Math.cos(r), Math.sin(r)), mix(right, down, -Math.sin(r), Math.cos(r))];
  }

  function draw() {
    const canvas = q('#coastCanvas'), wrap = q('#videoWrap');
    if (!canvas) return;
    const visible = st.shown && st.info && st.info.lines.length && ready();
    canvas.hidden = !visible; q('#coastDrag').hidden = !visible;
    if (!visible) return;
    const dpr = devicePixelRatio || 1, w = wrap.clientWidth, h = wrap.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr)}
    const g = canvas.getContext('2d'), cam = camera(), [f, r, d] = basis(cam), sx = w / cam.width, sy = h / cam.height;
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h); g.lineJoin = 'round'; g.lineCap = 'round';
    const horizonKm = st.info.horizon_distance_km;
    for (const pass of [['rgba(0,0,0,.75)', 4], ['#ffd84a', 1.6]]) {
      g.strokeStyle = pass[0]; g.lineWidth = pass[1];
      for (const line of st.info.lines) {
        g.beginPath();
        let pen = false;
        for (const [az, dep, dist] of line) {
          const a = az * rad, e = dep * rad, v = [Math.sin(a) * Math.cos(e), Math.cos(a) * Math.cos(e), -Math.sin(e)];
          const depth = v[0] * f[0] + v[1] * f[1] + v[2] * f[2];
          // Beyond the sea horizon the waterline itself is hidden; only what rises above it is seen.
          if (depth <= 0.02 || dist > horizonKm * 1000) {pen = false; continue}
          const x = (cam.width / 2 + cam.focal * (v[0] * r[0] + v[1] * r[1] + v[2] * r[2]) / depth) * sx;
          const y = (cam.height / 2 + cam.focal * (v[0] * d[0] + v[1] * d[1] + v[2] * d[2]) / depth) * sy;
          if (x < -w || x > 2 * w || y < -h || y > 2 * h) {pen = false; continue}
          pen ? g.lineTo(x, y) : g.moveTo(x, y); pen = true;
        }
        g.stroke();
      }
    }
  }
  function loop() {
    if (st.shown) {
      const key = ready() ? `${video.currentTime}|${yawOffset()}|${st.pitch}|${st.roll}|${q('#videoWrap').clientWidth}` : '';
      if (key !== st.lastDraw) {st.lastDraw = key; draw()}
    }
    requestAnimationFrame(loop);
  }

  function describe() {
    const info = st.info, row = q('#coastRow');
    if (!info) return;
    const changed = Math.abs(st.pitch - st.saved.pitch) > 1e-4 || Math.abs(st.roll - st.saved.roll) > 1e-4;
    q('#coastOffsets').textContent = `курс ${signed(yawOffset(), 1)}° · наклон ${signed(st.pitch)}° · крен ${signed(st.roll)}°${changed ? ' · не сохранено' : ''}`;
    q('#coastSave').disabled = !changed;
    q('#coastToggle').textContent = st.shown ? 'Скрыть береговую линию' : 'Наложить береговую линию';
    q('#coastToggle').disabled = !info.lines.length;
    row.hidden = false; q('#coastHint').textContent = info.in_view ? '· в кадре должен быть берег' : '';
    if (!info.data) {say(info.note, 'error'); return}
    if (info.in_view) say(`В кадре должен быть берег: ближайший в ${info.nearest_km} км, азимут ${info.azimuth_deg.toFixed(0)}°. ${st.shown ? 'Перетаскивайте линию по картинке: влево-вправо — курс, вверх-вниз — наклон камеры.' : 'Наложите береговую линию на картинку и проверьте, совпадает ли она.'}`, st.shown ? 'done' : 'working');
    else say(info.note || '', 'idle');
  }

  // Ask the server what land the current frame should show.  Cheap; repeated after seeks because the drone moves.
  async function check(force = false) {
    if (!ready()) return;
    const ticket = ++st.requested;
    if (st.video !== viewer.video_path) {
      st.video = viewer.video_path; st.info = null; st.shown = false; st.offered = false; st.sticky = 0;
      const saved = viewer.pose_offsets || {};
      st.saved = {pitch: Number(saved.pitch_offset_deg) || 0, roll: Number(saved.roll_offset_deg) || 0};
      st.pitch = st.saved.pitch; st.roll = st.saved.roll; q('#coastRow').hidden = true; q('#coastHint').textContent = '';
    }
    let info;
    try {info = await api('/api/coast-view', {method: 'POST', body: JSON.stringify(params())})} catch (error) {q('#coastRow').hidden = false; say(error.message, 'error'); return}
    if (ticket !== st.requested || !ready() || info.camera === undefined) return;
    st.info = info; st.focalMm = seriesAt('focal_length_mm', info.time_s);
    if (!info.in_view && !info.maybe_in_view && !force && info.data) {
      // No land around: the row stays hidden unless something was already being adjusted.
      q('#coastRow').hidden = !st.shown && Math.abs(st.pitch) < 1e-4 && Math.abs(st.roll) < 1e-4;
      st.lastDraw = ''; q('#coastHint').textContent = ''; if (!q('#coastRow').hidden) describe();
      return;
    }
    describe(); st.lastDraw = '';
  }
  function schedule() {clearTimeout(st.timer); st.timer = setTimeout(() => check().catch(() => {}), 450)}

  async function auto() {
    if (!ready()) return;
    const button = q('#coastAuto');
    button.disabled = true; say('Ищу линию горизонта и берег в кадре…', 'working', true);
    try {
      const r = await api('/api/coast-align', {method: 'POST', body: JSON.stringify(params())});
      const parts = [];
      if (r.horizon) {
        st.pitch = r.pitch_offset_deg; st.roll = r.roll_offset_deg;
        parts.push(`по горизонту: наклон ${r.horizon.pitch_deg.toFixed(2)}° (в логе ${r.log_pitch_deg.toFixed(2)}°), крен ${r.horizon.roll_deg.toFixed(2)}°`);
      }
      if (r.coast && r.quality !== 'failed') {
        form.elements.yaw_offset_deg.value = r.yaw_offset_deg.toFixed(1); updateFrameOverlay();
        if (!r.horizon) st.pitch = r.pitch_offset_deg;
        parts.push(`по берегу: поправка курса ${signed(r.yaw_offset_deg, 1)}° (${r.quality === 'good' ? 'уверенно' : 'неуверенно — проверьте глазами'})`);
      }
      st.shown = true; await check(true);
      const text = parts.length ? `${parts.join(' · ')}.${r.notes.length ? ' ' + r.notes.join('; ') + '.' : ''} Проверьте наложение и нажмите «Сохранить выравнивание».` : (r.notes.join('; ') || 'Выровнять не удалось') + '.';
      say(text, parts.length ? 'done' : 'error', true); describeButtons();
    } catch (error) {say(error.message, 'error', true)} finally {button.disabled = false}
  }
  function describeButtons() {
    const changed = Math.abs(st.pitch - st.saved.pitch) > 1e-4 || Math.abs(st.roll - st.saved.roll) > 1e-4;
    q('#coastOffsets').textContent = `курс ${signed(yawOffset(), 1)}° · наклон ${signed(st.pitch)}° · крен ${signed(st.roll)}°${changed ? ' · не сохранено' : ''}`;
    q('#coastSave').disabled = !changed;
    q('#coastToggle').textContent = st.shown ? 'Скрыть береговую линию' : 'Наложить береговую линию';
    if (typeof updateFrameOverlay === 'function') updateFrameOverlay();
  }
  async function save() {
    if (!ready()) return;
    try {
      const r = await api('/api/pose-offsets', {method: 'POST', body: JSON.stringify({video_path: viewer.video_path, pitch_offset_deg: st.pitch, roll_offset_deg: st.roll,
        yaw_offset_deg: yawOffset(), measured_at_s: Number((video.currentTime || 0).toFixed(3)), method: 'coast overlay'})});
      st.saved = {pitch: st.pitch, roll: st.roll}; viewer.pose_offsets = {pitch_offset_deg: st.pitch, roll_offset_deg: st.roll};
      describeButtons(); say(`Выравнивание сохранено (${baseName(r.path)}): наклон и крен учитываются во всех расчётах этого видео, курс — в поле «Поправка курса».`, 'done', true);
      DV.session.touch();
    } catch (error) {say(`Не удалось сохранить рядом с видео: ${error.message}`, 'error', true)}
  }
  function reset() {st.pitch = st.saved.pitch; st.roll = st.saved.roll; describeButtons(); st.lastDraw = ''; say('Наклон и крен возвращены к сохранённым.', 'idle')}

  function bind() {
    const dragLayer = q('#coastDrag');
    if (!dragLayer) return;
    q('#coastToggle').onclick = () => {st.shown = !st.shown; st.lastDraw = ''; describe(); draw()};
    q('#coastAuto').onclick = () => auto();
    q('#coastSave').onclick = () => save();
    q('#coastReset').onclick = () => reset();
    dragLayer.addEventListener('pointerdown', event => {
      if (!st.info) return;
      st.drag = {x: event.clientX, y: event.clientY, yaw: yawOffset(), pitch: st.pitch}; try {dragLayer.setPointerCapture(event.pointerId)} catch (e) {} event.preventDefault();
    });
    dragLayer.addEventListener('pointermove', event => {
      if (!st.drag) return;
      const cam = camera(), scale = cam.width / q('#videoWrap').clientWidth, fine = event.shiftKey ? 0.2 : 1;
      // Dragging the picture's coast right means the camera looks further left; down means it looks higher.
      const yaw = st.drag.yaw - (event.clientX - st.drag.x) * scale / cam.focal / rad * fine;
      st.pitch = Math.round((st.drag.pitch + (event.clientY - st.drag.y) * scale / cam.focal / rad * fine) * 1000) / 1000;
      form.elements.yaw_offset_deg.value = (Math.round(yaw * 100) / 100).toFixed(2);
      describeButtons();
    });
    ['pointerup', 'pointercancel'].forEach(name => dragLayer.addEventListener(name, () => {if (st.drag) {st.drag = null; DV.session.touch(); schedule()}}));
    ['seeked', 'pause', 'loadeddata'].forEach(name => video.addEventListener(name, schedule));
    form.elements.yaw_offset_deg.addEventListener('input', () => {st.lastDraw = ''; if (st.info) describeButtons()});
    form.elements.takeoff_height_m.addEventListener('change', schedule);
    requestAnimationFrame(loop);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bind); else setTimeout(bind, 0);

  // Depression of the sea horizon for the current height, degrees (same formula as coastdata.horizon_dip_deg).
  function dip() {
    if (!ready()) return 0;
    const h = seriesAt('height') + (Number(form.elements.takeoff_height_m.value) || 0);
    return h > 0 ? Math.sqrt(2 * h / (6371000 / 0.87)) / rad : 0;
  }
  return {check, pitchOffset: () => st.pitch, rollOffset: () => st.roll, dip, state: st};
})();
