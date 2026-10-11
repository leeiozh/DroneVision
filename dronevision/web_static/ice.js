// DroneVision: «Дрейф льдин» (experimental) — the panel at the bottom of the page (dronevision/icedrift.py).
// Loaded after waves.js (whose drawing helpers it uses) and before app.js; app.js globals are used only
// inside functions.  One button, no parameters.
DV.ice = (() => {
  const q = selector => document.querySelector(selector);
  const {fit, arrow, zoomable, zoomed} = DV.waves.draw, view = {z: 1, px: 0, py: 0}, compass = DV.waves.compass, QUALITY = DV.waves.QUALITY, VERDICT = DV.waves.VERDICT;
  let current = null, hover = -1, background = null;
  // With the box ticked the common turn of the field about the drone is taken out, otherwise shown as measured.
  const turned = () => q('#iceTurn').checked;
  const vector = c => turned() ? [c.u_east_m_s, c.v_north_m_s] : [c.measured_u_east_m_s, c.measured_v_north_m_s];

  function geometry(w, h) {
    const b = current.map.bounds_m, spanE = b.east[1] - b.east[0], spanN = b.north[1] - b.north[0], scale = Math.min(w / spanE, h / spanN);
    const x0 = (w - spanE * scale) / 2, y0 = (h - spanN * scale) / 2;
    return zoomed({scale, x: e => x0 + (e - b.east[0]) * scale, y: n => y0 + (b.north[1] - n) * scale, x0, y0, width: spanE * scale, height: spanN * scale}, view);
  }
  function draw() {
    const canvas = q('#iceMap'); if (!current || !canvas) return;
    const b = current.map.bounds_m, aspect = (b.north[1] - b.north[0]) / (b.east[1] - b.east[0]);
    const [g, w, h] = fit(canvas, Math.round(Math.min(Math.max(canvas.clientWidth * aspect, 300), 760)));
    const m = geometry(w, h), camera = current.camera_m;
    g.fillStyle = '#101514'; g.fillRect(0, 0, w, h);
    if (background && background.complete && background.naturalWidth) g.drawImage(background, m.x0, m.y0, m.width, m.height);
    const cells = current.cells, speeds = cells.map(c => Math.hypot(...vector(c))).sort((a, b) => a - b);
    const top = Math.max(0.02, speeds[Math.floor(0.95 * (speeds.length - 1))]);
    const unit = current.cell_m * m.scale * 0.9 / top;       // a typical fast vector is about one cell long
    // Paths of the tracked points over the record: thin lines under the arrows.
    const duration = current.interval_s[1] - current.interval_s[0], omega = turned() ? (current.summary.rotation_deg_s || 0) * Math.PI / 180 : 0;
    g.strokeStyle = 'rgba(120,220,255,.55)'; g.lineWidth = 1; g.beginPath();
    current.tracks.forEach(([e, n, u, v]) => {
      const cu = u + omega * (n - camera[1]), cv = v - omega * (e - camera[0]);
      g.moveTo(m.x(e), m.y(n)); g.lineTo(m.x(e + cu * duration), m.y(n + cv * duration));
    });
    g.stroke();
    g.font = '600 11px system-ui';
    cells.forEach((c, i) => {
      const x = m.x(c.east_m), y = m.y(c.north_m), half = c.size_m * m.scale / 2, [u, v] = vector(c);
      const color = c.outlier ? QUALITY.failed[0] : QUALITY[c.quality][0];
      if (i === hover) {g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.strokeRect(x - half, y - half, 2 * half, 2 * half)}
      arrow(g, x, y, u * unit, -v * unit, color, 2);
    });
    const cx = m.x(camera[0]), cy = m.y(camera[1]);
    g.fillStyle = '#fff'; g.strokeStyle = '#000'; g.lineWidth = 2; g.beginPath(); g.arc(cx, cy, 5, 0, 7); g.stroke(); g.fill();
    g.shadowColor = '#000'; g.shadowBlur = 3; g.fillText('под дроном', cx + 8, cy + 4);
    const nice = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2].find(value => value >= top * 0.6) || 2;
    g.shadowBlur = 0; arrow(g, 16, h - 18, nice * unit, 0, '#fff'); g.fillStyle = '#fff'; g.shadowBlur = 3;
    g.fillText(`${nice} м/с`, 16, h - 28);
    const bar = [10, 20, 50, 100, 200, 500, 1000].find(value => value * m.scale >= 60) || 1000;
    g.fillRect(w - 16 - bar * m.scale, h - 16, bar * m.scale, 3); g.fillText(`${bar} м`, w - 16 - bar * m.scale, h - 22);
    g.fillText('С ↑', w - 34, 18); g.shadowBlur = 0;
  }
  function mapHover(event) {
    const r = q('#iceMap').getBoundingClientRect(), m = geometry(r.width, r.height);
    const index = current.cells.findIndex(c => Math.abs(m.x(c.east_m) - (event.clientX - r.left)) <= c.size_m * m.scale / 2 && Math.abs(m.y(c.north_m) - (event.clientY - r.top)) <= c.size_m * m.scale / 2);
    if (index !== hover) {hover = index; draw(); highlight(true)}
  }
  function highlight(scroll = false) {
    q('#iceTable').querySelectorAll('tr[data-cell]').forEach(row => {const on = Number(row.dataset.cell) === hover; row.classList.toggle('hover', on); if (on && scroll) row.scrollIntoView({block: 'nearest'})});
  }
  function table() {
    const body = current.cells.map((c, i) => {
      const [u, v] = vector(c), speed = Math.hypot(u, v), to = (Math.atan2(u, v) * 180 / Math.PI + 360) % 360;
      const [color, label] = c.outlier ? [QUALITY.failed[0], 'выпадает из общего движения'] : QUALITY[c.quality];
      return `<tr data-cell="${i}"><td>${i + 1}</td><td>${c.east_m.toFixed(0)}</td><td>${c.north_m.toFixed(0)}</td><td><b>${speed.toFixed(3)}</b></td><td>${to.toFixed(0)}° ${compass(to)}</td><td>${u.toFixed(3)}</td><td>${v.toFixed(3)}</td><td>${c.tracks}</td><td>${c.spread_m_s.toFixed(3)}</td><td><i style="background:${color}"></i>${label}</td></tr>`;
    }).join('');
    q('#iceTable').innerHTML = `<table><thead><tr><th>№</th><th>восток, м</th><th>север, м</th><th>скорость, м/с</th><th>куда</th><th>на восток</th><th>на север</th><th title="Сколько прослеженных точек в участке">точек</th><th title="Разброс скоростей точек внутри участка">разброс, м/с</th><th>оценка</th></tr></thead><tbody>${body}</tbody></table>`;
    q('#iceTable').querySelectorAll('tr[data-cell]').forEach(row => {row.onmouseenter = () => {hover = Number(row.dataset.cell); draw(); highlight()}});
  }

  function show(result, restored = false) {
    current = result; hover = -1; view.z = 1; view.px = view.py = 0;
    const section = q('#iceResult'), s = result.summary, [t0, t1] = result.interval_s, verdict = s.reliability || 'good';
    const error = s.uncertainty_m_s != null ? ` ± ${s.uncertainty_m_s.toFixed(2)}` : '';
    section.hidden = false;
    q('#iceTitle').textContent = `${s.speed_m_s.toFixed(2)}${error} м/с на ${s.direction_to_text} (${s.direction_to_deg.toFixed(0)}°) · ${clock(t0)}–${clock(t1)}`;
    const lines = [`<span class="waves-verdict ${verdict}">Оценка: ${VERDICT[verdict]}${s.reasons.length ? ' — ' + s.reasons.join('; ') : ''}.</span>`,
      `Дрейф льдин под дроном: ${s.speed_m_s.toFixed(3)}${error} м/с — на восток ${s.u_east_m_s.toFixed(3)}, на север ${s.v_north_m_s.toFixed(3)}. Прослежено точек: ${s.tracks} из ${s.points_started}, участков ${s.cells}${s.outlier_cells ? `, из них выпадают ${s.outlier_cells}` : ''}; лёд занимает ${s.ice_share_percent}% кадра.`];
    if (s.rotation_deg_s != null) lines.push(`Общий поворот поля ${s.rotation_deg_s.toFixed(3)}°/с — это ${s.rotation_edge_m_s.toFixed(2)} м/с на краю кадра; после его вычитания участки расходятся с общим дрейфом на ${s.field_residual_m_s.toFixed(3)} м/с.`);
    lines.push(`Проверки: ${result.frames} кадров, ${result.frame_rate_hz} кадр/с, пиксель на воде ${s.pixel_m} м, высота ${result.camera_altitude_m} м, участок ${result.cell_m} м; порог различимой скорости ${s.speed_floor_m_s} м/с.`);
    q('#iceSummary').innerHTML = lines.map(text => `<p>${text}</p>`).join('') + (result.notes.length ? `<p class="waves-notes">Замечания: ${result.notes.join('; ')}.</p>` : '');
    const turn = q('#iceTurn'); turn.checked = true; turn.closest('label').hidden = s.rotation_deg_s == null;
    const b = result.map.bounds_m;
    q('#iceMapCaption').textContent = `Первый кадр интервала на поверхности моря, север сверху, ${Math.round(b.east[1] - b.east[0])} × ${Math.round(b.north[1] - b.north[0])} м. Голубые чёрточки — путь прослеженных точек за ${Math.round(t1 - t0)} с; стрелки — медианная скорость точек в квадрате ${result.cell_m} м (зелёные — 6 и более точек с малым разбросом, оранжевые — мало точек или большой разброс, красные — выпадают из общего движения). Наведите на стрелку; колесо — зум, перетаскивание — сдвиг, двойной щелчок — вся карта.`;
    q('#icePath').textContent = `${result.path}\n${result.csv_path}\n${result.tracks_path}`;
    const links = q('#iceLinks'); links.innerHTML = '';
    links.append(DV.exportRow({path: result.map.path}, pngLink(`/api/raster?path=${encodeURIComponent(result.map.path)}&saved=1`, 'Открыть карту PNG')));
    table();
    background = new Image(); background.onload = draw; background.src = `/api/raster?path=${encodeURIComponent(result.map.path)}&saved=1&t=${Date.now()}`;
    draw();
    if (!restored) section.scrollIntoView({behavior: 'smooth', block: 'start'});
  }

  async function run() {
    const button = q('#iceDrift'), status = q('#iceStatus');
    let ok = false; try {ok = !!viewer} catch (e) {}
    if (!ok) {activity(status, 'Сначала откройте видео.', 'error'); return}
    const range = nearestSelectedInterval(), body = {video_path: viewer.video_path, project_dir: form.elements.project_dir.value,
      takeoff_height_m: Number(form.elements.takeoff_height_m.value) || 0, horizontal_fov_deg: Number(form.elements.horizontal_fov_deg.value) || 0, yaw_offset_deg: Number(form.elements.yaw_offset_deg.value) || 0};
    if (range) {body.start_s = range[0]; body.end_s = range[1]}
    button.disabled = true; activity(status, range ? `Слежу за льдинами в интервале ${clock(range[0])}–${clock(range[1])}…` : 'Слежу за льдинами с начала видео (до 120 с)…', 'working');
    const timer = setInterval(async () => {try {const p = await api('/api/wave-progress'); if (p.running) activity(status, `${p.message} · ${Math.round(p.progress * 100)}%`, 'working')} catch (e) {}}, 1500);
    try {
      const result = await api('/api/ice-drift', {method: 'POST', body: JSON.stringify(body)});
      show(result); DV.session.setResult('ice', {path: result.path});
      activity(status, `Готово: дрейф ${result.summary.speed_m_s.toFixed(2)} м/с на ${result.summary.direction_to_text} (${VERDICT[result.summary.reliability]}). Результат — в панели внизу.`, result.summary.reliability === 'unreliable' ? 'error' : 'done');
    } catch (error) {activity(status, error.message, 'error')} finally {clearInterval(timer); button.disabled = false}
  }
  async function restore(saved) {show(await api(`/api/ice-result?path=${encodeURIComponent(saved.path)}`), true)}

  function bind() {
    const button = q('#iceDrift'); if (!button) return;
    button.onclick = () => run();
    q('#iceTurn').onchange = () => {if (current) {draw(); table()}};
    q('#iceMap').addEventListener('mousemove', event => {if (current) mapHover(event)});
    q('#iceMap').addEventListener('mouseleave', () => {hover = -1; if (current) {draw(); highlight()}});
    zoomable(q('#iceMap'), view, () => {if (current) draw()});
    window.addEventListener('resize', () => {if (current && !q('#iceResult').hidden) draw()});
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bind); else setTimeout(bind, 0);
  return {show, restore, hide: () => {q('#iceResult').hidden = true; current = null}};
})();
