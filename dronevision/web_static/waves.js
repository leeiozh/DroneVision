// DroneVision: «Течения и спектр волн» — the panel at the bottom of the page (dronevision/wavefield.py).
// Loaded after tools.js and before app.js; app.js globals are used only inside functions.
// One button, no parameters: the server picks the cells, the resolution and the camera attitude itself.
DV.waves = (() => {
  const q = selector => document.querySelector(selector);
  const compass = degrees => ['С', 'ССВ', 'СВ', 'ВСВ', 'В', 'ВЮВ', 'ЮВ', 'ЮЮВ', 'Ю', 'ЮЮЗ', 'ЮЗ', 'ЗЮЗ', 'З', 'ЗСЗ', 'СЗ', 'ССЗ'][Math.round(degrees / 22.5) % 16];
  const QUALITY = {good: ['#2fd08a', 'надёжно'], weak: ['#ffb02e', 'неуверенно'], failed: ['#ff5b45', 'нет решения']};
  let current = null, hover = -1, background = null;

  function fit(canvas, height) {
    const dpr = devicePixelRatio || 1, w = Math.max(240, canvas.clientWidth);
    canvas.style.height = `${height}px`;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(height * dpr)) {canvas.width = Math.round(w * dpr); canvas.height = Math.round(height * dpr)}
    const g = canvas.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, height);
    return [g, w, height];
  }
  function arrow(g, x, y, dx, dy, color, width = 2.5) {
    const length = Math.hypot(dx, dy); if (length < 1) return;
    const ux = dx / length, uy = dy / length, head = Math.min(10, length * 0.45);
    for (const [stroke, w] of [['rgba(0,0,0,.8)', width + 2.5], [color, width]]) {
      g.strokeStyle = stroke; g.fillStyle = stroke; g.lineWidth = w; g.lineCap = 'round';
      g.beginPath(); g.moveTo(x, y); g.lineTo(x + dx - ux * head * 0.6, y + dy - uy * head * 0.6); g.stroke();
      g.beginPath(); g.moveTo(x + dx, y + dy); g.lineTo(x + dx - ux * head - uy * head * 0.5, y + dy - uy * head + ux * head * 0.5);
      g.lineTo(x + dx - ux * head + uy * head * 0.5, y + dy - uy * head - ux * head * 0.5); g.closePath(); g.fill();
      if (w > width) {g.lineWidth = 2; g.stroke()}
    }
  }

  // --- map: mean rectified image, cells and current vectors ---
  function mapGeometry(w, h) {
    const b = current.map.bounds_m, spanE = b.east[1] - b.east[0], spanN = b.north[1] - b.north[0], scale = Math.min(w / spanE, h / spanN);
    const x0 = (w - spanE * scale) / 2, y0 = (h - spanN * scale) / 2;
    return {scale, x: e => x0 + (e - b.east[0]) * scale, y: n => y0 + (b.north[1] - n) * scale, x0, y0, width: spanE * scale, height: spanN * scale};
  }
  function drawMap() {
    const canvas = q('#wavesMap'); if (!current || !canvas) return;
    const b = current.map.bounds_m, aspect = (b.north[1] - b.north[0]) / (b.east[1] - b.east[0]);
    const [g, w, h] = fit(canvas, Math.round(Math.min(Math.max(canvas.clientWidth * aspect, 260), 620)));
    const m = mapGeometry(w, h);
    g.fillStyle = '#101514'; g.fillRect(0, 0, w, h);
    if (background && background.complete && background.naturalWidth) g.drawImage(background, m.x0, m.y0, m.width, m.height);
    const speeds = current.cells.filter(c => c.speed_m_s != null).map(c => c.speed_m_s), top = Math.max(0.1, ...speeds);
    const unit = current.cell_m * m.scale * 0.42 / top;      // the fastest vector spans 0.42 of a cell
    g.font = '600 11px system-ui';
    current.cells.forEach((c, i) => {
      const x = m.x(c.east_m), y = m.y(c.north_m), half = c.size_m * m.scale / 2, [color] = QUALITY[c.quality];
      g.strokeStyle = i === hover ? '#fff' : 'rgba(255,255,255,.35)'; g.lineWidth = i === hover ? 2 : 1; g.strokeRect(x - half, y - half, 2 * half, 2 * half);
      if (c.speed_m_s == null) {g.fillStyle = color; g.fillText('×', x - 3, y + 4); return}
      arrow(g, x, y, c.u_east_m_s * unit, -c.v_north_m_s * unit, color);
    });
    // camera and scale
    const cam = current.camera_m; g.fillStyle = '#fff'; g.strokeStyle = '#000'; g.lineWidth = 2;
    const cx = m.x(cam[0]), cy = m.y(cam[1]);
    if (cx > 0 && cx < w && cy > 0 && cy < h) {g.beginPath(); g.arc(cx, cy, 5, 0, 7); g.stroke(); g.fill(); g.fillText('дрон', cx + 8, cy + 4)}
    const nice = [0.05, 0.1, 0.2, 0.5, 1, 2].find(v => v >= top * 0.6) || 2;
    arrow(g, 16, h - 18, nice * unit, 0, '#fff'); g.fillStyle = '#fff'; g.shadowColor = '#000'; g.shadowBlur = 3;
    g.fillText(`${nice} м/с`, 16, h - 28);
    const bar = [10, 20, 50, 100, 200, 500, 1000].find(v => v * m.scale >= 60) || 1000;
    g.fillRect(w - 16 - bar * m.scale, h - 16, bar * m.scale, 3); g.fillText(`${bar} м`, w - 16 - bar * m.scale, h - 22);
    g.fillText('С ↑', w - 34, 18); g.shadowBlur = 0;
  }
  function mapHover(event) {
    const canvas = q('#wavesMap'), r = canvas.getBoundingClientRect(), m = mapGeometry(r.width, r.height);
    const index = current.cells.findIndex(c => Math.abs(m.x(c.east_m) - (event.clientX - r.left)) <= c.size_m * m.scale / 2 && Math.abs(m.y(c.north_m) - (event.clientY - r.top)) <= c.size_m * m.scale / 2);
    if (index !== hover) {hover = index; drawMap(); highlightRow()}
  }
  function highlightRow() {q('#wavesTable').querySelectorAll('tr[data-cell]').forEach(row => row.classList.toggle('hover', Number(row.dataset.cell) === hover))}

  // --- directional wavenumber spectrum E(kx, ky): north up, the direction the waves travel to ---
  function drawSpectrum() {
    const canvas = q('#wavesSpectrum'), s = current.waves.kxky; if (!s || !canvas) return;
    const [g, w, h] = fit(canvas, Math.min(Math.max(canvas.clientWidth, 240), 420)), size = Math.min(w, h) - 36, x0 = (w - size) / 2, y0 = 8, n = s.rows.length;
    const image = new ImageData(n, n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const v = s.rows[j][i] / 255, o = (j * n + i) * 4;           // dark blue -> cyan -> yellow -> white
      image.data[o] = 255 * Math.min(1, Math.max(0, 2.2 * v - 0.9)); image.data[o + 1] = 255 * Math.min(1, Math.max(0, 1.6 * v - 0.25));
      image.data[o + 2] = 255 * Math.min(1, 0.25 + 1.2 * v - (v > 0.7 ? (v - 0.7) * 1.5 : 0)); image.data[o + 3] = 255;
    }
    const buffer = document.createElement('canvas'); buffer.width = buffer.height = n; buffer.getContext('2d').putImageData(image, 0, 0);
    g.imageSmoothingEnabled = true; g.drawImage(buffer, x0, y0, size, size);
    const centre = [x0 + size / 2, y0 + size / 2], kMax = s.half * s.dk_rad_m, pxPerK = size / 2 / kMax;
    g.strokeStyle = 'rgba(255,255,255,.45)'; g.fillStyle = '#fff'; g.font = '11px system-ui'; g.lineWidth = 1;
    [50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1].filter(l => 2 * Math.PI / l < kMax * 0.98).slice(0, 4).forEach(l => {
      const radius = 2 * Math.PI / l * pxPerK; g.beginPath(); g.arc(centre[0], centre[1], radius, 0, 7); g.stroke(); g.fillText(`${l} м`, centre[0] + radius * 0.71 + 2, centre[1] - radius * 0.71);
    });
    g.beginPath(); g.moveTo(x0, centre[1]); g.lineTo(x0 + size, centre[1]); g.moveTo(centre[0], y0); g.lineTo(centre[0], y0 + size); g.stroke();
    g.fillText('С', centre[0] + 4, y0 + 12); g.fillText('В', x0 + size - 12, centre[1] - 4);
    // current direction
    const sum = current.summary, u = sum.u_east_m_s, v = sum.v_north_m_s, length = Math.hypot(u, v) || 1;
    arrow(g, centre[0], centre[1], u / length * size * 0.18, -v / length * size * 0.18, '#ff5b45', 2);
    g.fillStyle = 'var(--muted)'; g.fillStyle = '#69736f'; g.textAlign = 'center';
    g.fillText('куда идут волны · кольца — длина волны · красная стрелка — течение', w / 2, y0 + size + 18); g.textAlign = 'left';
  }

  function plot(canvas, xs, ys, options) {
    const [g, w, h] = fit(canvas, 150), pad = {l: 34, r: 10, t: 16, b: 26}, pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
    const x0 = options.x0 ?? Math.min(...xs), x1 = options.x1 ?? Math.max(...xs), X = x => pad.l + (x - x0) / (x1 - x0) * pw, Y = y => pad.t + (1 - y) * ph;
    g.font = '11px system-ui'; g.fillStyle = '#69736f'; g.strokeStyle = '#daddd6'; g.lineWidth = 1;
    g.strokeRect(pad.l, pad.t, pw, ph); g.fillText(options.title, pad.l, 11);
    options.ticks.forEach(t => {if (t < x0 || t > x1) return; g.beginPath(); g.moveTo(X(t), pad.t + ph); g.lineTo(X(t), pad.t + ph + 4); g.stroke(); g.textAlign = 'center'; g.fillText(options.label ? options.label(t) : String(t), X(t), h - 8)});
    g.textAlign = 'left'; g.strokeStyle = '#165b48'; g.lineWidth = 2; g.beginPath();
    xs.forEach((x, i) => {if (x < x0 || x > x1) return; const px = X(x), py = Y(Math.min(1, ys[i])); i && xs[i - 1] >= x0 ? g.lineTo(px, py) : g.moveTo(px, py)});
    g.stroke();
    (options.marks || []).forEach(([x, text]) => {if (x < x0 || x > x1) return; g.strokeStyle = '#c74747'; g.lineWidth = 1; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(X(x), pad.t); g.lineTo(X(x), pad.t + ph); g.stroke(); g.setLineDash([]); g.fillStyle = '#c74747'; g.fillText(text, Math.min(X(x) + 4, w - 70), pad.t + 12)});
  }
  function drawPlots() {
    const wv = current.waves; if (!wv.frequency) return;
    const f = wv.frequency.hz, top = Math.min(Math.max(...f), Math.max(1.2, 3 * wv.observed_peak_frequency_hz));
    plot(q('#wavesFrequency'), f, wv.frequency.energy, {title: 'Частотный спектр яркости (как видит неподвижный наблюдатель), Гц', x0: 0, x1: top,
      ticks: [0, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3], marks: [[wv.observed_peak_frequency_hz, `пик ${wv.observed_peak_frequency_hz.toFixed(2)} Гц`]]});
    plot(q('#wavesDirection'), wv.direction.to_deg, wv.direction.energy, {title: 'Распределение энергии по направлению (куда идут волны)', x0: 0, x1: 360,
      ticks: [0, 90, 180, 270, 360], label: t => ['С', 'В', 'Ю', 'З', 'С'][t / 90], marks: [[wv.mean_direction_to_deg, `среднее ${wv.mean_direction_to_deg.toFixed(0)}°`]]});
    const k = wv.wavenumber.k_rad_m.map(v => Math.log10(2 * Math.PI / v));
    plot(q('#wavesLength'), k.slice().reverse(), wv.wavenumber.energy.slice().reverse(), {title: 'Распределение энергии по длине волны, м', ticks: [-1, -0.699, -0.301, 0, 0.301, 0.699, 1, 1.301, 1.699, 2],
      label: t => String(Math.round(10 ** t * 100) / 100), marks: [[Math.log10(wv.peak_wavelength_m), `пик ${wv.peak_wavelength_m} м`]]});
  }

  function table() {
    const body = current.cells.map((c, i) => {
      const [color, label] = QUALITY[c.quality];
      if (c.speed_m_s == null) return `<tr data-cell="${i}"><td>${i + 1}</td><td>${c.range_m}</td><td colspan="5">—</td><td><i style="background:${color}"></i>${label}</td></tr>`;
      return `<tr data-cell="${i}"><td>${i + 1}</td><td>${c.range_m}</td><td><b>${c.speed_m_s.toFixed(2)}</b></td><td>${c.direction_to_deg.toFixed(0)}° ${compass(c.direction_to_deg)}</td><td>${c.u_east_m_s.toFixed(2)}</td><td>${c.v_north_m_s.toFixed(2)}</td><td>${c.halves_difference_m_s == null ? '—' : c.halves_difference_m_s.toFixed(2)}</td><td><i style="background:${color}"></i>${label}${c.one_direction ? ' · волны одного направления' : ''}</td></tr>`;
    }).join('');
    q('#wavesTable').innerHTML = `<table><thead><tr><th>№</th><th>дальность, м</th><th>скорость, м/с</th><th>куда</th><th>на восток</th><th>на север</th><th title="Расхождение между первой и второй половиной записи">разброс, м/с</th><th>оценка</th></tr></thead><tbody>${body}</tbody></table>`;
    q('#wavesTable').querySelectorAll('tr[data-cell]').forEach(row => {row.onmouseenter = () => {hover = Number(row.dataset.cell); drawMap(); highlightRow()}});
  }

  function show(result, restored = false) {
    current = result; hover = -1;
    const section = q('#wavesResult'), s = result.summary, w = result.waves || {}, a = result.attitude, [t0, t1] = result.interval_s;
    section.hidden = false;
    q('#wavesTitle').textContent = `${s.speed_m_s.toFixed(2)} м/с на ${s.direction_to_text} (${s.direction_to_deg.toFixed(0)}°) · ${clock(t0)}–${clock(t1)}`;
    const lines = [`Поверхностное течение (верхние ~${s.depth_of_measurement_m ?? 0.5} м): на восток ${s.u_east_m_s.toFixed(2)}, на север ${s.v_north_m_s.toFixed(2)} м/с · участков ${s.cells_used} из ${s.cells_total}${s.scatter_m_s != null ? ` · разброс между участками ${s.scatter_m_s.toFixed(2)} м/с` : ''}.`];
    if (w.systems && w.systems.length) lines.push(`Волнение: ${w.systems.map(x => `с ${compass(x.direction_from_deg)} (идёт на ${x.direction_to_deg.toFixed(0)}°), длина ${x.wavelength_m} м, период ${x.period_s} с, доля ${Math.round(x.share * 100)}%`).join('; ')}. Пик спектра ${w.peak_wavelength_m} м / ${w.peak_period_s} с, угловой разброс ${w.directional_spread_deg}°.`);
    const checks = [];
    if (a.horizon) checks.push(`наклон камеры по горизонту ${a.horizon_pitch_deg.toFixed(2)}° (в логе ${a.log_pitch_deg.toFixed(2)}°)`); else checks.push('горизонт не найден — углы из лога');
    checks.push(a.yaw_from_image ? `курс отслежен по деталям на горизонте, дрейф ${a.yaw_drift_deg >= 0 ? '+' : ''}${a.yaw_drift_deg.toFixed(2)}° за запись` : 'курс из лога');
    if (s.scale) checks.push(`масштаб по дисперсии волн ${s.scale.toFixed(2)} (1.00 — геометрия точна)`);
    lines.push(`Проверки: ${checks.join(' · ')} · ${result.frames} кадров, ${result.frame_rate_hz} кадр/с, сетка ${result.resolution_m_per_px} м, участок ${result.cell_m} м.`);
    q('#wavesSummary').innerHTML = lines.map(text => `<p>${text}</p>`).join('') + (result.notes.length ? `<p class="waves-notes">Замечания: ${result.notes.join('; ')}.</p>` : '') +
      '<p class="waves-caveat">Течение найдено по сдвигу дисперсионного соотношения ветровых волн (глубокая вода). Спектры — спектры яркости изображения: длины, периоды и направления достоверны, уровни энергии относительные.</p>';
    q('#wavesPath').textContent = `${result.path}\n${result.csv_path}`;
    const links = q('#wavesLinks'); links.innerHTML = '';
    links.append(DV.exportRow({path: result.map.path}, pngLink(`/api/raster?path=${encodeURIComponent(result.map.path)}&saved=1`, 'Открыть карту PNG')));
    table();
    background = new Image(); background.onload = drawMap; background.src = `/api/raster?path=${encodeURIComponent(result.map.path)}&saved=1&t=${Date.now()}`;
    drawMap(); drawSpectrum(); drawPlots();
    if (!restored) section.scrollIntoView({behavior: 'smooth', block: 'start'});
  }
  function redraw() {if (current && !q('#wavesResult').hidden) {drawMap(); drawSpectrum(); drawPlots()}}

  async function run() {
    const button = q('#waveCurrents'), status = q('#wavesStatus');
    let ok = false; try {ok = !!viewer} catch (e) {}
    if (!ok) {activity(status, 'Сначала откройте видео.', 'error'); return}
    const range = nearestSelectedInterval(), body = {video_path: viewer.video_path, project_dir: form.elements.project_dir.value,
      takeoff_height_m: Number(form.elements.takeoff_height_m.value) || 0, horizontal_fov_deg: Number(form.elements.horizontal_fov_deg.value) || 0, yaw_offset_deg: Number(form.elements.yaw_offset_deg.value) || 0};
    if (range) {body.start_s = range[0]; body.end_s = range[1]}
    button.disabled = true; activity(status, range ? `Считаю течения по интервалу ${clock(range[0])}–${clock(range[1])}…` : 'Считаю течения по всему видео (до 90 с)…', 'working');
    const timer = setInterval(async () => {try {const p = await api('/api/wave-progress'); if (p.running) activity(status, `${p.message} · ${Math.round(p.progress * 100)}%`, 'working')} catch (e) {}}, 1500);
    try {
      const result = await api('/api/wave-currents', {method: 'POST', body: JSON.stringify(body)});
      show(result); DV.session.setResult('waves', {path: result.path});
      activity(status, `Готово: течение ${result.summary.speed_m_s.toFixed(2)} м/с на ${result.summary.direction_to_text}. Результат — в панели внизу.`, 'done');
    } catch (error) {activity(status, error.message, 'error')} finally {clearInterval(timer); button.disabled = false}
  }
  async function restore(saved) {show(await api(`/api/wave-result?path=${encodeURIComponent(saved.path)}`), true)}

  function bind() {
    const button = q('#waveCurrents'); if (!button) return;
    button.onclick = () => run();
    q('#wavesMap').addEventListener('mousemove', event => {if (current) mapHover(event)});
    q('#wavesMap').addEventListener('mouseleave', () => {hover = -1; if (current) {drawMap(); highlightRow()}});
    window.addEventListener('resize', redraw);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bind); else setTimeout(bind, 0);
  return {show, restore, hide: () => {q('#wavesResult').hidden = true; current = null}};
})();
