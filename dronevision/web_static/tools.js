// DroneVision: image adjustments, geographic lookup and drawing tools for projections/mosaics.
// Loaded before app.js; everything lives under window.DV.
const DV = window.DV = {};
DV.store = {get(key, fallback) {try {const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v)} catch (e) {return fallback}},
  set(key, value) {try {localStorage.setItem(key, JSON.stringify(value))} catch (e) {}}};

// --- Photo adjustments (same pipeline as dronevision/adjust.py) ---
DV.adjust = (() => {
  const specs = [
    {key: 'exposure', label: 'Экспозиция', min: -3, max: 3, step: .02, neutral: 0, fmt: v => `${v > 0 ? '+' : ''}${v.toFixed(2)} EV`},
    {key: 'brightness', label: 'Яркость', min: -.5, max: .5, step: .002, neutral: 0, fmt: v => `${v > 0 ? '+' : ''}${Math.round(v * 100)}`},
    {key: 'contrast', label: 'Контраст', min: .2, max: 3, step: .01, neutral: 1, fmt: v => `×${v.toFixed(2)}`},
    {key: 'gamma', label: 'Гамма', min: .3, max: 3, step: .01, neutral: 1, fmt: v => `γ ${v.toFixed(2)}`},
    {key: 'saturation', label: 'Насыщен.', min: 0, max: 3, step: .01, neutral: 1, fmt: v => `×${v.toFixed(2)}`},
    {key: 'temperature', label: 'Темпер.', min: -1, max: 1, step: .01, neutral: 0, fmt: v => `${v > 0 ? '+' : ''}${Math.round(v * 100)}`},
  ];
  const values = Object.fromEntries(specs.map(s => [s.key, s.neutral]));
  Object.assign(values, DV.store.get('dv.adjust', {}));
  const inputs = {}, outputs = {};
  let filterNodes = null, target = null;
  const neutral = () => specs.every(s => Math.abs(values[s.key] - s.neutral) < 1e-9);
  function buildFilter() {
    const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('width', 0); svg.setAttribute('height', 0); svg.setAttribute('aria-hidden', 'true');
    svg.style.position = 'absolute';
    svg.innerHTML = `<filter id="dvAdjust" color-interpolation-filters="sRGB" x="0" y="0" width="1" height="1">
      <feComponentTransfer data-stage="linear"><feFuncR type="linear"/><feFuncG type="linear"/><feFuncB type="linear"/></feComponentTransfer>
      <feComponentTransfer data-stage="gamma"><feFuncR type="gamma" amplitude="1" offset="0"/><feFuncG type="gamma" amplitude="1" offset="0"/><feFuncB type="gamma" amplitude="1" offset="0"/></feComponentTransfer>
      <feColorMatrix type="matrix"/></filter>`;
    document.body.append(svg);
    const f = svg.querySelector('filter');
    filterNodes = {linear: f.querySelectorAll('[data-stage="linear"] > *'), gamma: f.querySelectorAll('[data-stage="gamma"] > *'), matrix: f.querySelector('feColorMatrix')};
  }
  function updateFilter() {
    if (!filterNodes) return;
    const v = values, gains = [1 + .25 * v.temperature, 1, 1 - .25 * v.temperature], intercept = v.contrast * (v.brightness - .5) + .5;
    filterNodes.linear.forEach((node, i) => {node.setAttribute('slope', v.contrast * gains[i] * 2 ** v.exposure); node.setAttribute('intercept', intercept)});
    filterNodes.gamma.forEach(node => node.setAttribute('exponent', 1 / v.gamma));
    const s = v.saturation;
    filterNodes.matrix.setAttribute('values', [
      .213 + .787 * s, .715 - .715 * s, .072 - .072 * s, 0, 0,
      .213 - .213 * s, .715 + .285 * s, .072 - .072 * s, 0, 0,
      .213 - .213 * s, .715 - .715 * s, .072 + .928 * s, 0, 0, 0, 0, 0, 1, 0].join(' '));
    if (target) target.style.filter = neutral() ? '' : 'url(#dvAdjust)';
  }
  function set(key, value, save = true) {
    const spec = specs.find(s => s.key === key);
    values[key] = Math.max(spec.min, Math.min(spec.max, Number(value)));
    if (inputs[key]) {inputs[key].value = values[key]; outputs[key].textContent = spec.fmt(values[key]); inputs[key].closest('.adjust-slider').classList.toggle('changed', Math.abs(values[key] - spec.neutral) > 1e-9)}
    updateFilter();
    if (save) DV.store.set('dv.adjust', values);
  }
  function init(panel, video) {
    target = video; buildFilter();
    panel.innerHTML = '<div class="adjust-title"><b>Коррекция</b><small>видео, проекция и мозаика</small><button type="button" data-reset>Сброс</button></div>';
    specs.forEach(spec => {
      const box = document.createElement('label'), input = document.createElement('input'), output = document.createElement('output');
      box.className = 'adjust-slider'; box.title = `${spec.label}: двойной щелчок — сброс`;
      Object.assign(input, {type: 'range', min: spec.min, max: spec.max, step: spec.step, value: values[spec.key]});
      input.setAttribute('orient', 'vertical'); input.setAttribute('aria-label', spec.label);
      input.addEventListener('input', () => set(spec.key, input.value));
      input.addEventListener('dblclick', () => set(spec.key, spec.neutral));
      const name = document.createElement('span'); name.textContent = spec.label;
      box.append(name, input, output); panel.append(box);
      inputs[spec.key] = input; outputs[spec.key] = output;
    });
    panel.querySelector('[data-reset]').onclick = () => specs.forEach(s => set(s.key, s.neutral));
    specs.forEach(s => set(s.key, values[s.key], false));
  }
  // Restore all sliders at once (saved session); unknown or missing keys keep their current value.
  const setAll = saved => specs.forEach(s => {if (saved && Number.isFinite(Number(saved[s.key]))) set(s.key, saved[s.key])});
  return {init, params: () => ({...values}), neutral, specs, setAll};
})();

// --- ENU metres -> degrees via the server-side lookup grid (bilinear; exact to < 1 mm) ---
DV.lonlat = (grid, e, n) => {
  if (!grid) return null;
  const k = grid.nodes - 1, fx = (e - grid.east[0]) / (grid.east[1] - grid.east[0]) * k, fy = (n - grid.north[0]) / (grid.north[1] - grid.north[0]) * k;
  const i = Math.max(0, Math.min(k - 1, Math.floor(fx))), j = Math.max(0, Math.min(k - 1, Math.floor(fy))), u = fx - i, v = fy - j;
  const at = a => (a[j][i] * (1 - u) + a[j][i + 1] * u) * (1 - v) + (a[j + 1][i] * (1 - u) + a[j + 1][i + 1] * u) * v;
  return {lat: at(grid.lat), lon: at(grid.lon)};
};
DV.formatLatLon = p => p ? `${Math.abs(p.lat).toFixed(6)}° ${p.lat >= 0 ? 'N' : 'S'} · ${Math.abs(p.lon).toFixed(6)}° ${p.lon >= 0 ? 'E' : 'W'}` : '';

// --- Overlay switches shared by every projection viewer ---
DV.overlays = Object.assign({land: false, coast: true, vectors: true}, DV.store.get('dv.overlays', {}));
DV.viewers = new Set();
DV.setOverlay = (key, value) => {DV.overlays[key] = value; DV.store.set('dv.overlays', DV.overlays); DV.viewers.forEach(v => v.syncOverlays())};

// --- Annotation store: shapes in ENU metres per image; slot = object id (colour), shared across images ---
// Designed for the next step (displacement/velocity between two frames): the same slot on two images
// of different times is the same tracked object, see DV.annotationStore.pairs().
DV.COLORS = [['#e6194b', 'красный'], ['#3cb44b', 'зелёный'], ['#ffe119', 'жёлтый'], ['#4363d8', 'синий'], ['#f58231', 'оранжевый'],
  ['#911eb4', 'фиолетовый'], ['#42d4f4', 'голубой'], ['#f032e6', 'пурпурный'], ['#bfef45', 'салатовый'], ['#fabed4', 'розовый']];
DV.annotationStore = {
  images: DV.store.get('dv.annotations', {}),
  entry(variant) {
    const key = variant.path;
    if (!this.images[key]) this.images[key] = {shapes: []};
    Object.assign(this.images[key], {kind: variant.frames ? 'mosaic' : 'projection',
      times_s: variant.frames ? variant.frames.map(f => f.time_s) : [variant.time_s], saved: Date.now()});
    return this.images[key];
  },
  save() {
    const keys = Object.keys(this.images).sort((a, b) => (this.images[b].saved || 0) - (this.images[a].saved || 0));
    keys.slice(40).forEach(k => delete this.images[k]);  // keep localStorage small
    DV.store.set('dv.annotations', this.images);
    window.dispatchEvent(new Event('dv:annotations-changed'));
  },
  // Objects (slots) marked on two images: the input for displacement and velocity estimation.
  pairs(pathA, pathB) {
    const a = this.images[pathA], b = this.images[pathB];
    if (!a || !b) return [];
    return a.shapes.filter(s => b.shapes.some(t => t.slot === s.slot)).map(s => ({slot: s.slot, a: s, b: b.shapes.find(t => t.slot === s.slot), dt: (b.times_s[0] ?? 0) - (a.times_s[0] ?? 0)}));
  },
};

DV.attachTopResize = (element, minimum = 38, maximum = () => Math.max(240, innerHeight * .8)) => {
  const handle = document.createElement('div'); handle.className = 'panel-top-resize'; handle.title = 'Потяните вверх или вниз, чтобы изменить высоту'; element.prepend(handle);
  let startY = 0, startHeight = 0;
  handle.addEventListener('pointerdown', event => {if (event.button !== 0) return; event.preventDefault(); startY = event.clientY; startHeight = element.getBoundingClientRect().height; handle.setPointerCapture(event.pointerId); element.classList.add('resizing')});
  handle.addEventListener('pointermove', event => {if (!handle.hasPointerCapture(event.pointerId)) return; const limit = typeof maximum === 'function' ? maximum() : maximum; element.style.height = `${Math.max(minimum, Math.min(limit, startHeight - (event.clientY - startY)))}px`});
  const finish = event => {if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId); element.classList.remove('resizing')};
  handle.addEventListener('pointerup', finish); handle.addEventListener('pointercancel', finish);
  return handle;
};

const ARROW_MIN_PX = 26;
function shapeLength(points) {let d = 0; for (let i = 1; i < points.length; i++) d += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]); return d}
// A closed contour (type 'polygon') stores its vertices once; the closing edge is implied.
function shapeRing(shape) {return shape.type === 'polygon' && shape.points.length > 2 ? [...shape.points, shape.points[0]] : shape.points}
function shapeArea(points) {let a = 0; for (let i = 0; i < points.length; i++) {const p = points[i], q = points[(i + 1) % points.length]; a += p[0] * q[1] - q[0] * p[1]} return Math.abs(a) / 2}
function formatArea(m2) {return m2 >= 1e6 ? `${(m2 / 1e6).toFixed(2)} км²` : m2 >= 100 ? `${m2.toFixed(0)} м²` : `${m2.toFixed(1)} м²`}
function shapeSize(shape) {return shape.type === 'polygon' ? formatArea(shapeArea(shape.points)) : formatLength(shapeLength(shape.points))}
function shapeKind(shape) {return shape.type === 'segment' ? 'отрезок' : shape.type === 'polygon' ? 'контур' : 'ломаная'}
function formatLength(m) {return m >= 1000 ? `${(m / 1000).toFixed(2)} км` : `${m.toFixed(0)} м`}
function downloadBlob(blob, name) {const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = name; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 5000)}

// Adds the toolbar, drawing, context menu and profile chart to one projection viewer.
// view: {card, tools, wrap, canvas, context, variant, imagePoint(e,n), worldAt(x,y), redraw(), fit()}
DV.attachTools = view => {
  const {wrap, context} = view; let variant = view.variant, entry = DV.annotationStore.entry(variant), shapes = entry.shapes;
  const state = {mode: 'pan', slot: firstFreeSlot(0), drawing: null, cursor: null, selected: -1, down: null,
    profiles: [], profileMode: 'overlay', profileWindow: null, profileYSpan: 255, hoverDistance: null, hoverProfile: null, hoverCurve: null, profileFocus: null, profilePin: null,
    syncTarget: null, syncStep: 1, syncChain: false, syncPreview: null};
  const bar = document.createElement('div'), list = document.createElement('div'), menu = document.createElement('div'), panel = document.createElement('div');
  bar.className = 'annotate-bar'; list.className = 'shape-list'; menu.className = 'shape-menu'; menu.hidden = true; panel.className = 'profile-panel'; panel.hidden = true;
  bar.innerHTML = `<div class="tool-group" role="group" aria-label="Инструмент">
      <button type="button" data-mode="pan" title="Перемещение и зум (Esc)">✋</button>
      <button type="button" data-mode="polyline" title="Ломаная: ЛКМ — точки, двойной щелчок или ПКМ — закончить, Backspace — убрать точку">✎ Ломаная</button>
      <button type="button" data-mode="polygon" title="Замкнутый контур (льдина, пятно пены): ЛКМ — вершины; двойной щелчок, ПКМ или Enter замыкают контур на первую точку. Контур можно проследить на соседнем кадре.">⬠ Контур</button>
      <button type="button" data-mode="segment" title="Отрезок: два щелчка ЛКМ; сразу показывает профиль яркости. Профили нескольких отрезков совмещаются по одной или двум точкам (кнопка «Синхр.»).">／ Отрезок</button></div>
    <div class="swatches" role="group" aria-label="Цвет объекта">${DV.COLORS.map(([c, name], i) => `<button type="button" data-slot="${i}" style="--c:${c}" title="${name}" aria-label="${name}"></button>`).join('')}</div>
    <label class="overlay-check"><input type="checkbox" data-overlay="land"> Суша</label>
    <label class="overlay-check"><input type="checkbox" data-overlay="coast"> Берег</label>
    <label class="overlay-check" title="Полупрозрачные векторы смещения прослеженных контуров и их положение на соседнем кадре"><input type="checkbox" data-overlay="vectors"> Векторы</label>
    <span class="annotate-status"></span>`;
  view.tools.after(bar); bar.after(list); wrap.append(menu); wrap.after(panel);
  wrap.tabIndex = 0;
  const status = bar.querySelector('.annotate-status');
  function say(text, stateName = 'idle') {status.textContent = text; status.classList.toggle('status-working', stateName === 'working'); status.classList.toggle('status-done', stateName === 'done'); status.classList.toggle('status-error', stateName === 'error')}
  function firstFreeSlot(from) {for (let k = 0; k < 10; k++) {const s = (from + k) % 10; if (!shapes.some(x => x.slot === s)) return s} return from % 10}
  function syncBar() {
    bar.querySelectorAll('[data-mode]').forEach(b => b.classList.toggle('active', b.dataset.mode === state.mode));
    bar.querySelectorAll('[data-slot]').forEach(b => {const s = Number(b.dataset.slot); b.classList.toggle('active', s === state.slot); b.classList.toggle('used', shapes.some(x => x.slot === s))});
    wrap.classList.toggle('drawing-mode', state.mode !== 'pan');
    wrap.classList.toggle('syncing-mode', !!state.syncTarget);
  }
  function syncOverlays() {bar.querySelectorAll('[data-overlay]').forEach(i => i.checked = !!DV.overlays[i.dataset.overlay]); view.redraw()}
  bar.querySelectorAll('[data-overlay]').forEach(input => input.onchange = () => DV.setOverlay(input.dataset.overlay, input.checked));
  bar.querySelectorAll('[data-mode]').forEach(b => b.onclick = () => {setMode(b.dataset.mode); wrap.focus()});
  bar.querySelectorAll('[data-slot]').forEach(b => b.onclick = () => {state.slot = Number(b.dataset.slot); if (state.drawing) state.drawing.slot = state.slot; syncBar(); view.redraw(); wrap.focus()});
  function setMode(mode) {
    state.mode = mode; state.drawing = null; hideMenu();
    say(mode === 'polyline' ? 'ЛКМ — точки ломаной, двойной щелчок или ПКМ — закончить. Перетаскивание по-прежнему двигает карту.' :
      mode === 'polygon' ? 'ЛКМ — вершины контура, двойной щелчок или ПКМ — замкнуть на первую точку.' :
      mode === 'segment' ? 'Щёлкните начало и конец отрезка.' : '');
    syncBar(); view.redraw();
  }
  function persist() {entry.saved = Date.now(); DV.annotationStore.save(); renderList(); syncBar()}
  function finish() {
    const d = state.drawing;
    if (!d) return;
    state.drawing = null;
    if (d.points.length < (d.type === 'polygon' ? 3 : 2)) {say(d.type === 'polygon' ? 'Для контура нужно минимум три точки.' : 'Нужно минимум две точки.'); view.redraw(); return}
    const replaced = shapes.findIndex(s => s.slot === d.slot);
    if (replaced >= 0) state.profiles = state.profiles.filter(profile => profile.shape !== shapes[replaced]);
    const shape = {slot: d.slot, type: d.type, points: d.points.map(p => p.map(v => Math.round(v * 100) / 100)), name: `obj${d.slot + 1}_${DV.COLORS[d.slot][1]}`, hidden: false};
    if (replaced >= 0) shapes.splice(replaced, 1, shape); else shapes.push(shape);
    const index = shapes.indexOf(shape);
    say(`${replaced >= 0 ? 'Разметка заменена' : 'Добавлена разметка'}: ${DV.COLORS[d.slot][1]}, ${shapeSize(shape)}.`);
    state.slot = firstFreeSlot(d.slot); persist(); updateProfilePanel(); select(index);
    if (d.type === 'segment') showProfile(index);
    view.redraw();
  }
  function select(index) {state.selected = index; renderList(); view.redraw()}
  function remove(index) {
    if (index < 0 || index >= shapes.length) return;
    const [shape] = shapes.splice(index, 1);
    if (state.syncTarget === shape) {stopSync(); syncBar()}
    state.profiles = state.profiles.filter(profile => profile.shape !== shape); updateProfilePanel();
    state.selected = -1; hideMenu(); persist(); view.redraw(); say(`Разметка «${DV.COLORS[shape.slot][1]}» удалена.`);
  }
  async function exportShape(index) {
    const shape = shapes[index];
    if (!shape) return;
    say('Сохраняю профиль…', 'working');
    try {
      const r = await api('/api/profile-export', {method: 'POST', body: JSON.stringify({path: variant.path, points: shapeRing(shape), name: shape.name, type: shape.type, color: DV.COLORS[shape.slot][0]})});
      downloadBlob(new Blob([r.text], {type: 'text/plain;charset=utf-8'}), r.filename);
      say(`TXT сохранён: ${r.count} точек · ${r.path}`, 'done');
    } catch (e) {say(e.message, 'error')}
  }
  async function exportAll() {for (let i = 0; i < shapes.length; i++) await exportShape(i)}
  function toggleShape(index) {const shape = shapes[index]; if (!shape) return; shape.hidden = !shape.hidden; persist(); view.redraw()}
  // Profiles of several segments share one distance axis.  One sync point per segment (`sync_distance_m`, metres
  // from its start) is the zero of that axis: the profiles are shifted.  A second point (`sync_distance2_m`) also
  // stretches the profile, so that both points of every segment coincide; a segment whose second point lies
  // before the first is mirrored.  The axis keeps the metres of the first open two-point profile.
  function syncCount(shape) {return Number.isFinite(shape.sync_distance_m) ? (Number.isFinite(shape.sync_distance2_m) ? 2 : 1) : 0}
  function stopSync() {state.syncTarget = null; state.syncPreview = null; state.syncStep = 1; state.syncChain = false; state.down = null}
  // step: which point is picked; chain: after the first point keep picking the (optional) second one.
  function startSync(index, step = 1, chain = true) {
    const shape = shapes[index]; if (!shape || shape.type !== 'segment') return;
    shape.hidden = false; state.syncTarget = shape; state.syncStep = step; state.syncChain = chain && step === 1; state.syncPreview = null; state.drawing = null; state.mode = 'pan'; select(index); syncBar();
    say(step === 1 ? `Щёлкните на ${DV.COLORS[shape.slot][1]} линии первую точку совмещения — она станет нулём общей шкалы.`
      : `Щёлкните на ${DV.COLORS[shape.slot][1]} линии вторую точку совмещения: профиль растянется так, чтобы обе точки совпали с другими отрезками.`); wrap.focus(); view.redraw();
  }
  function endSync(text) {const shape = state.syncTarget; stopSync(); persist(); updateProfilePanel(); syncBar(); view.redraw(); if (shape) say(text || syncSummary(shape))}
  function syncSummary(shape) {
    const name = DV.COLORS[shape.slot][1], count = syncCount(shape);
    return count === 2 ? `Совмещение «${name}» по двум точкам: ${formatSyncDistance(shape.sync_distance_m)} и ${formatSyncDistance(shape.sync_distance2_m)} от начала отрезка.`
      : count === 1 ? `Совмещение «${name}» по одной точке: ${formatSyncDistance(shape.sync_distance_m)} от начала отрезка.` : `У отрезка «${name}» нет точек совмещения.`;
  }
  function formatSyncDistance(m) {return m >= 1000 ? `${(m / 1000).toFixed(3)} км` : `${m.toFixed(m < 100 ? 1 : 0)} м`}
  function pickSync(shape, distance) {
    const value = Math.round(distance * 100) / 100, other = state.syncStep === 1 ? (state.syncChain ? NaN : shape.sync_distance2_m) : shape.sync_distance_m;
    if (Number.isFinite(other) && Math.abs(value - other) < Math.max(.05, shapeLength(shape.points) * .01)) {say('Две точки совмещения должны быть в разных местах отрезка — щёлкните дальше от первой.', 'error'); return}
    if (state.syncStep === 2) {shape.sync_distance2_m = value; endSync(); return}
    shape.sync_distance_m = value;
    if (!state.syncChain) {endSync(); return}
    delete shape.sync_distance2_m; state.syncStep = 2; state.syncPreview = null; state.down = null; persist(); updateProfilePanel(); syncBar(); view.redraw();
    say(`Точка 1 задана — профили уже совмещены по ней. Щёлкните вторую точку, чтобы совместить и по масштабу, или Esc / ПКМ — оставить одну.`);
  }
  function resetSync(index, secondOnly = false) {
    const shape = shapes[index]; if (!shape) return;
    delete shape.sync_distance2_m; if (!secondOnly) delete shape.sync_distance_m;
    if (state.syncTarget === shape) stopSync();
    persist(); updateProfilePanel(); syncBar(); view.redraw(); say(secondOnly ? 'Вторая точка убрана: совмещение по одной точке.' : 'Точки совмещения сброшены.');
  }
  function renderList() {
    list.innerHTML = '';
    shapes.slice().sort((a, b) => a.slot - b.slot).forEach(shape => {
      const index = shapes.indexOf(shape), chip = document.createElement('span');
      const profileOpen = state.profiles.some(profile => profile.shape === shape);
      chip.className = 'shape-chip' + (index === state.selected ? ' selected' : '') + (shape.hidden ? ' hidden-shape' : '') + (state.syncTarget === shape ? ' syncing' : ''); chip.style.setProperty('--c', DV.COLORS[shape.slot][0]);
      chip.title = `${DV.COLORS[shape.slot][1]} · ${shapeKind(shape)}${shape.track ? ` · ${shape.track.auto ? 'найден автоматически' : 'прослежен'}: ${shape.track.displacement_m.toFixed(2)} м, ${(shape.track.speed_m_s * 100).toFixed(1)} см/с` : ''}`;
      chip.innerHTML = `<i title="${DV.COLORS[shape.slot][1]}"></i><span>${shapeSize(shape)}${shape.track ? ` · ${shape.track.auto ? '⇥' : '→'} ${(shape.track.speed_m_s * 100).toFixed(1)} см/с` : ''}</span>`;
      const buttons = [[shape.hidden ? 'Показать' : 'Скрыть', shape.hidden ? 'Показать разметку' : 'Скрыть разметку', () => toggleShape(index)], ...(shape.type === 'segment' ? [[state.syncTarget === shape ? (state.syncStep === 1 ? 'Точка 1…' : 'Точка 2…') : syncCount(shape) === 2 ? 'Синхр. 2 т. ✓' : syncCount(shape) === 1 ? 'Синхр. 1 т. ✓' : 'Синхр.', 'Совмещение профилей: щёлкните на отрезке одну точку (сдвиг) и, если нужно, вторую (сдвиг и растяжение). Отдельные точки меняются через ПКМ по отрезку.', () => startSync(index)]] : []), [profileOpen ? 'Убрать график' : 'Профиль', profileOpen ? 'Закрыть этот профиль' : 'Добавить профиль к сравнению', () => toggleProfile(index)], ['TXT', 'Сохранить координаты и интенсивность в TXT', () => exportShape(index)], ['×', 'Удалить', () => remove(index)]];
      buttons.forEach(([text, title, handler]) => {const b = document.createElement('button'); b.type = 'button'; b.textContent = text; b.title = title; b.onclick = event => {event.stopPropagation(); handler()}; chip.append(b)});
      chip.onclick = () => select(index);
      list.append(chip);
    });
    if (shapes.length > 1) {const all = document.createElement('button'); all.type = 'button'; all.className = 'shape-all'; all.textContent = 'Сохранить все в TXT'; all.onclick = exportAll; list.append(all)}
  }
  // Hit test in screen pixels.
  function shapeAt(x, y) {
    let best = -1, distance = 7;
    shapes.forEach((shape, index) => {
      if (shape.hidden) return;
      const p = shapeRing(shape).map(([e, n]) => view.imagePoint(e, n));
      for (let i = 1; i < p.length; i++) {
        const ax = p[i - 1].x, ay = p[i - 1].y, dx = p[i].x - ax, dy = p[i].y - ay, len2 = dx * dx + dy * dy;
        const t = len2 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / len2)) : 0, d = Math.hypot(x - ax - t * dx, y - ay - t * dy);
        if (d < distance) {distance = d; best = index}
      }
    });
    return best;
  }
  function hideMenu() {menu.hidden = true}
  function showMenu(index, x, y) {
    const shape = shapes[index];
    menu.innerHTML = `<b style="--c:${DV.COLORS[shape.slot][0]}">${DV.COLORS[shape.slot][1]} · ${shapeSize(shape)}</b>`;
    [[shape.hidden ? 'Показать' : 'Скрыть', () => toggleShape(index)], ...(shape.type === 'segment' ? [[syncCount(shape) ? 'Совмещение: указать точки заново' : 'Совмещение: указать точки (1 или 2)', () => startSync(index)],
      ...(syncCount(shape) === 1 ? [['Добавить вторую точку', () => startSync(index, 2)]] : []),
      ...(syncCount(shape) === 2 ? [['Изменить точку 1', () => startSync(index, 1, false)], ['Изменить точку 2', () => startSync(index, 2)], ['Убрать вторую точку', () => resetSync(index, true)]] : []),
      ...(syncCount(shape) ? [['Сбросить совмещение', () => resetSync(index)]] : [])] : []), [state.profiles.some(profile => profile.shape === shape) ? 'Убрать профиль' : 'Добавить профиль', () => toggleProfile(index)], ['Сохранить в TXT', () => exportShape(index)], ['Удалить', () => remove(index)]].forEach(([text, handler]) => {
      const b = document.createElement('button'); b.type = 'button'; b.textContent = text; b.onclick = () => {hideMenu(); handler()}; menu.append(b);
    });
    menu.hidden = false;
    menu.style.left = `${Math.min(x, wrap.clientWidth - 200)}px`; menu.style.top = `${Math.max(0, Math.min(y, wrap.clientHeight - menu.offsetHeight - 4))}px`;
  }
  function screen(event) {const r = wrap.getBoundingClientRect(); return {x: event.clientX - r.left, y: event.clientY - r.top}}
  function world(p) {const w = view.worldAt(p.x, p.y); return [w.e, w.n]}
  function closestAlongShape(shape, point) {
    let best = {distancePx: Infinity, distanceM: 0}; let travelled = 0;
    for (let i = 1; i < shape.points.length; i++) {
      const a = view.imagePoint(...shape.points[i - 1]), b = view.imagePoint(...shape.points[i]), dx = b.x - a.x, dy = b.y - a.y, screenLength2 = dx * dx + dy * dy;
      const t = screenLength2 ? Math.max(0, Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / screenLength2)) : 0;
      const distancePx = Math.hypot(point.x - a.x - t * dx, point.y - a.y - t * dy), segmentM = Math.hypot(shape.points[i][0] - shape.points[i - 1][0], shape.points[i][1] - shape.points[i - 1][1]);
      if (distancePx < best.distancePx) best = {distancePx, distanceM: travelled + t * segmentM};
      travelled += segmentM;
    }
    return best;
  }

  // Pointer handling: returns true when the event was consumed (no panning).
  const handlers = {
    down(event) {
      if (menu.contains(event.target)) return true;  // clicks inside the context menu are not map clicks
      hideMenu();
      if (event.button !== 0) return event.button === 2;
      if (state.syncTarget) {state.down = {...screen(event), moved: false}; state.syncPreview = closestAlongShape(state.syncTarget, state.down); view.redraw(); return true}
      state.down = {...screen(event), moved: false};
      return false;  // a drag still pans in every mode
    },
    move(event) {
      const p = screen(event);
      if (state.syncTarget) {state.syncPreview = closestAlongShape(state.syncTarget, p); view.redraw(); return}
      if (state.down && Math.hypot(p.x - state.down.x, p.y - state.down.y) > 4) state.down.moved = true;
      if (state.drawing) {state.cursor = world(p); view.redraw()}
    },
    up(event) {
      if (menu.contains(event.target)) return;
      if (state.syncTarget && event.button === 0) {pickSync(state.syncTarget, closestAlongShape(state.syncTarget, screen(event)).distanceM); return}
      const down = state.down;
      state.down = null;
      if (!down || down.moved || event.button !== 0) return;
      const p = screen(event);
      if (state.mode === 'pan') {const hit = shapeAt(p.x, p.y); if (hit >= 0 || state.selected >= 0) select(hit); return}
      const point = world(p);
      if (!state.drawing) state.drawing = {type: state.mode, slot: state.slot, points: []};
      const last = state.drawing.points[state.drawing.points.length - 1];
      if (last) {const q = view.imagePoint(...last); if (Math.hypot(q.x - p.x, q.y - p.y) < 3) return}  // second click of a double-click
      state.drawing.points.push(point);
      if (state.mode === 'segment' && state.drawing.points.length === 2) finish();
      else if (state.drawing.points.length >= 500) finish();
      view.redraw();
    },
    dblclick(event) {
      if (menu.contains(event.target)) return;
      if ((state.mode === 'polyline' || state.mode === 'polygon') && state.drawing) {event.preventDefault(); finish(); return}
      if (state.mode === 'pan') {const p = screen(event), hit = shapeAt(p.x, p.y); if (hit >= 0) showProfile(hit)}
    },
    contextmenu(event) {
      event.preventDefault();
      if (menu.contains(event.target)) return;
      if (state.drawing && (state.mode === 'polyline' || state.mode === 'polygon')) {finish(); return}
      if (state.syncTarget) {endSync(); return}
      const p = screen(event), hit = shapeAt(p.x, p.y);
      if (hit >= 0) {select(hit); showMenu(hit, p.x, p.y)}
    },
    key(event) {
      if (event.key === 'Escape') {if (state.syncTarget) endSync(state.syncStep === 1 ? 'Выбор точки совмещения отменён.' : ''); else if (state.drawing) {state.drawing = null; view.redraw()} else setMode('pan'); hideMenu()}
      else if (event.key === 'Enter' && state.syncTarget) endSync();
      else if (event.key === 'Enter' && state.drawing) finish();
      else if (event.key === 'Backspace' && state.drawing) {state.drawing.points.pop(); if (!state.drawing.points.length) state.drawing = null; view.redraw()}
      else if (event.key === 'Delete' && state.selected >= 0) remove(state.selected);
      else return;
      event.preventDefault();
    },
    leave() {state.cursor = null; if (state.syncTarget) state.syncPreview = null; if (state.drawing || state.syncTarget) view.redraw()},
  };
  wrap.addEventListener('keydown', handlers.key);

  function strokePath(points, colour, width, dash = []) {
    if (points.length < 2) return;
    context.beginPath(); points.forEach(([e, n], i) => {const p = view.imagePoint(e, n); i ? context.lineTo(p.x, p.y) : context.moveTo(p.x, p.y)});
    context.setLineDash(dash); context.strokeStyle = 'rgba(0,0,0,.85)'; context.lineWidth = width + 1.5; context.stroke();
    context.strokeStyle = colour; context.lineWidth = width; context.stroke(); context.setLineDash([]);
  }
  function drawVertices(points, colour, size) {points.forEach(([e, n]) => {const p = view.imagePoint(e, n); context.fillStyle = 'rgba(0,0,0,.85)'; context.fillRect(p.x - size - 1, p.y - size - 1, 2 * size + 2, 2 * size + 2); context.fillStyle = colour; context.fillRect(p.x - size, p.y - size, 2 * size, 2 * size)})}
  function pointAtDistance(points, distance) {
    for (let i = 1; i < points.length; i++) {
      const len = Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
      if (distance <= len || i === points.length - 1) {const t = len ? Math.min(1, distance / len) : 0; return [points[i - 1][0] + (points[i][0] - points[i - 1][0]) * t, points[i - 1][1] + (points[i][1] - points[i - 1][1]) * t]}
      distance -= len;
    }
    return points[0];
  }
  function drawSyncMarker(shape, distance, preview = false, number = 0) {
    const point = pointAtDistance(shape.points, distance), p = view.imagePoint(...point), colour = DV.COLORS[shape.slot][0];
    context.save(); context.translate(p.x, p.y); context.beginPath(); context.arc(0, 0, preview ? 9 : 7, 0, Math.PI * 2); context.fillStyle = preview ? 'rgba(255,255,255,.75)' : '#fff'; context.fill(); context.lineWidth = 3; context.strokeStyle = '#000'; context.stroke(); context.beginPath(); context.arc(0, 0, 3.5, 0, Math.PI * 2); context.fillStyle = colour; context.fill(); context.strokeStyle = colour; context.lineWidth = 2; context.beginPath(); context.moveTo(-12, 0); context.lineTo(-7, 0); context.moveTo(7, 0); context.lineTo(12, 0); context.moveTo(0, -12); context.lineTo(0, -7); context.moveTo(0, 7); context.lineTo(0, 12); context.stroke();
    if (number) {context.font = '700 12px system-ui'; context.textAlign = 'left'; context.textBaseline = 'alphabetic'; context.lineWidth = 3; context.strokeStyle = 'rgba(0,0,0,.85)'; context.strokeText(number, 11, -9); context.fillStyle = '#fff'; context.fillText(number, 11, -9)}
    context.restore();
  }
  function fillRing(points, colour, alpha) {
    if (points.length < 3) return;
    context.beginPath(); points.forEach(([e, n], i) => {const p = view.imagePoint(e, n); i ? context.lineTo(p.x, p.y) : context.moveTo(p.x, p.y)}); context.closePath();
    context.save(); context.globalAlpha = alpha; context.fillStyle = colour; context.fill(); context.restore();
  }
  function drawArrow(from, to, colour) {
    const a = view.imagePoint(...from), b = view.imagePoint(...to), length = Math.hypot(b.x - a.x, b.y - a.y);
    context.save(); context.globalAlpha = .6; context.lineCap = 'round'; context.lineJoin = 'round';
    context.beginPath(); context.arc(a.x, a.y, 4, 0, Math.PI * 2); context.fillStyle = '#000'; context.fill();
    context.beginPath(); context.arc(a.x, a.y, 2.6, 0, Math.PI * 2); context.fillStyle = colour; context.fill();
    // A displacement shorter than ARROW_MIN_PX on screen is drawn as an enlarged dashed arrow of that length
    // (direction kept), otherwise it would be an invisible dot; the caller marks this in the label.
    const enlarged = length < ARROW_MIN_PX && length > 1e-6, shown = enlarged ? ARROW_MIN_PX : length;
    if (length > 1e-6) {
      const ux = (b.x - a.x) / length, uy = (b.y - a.y) / length, head = Math.min(14, Math.max(6, shown * .3)), tip = {x: a.x + ux * shown, y: a.y + uy * shown};
      const path = () => {context.beginPath(); context.moveTo(a.x, a.y); context.lineTo(tip.x, tip.y);
        context.moveTo(tip.x - head * (ux * .87 - uy * .5), tip.y - head * (uy * .87 + ux * .5)); context.lineTo(tip.x, tip.y);
        context.lineTo(tip.x - head * (ux * .87 + uy * .5), tip.y - head * (uy * .87 - ux * .5))};
      context.setLineDash(enlarged ? [5, 4] : []);
      path(); context.strokeStyle = '#000'; context.lineWidth = 5; context.stroke();
      path(); context.strokeStyle = colour; context.lineWidth = 2.6; context.stroke(); context.setLineDash([]);
      context.restore(); return {tip, enlarged};
    }
    context.restore(); return {tip: b, enlarged: false};
  }
  // Tracked contour: its position on the neighbouring frame (dashed) and the displacement vector, both translucent.
  function drawTrack(shape, colour) {
    const t = shape.track;
    if (!t || !DV.overlays.vectors || !Array.isArray(t.ghost) || t.ghost.length < 2) return;
    context.save(); context.globalAlpha = .5; strokePath(t.closed === false ? t.ghost : [...t.ghost, t.ghost[0]], colour, 1.2, [5, 4]); context.restore();
    const arrow = drawArrow(t.start, t.end, colour), p = arrow.tip;
    const lines = [`${t.displacement_m < 10 ? t.displacement_m.toFixed(2) : t.displacement_m.toFixed(1)} м · ${(t.speed_m_s * 100).toFixed(1)} см/с · ${Math.round(t.azimuth_deg)}°${arrow.enlarged ? ' · стрелка увеличена' : ''}`];
    if (Number.isFinite(t.rotation_deg) && !(t.manual && t.rotation_found === false)) lines.push(`поворот ${t.rotation_deg >= 0 ? '+' : ''}${t.rotation_deg.toFixed(1)}°`);
    context.save(); context.globalAlpha = .85; context.font = '600 11px system-ui'; context.lineWidth = 3; context.strokeStyle = 'rgba(0,0,0,.8)';
    lines.forEach((text, i) => {context.strokeText(text, p.x + 8, p.y - 6 + 13 * i); context.fillStyle = '#fff'; context.fillText(text, p.x + 8, p.y - 6 + 13 * i)}); context.restore();
  }
  function draw() {
    context.save(); context.lineJoin = 'round'; context.lineCap = 'round'; context.font = '700 12px system-ui';
    shapes.forEach((shape, index) => {
      if (shape.hidden) return;
      const colour = DV.COLORS[shape.slot][0], selected = index === state.selected;
      if (shape.type === 'polygon') fillRing(shape.points, colour, selected ? .18 : .08);
      strokePath(shapeRing(shape), colour, selected ? 1.8 : 1.1);
      drawVertices(shape.points, colour, selected ? 3 : 1.5);
      drawTrack(shape, colour);
      if (syncCount(shape)) drawSyncMarker(shape, shape.sync_distance_m, false, syncCount(shape) === 2 ? 1 : 0);
      if (syncCount(shape) === 2) drawSyncMarker(shape, shape.sync_distance2_m, false, 2);
    });
    const d = state.drawing;
    if (d) {
      const colour = DV.COLORS[d.slot][0];
      strokePath(d.points, colour, 1.3);
      context.save(); context.globalAlpha = .55;
      if (state.cursor && d.points.length) strokePath([d.points[d.points.length - 1], state.cursor], colour, .8);
      if (d.type === 'polygon' && state.cursor && d.points.length > 1) strokePath([state.cursor, d.points[0]], colour, .8);
      context.restore();
      drawVertices(d.points, colour, 3);
    }
    if (state.syncTarget && state.syncPreview) drawSyncMarker(state.syncTarget, state.syncPreview.distanceM, true, state.syncStep);
    if (state.hoverDistance !== null) state.profiles.filter(profile => !state.hoverProfile || profile === state.hoverProfile).forEach(profile => {
      const absolute = fromAxis(syncTransform(profile), state.hoverDistance); if (absolute < 0 || absolute > profile.data.length_m) return;
      const [e, n] = pointAtDistance(profile.shape.points, absolute), p = view.imagePoint(e, n), colour = DV.COLORS[profile.shape.slot][0];
      context.beginPath(); context.arc(p.x, p.y, 7, 0, Math.PI * 2); context.lineWidth = 3; context.strokeStyle = '#000'; context.stroke(); context.lineWidth = 2; context.strokeStyle = colour; context.stroke();
    });
    context.restore();
  }

  // --- One or several intensity profiles vs distance ---
  // Every curve is a solid line.  What tells curves apart is the layout («Вместе» / one row per profile, coloured
  // by channel / one row per channel, coloured by profile), the switch matrix above the chart, whose cells show the
  // curve's own colour, and highlighting: pointing at a cell, a profile or a channel dims everything else.
  panel.innerHTML = `<div class="profile-head"><b data-profile-title>Профили</b><div class="profile-modes" role="group" aria-label="Расположение профилей"><button type="button" data-profile-mode="overlay" title="Все кривые на одном графике">Вместе</button><button type="button" data-profile-mode="stacked" title="Свой график у каждого отрезка; цвет кривой — канал">По профилям</button><button type="button" data-profile-mode="channels" title="Свой график у каждого канала; цвет кривой — отрезок">По каналам</button></div><button type="button" data-profile-fit title="Вернуть всю ось расстояний">Вся длина</button><div class="profile-y-scale" role="group" aria-label="Вертикальный масштаб"><button type="button" data-profile-y-in title="Увеличить вертикальный масштаб">+</button><button type="button" data-profile-y-out title="Уменьшить вертикальный масштаб">−</button></div><span class="profile-readout"></span><button type="button" data-close title="Закрыть все профили">×</button></div><div class="profile-body"><div class="profile-channel-controls"></div><div class="profile-chart"><canvas height="190"></canvas></div></div>`;
  DV.attachTopResize(panel, 38);
  const chart = panel.querySelector('canvas'), readout = panel.querySelector('.profile-readout'), profileTitle = panel.querySelector('[data-profile-title]'), channelControls = panel.querySelector('.profile-channel-controls');
  const idleReadout = 'колесо — горизонтальный масштаб';
  panel.querySelector('[data-close]').onclick = closeProfiles;
  panel.querySelector('[data-profile-fit]').onclick = () => {state.profileWindow = null; state.hoverDistance = null; drawChart()};
  panel.querySelector('[data-profile-y-in]').onclick = () => changeVerticalScale(true);
  panel.querySelector('[data-profile-y-out]').onclick = () => changeVerticalScale(false);
  panel.querySelectorAll('[data-profile-mode]').forEach(button => button.onclick = () => {state.profileMode = button.dataset.profileMode; clearHover(); updateProfilePanel()});
  function clearHover() {state.hoverDistance = null; state.hoverProfile = null; state.hoverCurve = null}
  function changeVerticalScale(zoomIn) {
    const minimumSpan = minimumVisibleVerticalSpan();
    state.profileYSpan = Math.max(minimumSpan, Math.min(255, state.profileYSpan * (zoomIn ? .75 : 1 / .75)));
    if (state.profileYSpan > 254.5) state.profileYSpan = 255;
    readout.textContent = `вертикальный диапазон: ${Math.round(state.profileYSpan)}`;
    updateVerticalButtons(); drawChart();
  }
  function updateVerticalButtons() {
    panel.querySelector('[data-profile-y-in]').disabled = state.profileYSpan <= minimumVisibleVerticalSpan() + .01;
    panel.querySelector('[data-profile-y-out]').disabled = state.profileYSpan >= 254.99;
  }
  function closeProfiles() {state.profiles = []; state.profileWindow = null; state.profileYSpan = 255; state.profileFocus = null; state.profilePin = null; clearHover(); updateProfilePanel(); renderList(); view.redraw()}

  const channels = [['gray', 'Общий'], ['r', 'R'], ['g', 'G'], ['b', 'B']], channelKeys = channels.map(([key]) => key);
  const pad = {l: 40, r: 12, t: 18, b: 27};
  function profileChannels(profile) {if (Array.isArray(profile.channels)) return channelKeys.filter(key => profile.channels.includes(key)); return [channelKeys.includes(profile.channel) ? profile.channel : 'gray']}
  function channelLabel(key) {return channels.find(([candidate]) => candidate === key)?.[1] || key}
  function channelColour(profile, key) {return key === 'r' ? '#d23c3c' : key === 'g' ? '#2e9b4f' : key === 'b' ? '#3563d0' : DV.COLORS[profile.shape.slot][0]}
  // Colour of one curve: by channel, except in the per-channel layout, where the rows are channels and the colour is the segment.
  function curveColour(profile, key) {return state.profileMode === 'channels' ? DV.COLORS[profile.shape.slot][0] : channelColour(profile, key)}
  function setChannels(profile, keys) {profile.channels = channelKeys.filter(key => keys.includes(key)); delete profile.channel}
  function channelsChanged() {clearHover(); readout.textContent = idleReadout; refreshChannelControls(); drawChart()}

  // Switch matrix: rows = profiles, columns = channels.  The first row and the first column switch a whole
  // column / row / everything (a dash in the box = partly on).
  function updateChannelControls() {
    if (!state.profiles.length) {channelControls.innerHTML = ''; return}
    const reference = referenceSpan();
    channelControls.innerHTML = `<div class="profile-matrix" style="grid-template-columns: auto repeat(${channels.length}, auto) auto">
      <label class="pm-head" title="Включить или выключить все каналы всех профилей"><input type="checkbox" data-pm-all><span>Все</span></label>
      ${channels.map(([key, label]) => `<label class="pm-head" data-pm-focus-key="${key}" title="Канал «${label}» у всех профилей; наведите — выделить на графике"><input type="checkbox" data-pm-column="${key}"><span>${label}</span></label>`).join('')}
      <span></span>
      ${state.profiles.map((profile, index) => {
        const name = DV.COLORS[profile.shape.slot][1], transform = syncTransform(profile, reference), count = syncCount(profile.shape);
        const note = count === 2 ? `2 т. ×${Math.abs(transform.scale).toFixed(2)}${transform.scale < 0 ? ' ⇄' : ''}` : count === 1 ? '1 т.' : '';
        const noteTitle = `Длина ${formatLength(profile.data.length_m)} · ${count === 2 ? `совмещён по двум точкам, растяжение ×${Math.abs(transform.scale).toFixed(3)}${transform.scale < 0 ? ', зеркально' : ''}` : count === 1 ? 'совмещён по одной точке (сдвиг)' : 'без совмещения: ноль шкалы — начало отрезка'}`;
        return `<span class="pm-row" data-pm-focus-profile="${index}" style="--c:${DV.COLORS[profile.shape.slot][0]}"><input type="checkbox" data-pm-row="${index}" title="Все каналы профиля «${name}»"><i></i><button type="button" data-pm-pin="${index}" title="Закрепить выделение этого профиля на графике">${name}</button></span>
        ${channels.map(([key, label]) => `<label class="pm-cell" data-pm-focus-profile="${index}" data-pm-focus-key="${key}" style="--channel:${curveColour(profile, key)}" title="${name} · ${label}"><input type="checkbox" data-pm-cell="${index}" value="${key}"><span class="pm-line"></span></label>`).join('')}
        <span class="pm-note" title="${noteTitle}">${note}<button type="button" data-pm-remove="${index}" title="Убрать профиль «${name}» с графика">×</button></span>`;
      }).join('')}</div><small class="pm-hint">Наведите на строку, столбец или клетку — кривые выделятся. Щелчок по названию закрепляет выделение.</small>`;
    const all = () => state.profiles.every(profile => profileChannels(profile).length === channels.length);
    channelControls.querySelector('[data-pm-all]').onchange = () => {const keys = all() ? [] : channelKeys; state.profiles.forEach(profile => setChannels(profile, keys)); channelsChanged()};
    channelControls.querySelectorAll('[data-pm-column]').forEach(input => input.onchange = () => {
      const key = input.dataset.pmColumn, on = !state.profiles.every(profile => profileChannels(profile).includes(key));
      state.profiles.forEach(profile => setChannels(profile, on ? [...profileChannels(profile), key] : profileChannels(profile).filter(k => k !== key))); channelsChanged();
    });
    channelControls.querySelectorAll('[data-pm-row]').forEach(input => input.onchange = () => {
      const profile = state.profiles[Number(input.dataset.pmRow)]; if (!profile) return;
      setChannels(profile, profileChannels(profile).length === channels.length ? [] : channelKeys); channelsChanged();
    });
    channelControls.querySelectorAll('[data-pm-cell]').forEach(input => input.onchange = () => {
      const profile = state.profiles[Number(input.dataset.pmCell)]; if (!profile) return;
      setChannels(profile, input.checked ? [...profileChannels(profile), input.value] : profileChannels(profile).filter(k => k !== input.value)); channelsChanged();
    });
    channelControls.querySelectorAll('[data-pm-pin]').forEach(button => button.onclick = () => {
      const profile = state.profiles[Number(button.dataset.pmPin)]; state.profilePin = state.profilePin && state.profilePin.profile === profile ? null : {profile};
      refreshChannelControls(); drawChart();
    });
    channelControls.querySelectorAll('[data-pm-remove]').forEach(button => button.onclick = () => toggleProfile(shapes.indexOf(state.profiles[Number(button.dataset.pmRemove)].shape)));
    channelControls.querySelectorAll('[data-pm-focus-profile], [data-pm-focus-key]').forEach(element => {
      element.onmouseenter = () => {state.profileFocus = {profile: state.profiles[Number(element.dataset.pmFocusProfile)] || null, key: element.dataset.pmFocusKey || null}; drawChart()};
      element.onmouseleave = () => {state.profileFocus = null; drawChart()};
    });
    refreshChannelControls();
  }
  function refreshChannelControls() {
    const box = (input, on, total) => {input.checked = on > 0; input.indeterminate = on > 0 && on < total};
    const count = key => state.profiles.filter(profile => profileChannels(profile).includes(key)).length;
    channelControls.querySelectorAll('[data-pm-all]').forEach(input => box(input, state.profiles.reduce((sum, profile) => sum + profileChannels(profile).length, 0), state.profiles.length * channels.length));
    channelControls.querySelectorAll('[data-pm-column]').forEach(input => box(input, count(input.dataset.pmColumn), state.profiles.length));
    channelControls.querySelectorAll('[data-pm-row]').forEach(input => {const profile = state.profiles[Number(input.dataset.pmRow)]; if (profile) box(input, profileChannels(profile).length, channels.length)});
    channelControls.querySelectorAll('[data-pm-cell]').forEach(input => {const profile = state.profiles[Number(input.dataset.pmCell)]; if (profile) {input.checked = profileChannels(profile).includes(input.value); input.parentElement.classList.toggle('off', !input.checked)}});
    channelControls.querySelectorAll('[data-pm-pin]').forEach(button => button.classList.toggle('pinned', !!state.profilePin && state.profilePin.profile === state.profiles[Number(button.dataset.pmPin)]));
    markNearestCell();
  }
  function markNearestCell() {
    channelControls.querySelectorAll('.pm-cell').forEach(cell => cell.classList.toggle('near', !!state.hoverCurve && state.profiles[Number(cell.dataset.pmFocusProfile)] === state.hoverCurve.profile && cell.dataset.pmFocusKey === state.hoverCurve.key));
  }
  function updateProfilePanel() {
    panel.hidden = !state.profiles.length;
    if (state.profilePin && !state.profiles.includes(state.profilePin.profile)) state.profilePin = null;
    state.profileFocus = null;
    profileTitle.innerHTML = state.profiles.length ? `${state.profiles.map(profile => `<i style="--c:${DV.COLORS[profile.shape.slot][0]}" title="${DV.COLORS[profile.shape.slot][1]}"></i>`).join('')} ${state.profiles.length} профил${state.profiles.length === 1 ? 'ь' : state.profiles.length < 5 ? 'я' : 'ей'}` : 'Профили';
    updateChannelControls();
    panel.querySelectorAll('[data-profile-mode]').forEach(button => button.classList.toggle('active', button.dataset.profileMode === state.profileMode));
    updateVerticalButtons();
    if (state.profiles.length) {readout.textContent = idleReadout; drawChart()}
  }
  function toggleProfile(index) {const shape = shapes[index], open = state.profiles.findIndex(profile => profile.shape === shape); state.profileWindow = null; if (open >= 0) {state.profiles.splice(open, 1); clearHover(); updateProfilePanel(); renderList(); view.redraw()} else showProfile(index)}
  async function showProfile(index) {
    const shape = shapes[index];
    if (!shape || state.profiles.some(profile => profile.shape === shape)) return;
    select(index); say('Считаю профиль…', 'working');
    try {
      const data = await api('/api/profile', {method: 'POST', body: JSON.stringify({path: variant.path, points: shapeRing(shape)})});
      // A new profile starts with the channels of the first open one, so that the curves are comparable at once.
      const inherited = state.profiles.length ? profileChannels(state.profiles[0]) : [];
      state.profiles.push({shape, data, channels: inherited.length ? inherited : ['gray']}); clearHover();
      updateProfilePanel(); renderList(); say('Профиль готов.', 'done');
    } catch (e) {say(e.message, 'error')}
  }

  function niceStep(target) {const p = 10 ** Math.floor(Math.log10(Math.max(target, 1e-6))), f = target / p; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p}
  // Common axis (see startSync): x = (distance - origin) * scale.  `referenceSpan` is the distance between the two
  // sync points of the first open two-point profile; the second point of every two-point profile lands on it.
  function clampedSync(profile, value) {return Math.max(0, Math.min(profile.data.length_m, value))}
  function referenceSpan() {const base = state.profiles.find(profile => syncCount(profile.shape) === 2); return base ? Math.abs(clampedSync(base, base.shape.sync_distance2_m) - clampedSync(base, base.shape.sync_distance_m)) : null}
  function syncTransform(profile, reference = referenceSpan()) {
    const count = syncCount(profile.shape), origin = count ? clampedSync(profile, profile.shape.sync_distance_m) : 0;
    const own = count === 2 ? clampedSync(profile, profile.shape.sync_distance2_m) - origin : 0;
    return {origin, scale: reference && Math.abs(own) > 1e-6 ? reference / own : 1};
  }
  function toAxis(transform, distance) {return (distance - transform.origin) * transform.scale}
  function fromAxis(transform, x) {return transform.origin + x / transform.scale}
  function profileTransforms() {const reference = referenceSpan(); return new Map(state.profiles.map(profile => [profile, syncTransform(profile, reference)]))}
  function fullProfileDomain() {
    const transforms = profileTransforms(), ends = state.profiles.flatMap(profile => [toAxis(transforms.get(profile), 0), toAxis(transforms.get(profile), profile.data.length_m)]);
    const minimum = Math.min(...ends, 0), maximum = Math.max(...ends, 0);
    return {minimum, maximum: maximum > minimum ? maximum : minimum + 1};
  }
  function profileDomain() {return state.profileWindow || fullProfileDomain()}
  function visibleValueRange() {
    const domain = profileDomain(), transforms = profileTransforms();
    let minimum = Infinity, maximum = -Infinity;
    state.profiles.forEach(profile => {
      const transform = transforms.get(profile), distances = profile.data.distance_m, keys = profileChannels(profile);
      distances.forEach((distance, index) => {
        const relative = toAxis(transform, distance);
        if (relative < domain.minimum || relative > domain.maximum) return;
        keys.forEach(key => {const value = profile.data[key]?.[index]; if (Number.isFinite(value)) {minimum = Math.min(minimum, value); maximum = Math.max(maximum, value)}});
      });
    });
    return Number.isFinite(minimum) ? {minimum, maximum} : {minimum: 0, maximum: 255};
  }
  function minimumVisibleVerticalSpan() {const range = visibleValueRange(); return Math.min(255, Math.max(8, (range.maximum - range.minimum) * 1.08))}
  function verticalDomain() {
    const values = visibleValueRange(), minimumSpan = minimumVisibleVerticalSpan();
    state.profileYSpan = Math.max(minimumSpan, Math.min(255, state.profileYSpan));
    const span = state.profileYSpan, centre = (values.minimum + values.maximum) / 2;
    let minimum = centre - span / 2, maximum = centre + span / 2;
    if (minimum < 0) {maximum -= minimum; minimum = 0}
    if (maximum > 255) {minimum -= maximum - 255; maximum = 255}
    return {minimum: Math.max(0, minimum), maximum: Math.min(255, maximum), span};
  }
  // Value of one curve at a position of the common axis (nearest sample), null outside the segment.
  function valueAt(profile, key, transform, x) {
    const absolute = fromAxis(transform, x), distances = profile.data.distance_m;
    if (!distances.length || absolute < 0 || absolute > profile.data.length_m) return null;
    let low = 0, high = distances.length - 1;
    while (high - low > 1) {const middle = (low + high) >> 1; distances[middle] <= absolute ? low = middle : high = middle}
    const value = profile.data[key]?.[Math.abs(distances[high] - absolute) < Math.abs(distances[low] - absolute) ? high : low];
    return Number.isFinite(value) ? value : null;
  }
  // Rows of the chart for the current layout: {label: [[text, colour], ...], profiles, curves: [{profile, key, colour}]}.
  function chartRows() {
    const curve = (profile, key) => ({profile, key, colour: curveColour(profile, key)}), name = profile => DV.COLORS[profile.shape.slot][1], ink = '#17201e';
    const sync = profile => syncCount(profile.shape) ? ` · совм. ${syncCount(profile.shape)} т.` : '';
    if (state.profileMode === 'stacked') return state.profiles.map(profile => ({profiles: [profile], curves: profileChannels(profile).map(key => curve(profile, key)),
      label: [[`${name(profile)} · ${formatLength(profile.data.length_m)}${sync(profile)}`, DV.COLORS[profile.shape.slot][0]], ...(profileChannels(profile).length ? profileChannels(profile).map(key => [channelLabel(key), channelColour(profile, key)]) : [['каналы выключены', '#6d7773']])]}));
    if (state.profileMode === 'channels') {
      const rows = channelKeys.filter(key => state.profiles.some(profile => profileChannels(profile).includes(key))).map(key => {
        const members = state.profiles.filter(profile => profileChannels(profile).includes(key));
        return {profiles: members, curves: members.map(profile => curve(profile, key)), label: [[key === 'gray' ? 'Общий (серый)' : `Канал ${channelLabel(key)}`, ink], ...members.map(profile => [name(profile), DV.COLORS[profile.shape.slot][0]])]};
      });
      if (rows.length) return rows;
    }
    const curves = state.profiles.flatMap(profile => profileChannels(profile).map(key => curve(profile, key)));
    return [{profiles: state.profiles, curves, label: curves.length ? curves.map(item => [`${name(item.profile)} · ${channelLabel(item.key)}`, item.colour]) : [['все каналы выключены — включите их в таблице выше', '#6d7773']]}];
  }
  function chartHeight(rowCount) {
    // The switch matrix stands to the left of the chart; on a narrow panel it wraps above it and takes height.
    const above = channelControls.offsetTop + channelControls.offsetHeight <= chart.parentElement.offsetTop + 1 ? channelControls.offsetHeight : 0;
    const chromeHeight = panel.querySelector('.profile-head').offsetHeight + above, available = Math.max(140, panel.clientHeight - chromeHeight - 8);
    return rowCount > 1 ? Math.max(145 * rowCount, available) : available;
  }
  function curveFocus() {return state.profileFocus || state.profilePin}
  function curveInFocus(item, focus) {return (!focus.profile || focus.profile === item.profile) && (!focus.key || focus.key === item.key)}
  function formatAxisDistance(distance, step) {if (Math.abs(distance) < step * .01) return '0 м'; const sign = distance > 0 ? '+' : '−', value = Math.abs(distance); return value >= 1000 ? `${sign}${(value / 1000).toFixed(step >= 1000 ? 0 : 1)} км` : `${sign}${Math.round(value)} м`}
  function breakDistances(shape) {let total = 0; return shape.points.slice(1, -1).map((point, i) => {total += Math.hypot(point[0] - shape.points[i][0], point[1] - shape.points[i][1]); return total})}
  function drawAxes(g, x, y, top, bottom, minimum, maximum, verticalMinimum, verticalMaximum, showLabels = true) {
    g.font = '10px system-ui'; g.textAlign = 'right'; g.textBaseline = 'middle';
    const verticalStep = niceStep((verticalMaximum - verticalMinimum) / 4);
    for (let v = Math.ceil(verticalMinimum / verticalStep) * verticalStep; v <= verticalMaximum + 1e-6; v += verticalStep) {g.strokeStyle = '#e6e9e5'; g.beginPath(); g.moveTo(pad.l, y(v)); g.lineTo(chart.clientWidth - pad.r, y(v)); g.stroke(); g.fillStyle = '#6d7773'; g.fillText(Math.round(v), pad.l - 6, y(v))}
    const step = niceStep((maximum - minimum) / 6); g.textAlign = 'center'; g.textBaseline = 'top';
    for (let d = Math.ceil(minimum / step) * step; d <= maximum + 1e-6; d += step) {const px = x(d), zero = Math.abs(d) < step * .01; g.strokeStyle = zero ? '#17201e' : '#eef0ec'; g.lineWidth = zero ? 1.6 : 1; g.beginPath(); g.moveTo(px, top); g.lineTo(px, bottom); g.stroke(); if (showLabels) {g.fillStyle = zero ? '#17201e' : '#6d7773'; g.fillText(formatAxisDistance(d, step), px, bottom + 7)}}
  }
  // Second sync point of the two-point profiles: the same bold line as the zero of the axis.
  function drawReference(g, x, top, bottom, minimum, maximum) {
    const reference = referenceSpan(); if (!reference || reference < minimum || reference > maximum) return;
    g.strokeStyle = '#17201e'; g.lineWidth = 1.6; g.beginPath(); g.moveTo(x(reference), top); g.lineTo(x(reference), bottom); g.stroke();
    g.font = '9px system-ui'; g.textAlign = 'center'; g.textBaseline = 'top'; g.fillStyle = '#17201e'; g.fillText('точка 2', x(reference), top - 11); if (0 >= minimum && 0 <= maximum) g.fillText('точка 1', x(0), top - 11);
  }
  function drawBreaks(g, profile, transform, x, top, bottom) {
    const colour = DV.COLORS[profile.shape.slot][0]; g.font = '9px system-ui'; g.textAlign = 'center'; g.textBaseline = 'top'; g.lineWidth = 1;
    breakDistances(profile.shape).forEach(distance => {const relative = toAxis(transform, distance), px = x(relative); g.strokeStyle = colour; g.globalAlpha = .45; g.beginPath(); g.moveTo(px, top); g.lineTo(px, bottom); g.stroke(); g.globalAlpha = 1; g.fillStyle = colour; g.beginPath(); g.moveTo(px, bottom); g.lineTo(px - 4, bottom - 6); g.lineTo(px + 4, bottom - 6); g.closePath(); g.fill(); g.fillText(formatAxisDistance(relative, 1), px, top - 11)});
  }
  function drawValues(g, profile, key, colour, transform, x, y, width = 1.5, top = 0, bottom = chart.clientHeight, alpha = 1) {const data = profile.data; g.save(); g.beginPath(); g.rect(pad.l, top, Math.max(1, chart.clientWidth - pad.l - pad.r), Math.max(1, bottom - top)); g.clip(); g.globalAlpha = alpha; g.strokeStyle = colour; g.lineWidth = width; g.lineJoin = 'round'; g.setLineDash([]); g.beginPath(); let open = false; data[key].forEach((v, i) => {if (v === null) {open = false; return} const px = x(toAxis(transform, data.distance_m[i])), py = y(v); open ? g.lineTo(px, py) : g.moveTo(px, py); open = true}); g.stroke(); g.restore()}
  function drawLabel(g, parts, left, top, right) {
    g.font = '600 11px system-ui'; g.textAlign = 'left'; g.textBaseline = 'top'; let lx = left;
    for (const [text, colour] of parts) {const width = g.measureText(text).width; if (lx + width > right) {g.fillStyle = '#6d7773'; g.fillText('…', lx, top); break} g.fillStyle = colour; g.fillText(text, lx, top); lx += width + 14}
  }
  function drawChart() {
    if (!state.profiles.length) return;
    const rows = chartRows(), dpr = devicePixelRatio || 1, w = Math.max(320, chart.clientWidth), h = chartHeight(rows.length), rowHeight = h / rows.length, g = chart.getContext('2d');
    const domain = profileDomain(), span = domain.maximum - domain.minimum, vertical = verticalDomain(), transforms = profileTransforms(), focus = curveFocus(), near = state.hoverCurve;
    updateVerticalButtons();
    chart.style.height = `${h}px`; chart.width = Math.round(w * dpr); chart.height = Math.round(h * dpr); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h); g.fillStyle = '#fff'; g.fillRect(0, 0, w, h);
    const x = d => pad.l + (d - domain.minimum) / span * (w - pad.l - pad.r);
    rows.forEach((row, index) => {
      const top = index * rowHeight + pad.t, bottom = (index + 1) * rowHeight - pad.b, y = v => top + (vertical.maximum - v) / vertical.span * (bottom - top);
      drawAxes(g, x, y, top, bottom, domain.minimum, domain.maximum, vertical.minimum, vertical.maximum, index === rows.length - 1);
      drawReference(g, x, top, bottom, domain.minimum, domain.maximum);
      row.profiles.forEach(profile => drawBreaks(g, profile, transforms.get(profile), x, top, bottom));
      // Dimmed curves first, then the ordinary ones, the highlighted ones on top.
      const rank = item => focus ? (curveInFocus(item, focus) ? 2 : 0) : near && near.profile === item.profile && near.key === item.key ? 2 : 1;
      row.curves.slice().sort((a, b) => rank(a) - rank(b)).forEach(item => {const r = rank(item); drawValues(g, item.profile, item.key, item.colour, transforms.get(item.profile), x, y, r === 2 ? 2.6 : 1.5, top, bottom, r === 0 ? .16 : 1)});
      drawLabel(g, row.label, pad.l + 6, top + 2, w - pad.r);
      if (state.hoverDistance === null) return;
      const px = x(state.hoverDistance); g.strokeStyle = '#17201e'; g.lineWidth = 1; g.beginPath(); g.moveTo(px, top); g.lineTo(px, bottom); g.stroke();
      row.curves.forEach(item => {
        if (focus && !curveInFocus(item, focus)) return;
        const value = valueAt(item.profile, item.key, transforms.get(item.profile), state.hoverDistance); if (value === null || value < vertical.minimum || value > vertical.maximum) return;
        g.beginPath(); g.arc(px, y(value), near && near.profile === item.profile && near.key === item.key ? 4.5 : 3, 0, Math.PI * 2); g.fillStyle = item.colour; g.fill(); g.lineWidth = 1.2; g.strokeStyle = '#fff'; g.stroke();
      });
    });
  }
  chart.addEventListener('mousemove', event => {
    if (!state.profiles.length) return;
    const r = chart.getBoundingClientRect(), domain = profileDomain(), rows = chartRows(), rowHeight = r.height / rows.length, index = Math.max(0, Math.min(rows.length - 1, Math.floor((event.clientY - r.top) / rowHeight))), row = rows[index];
    state.hoverProfile = state.profileMode === 'stacked' && rows.length > 1 ? row.profiles[0] : null;
    state.hoverDistance = Math.max(domain.minimum, Math.min(domain.maximum, domain.minimum + (event.clientX - r.left - pad.l) / (r.width - pad.l - pad.r) * (domain.maximum - domain.minimum)));
    // The curve under the cursor is drawn thicker and named first: this is how a line is followed among many.
    const vertical = verticalDomain(), transforms = profileTransforms(), top = index * rowHeight + pad.t, bottom = (index + 1) * rowHeight - pad.b, focus = curveFocus();
    const values = row.curves.filter(item => !focus || curveInFocus(item, focus)).map(item => ({item, value: valueAt(item.profile, item.key, transforms.get(item.profile), state.hoverDistance)})).filter(entry => entry.value !== null);
    values.forEach(entry => {entry.gap = Math.abs(top + (vertical.maximum - entry.value) / vertical.span * (bottom - top) - (event.clientY - r.top))});
    const nearest = values.reduce((best, entry) => entry.gap < 16 && (!best || entry.gap < best.gap) ? entry : best, null);
    state.hoverCurve = nearest ? {profile: nearest.item.profile, key: nearest.item.key} : null; markNearestCell();
    const offsetText = Math.abs(state.hoverDistance) < .05 ? '0.0' : `${state.hoverDistance > 0 ? '+' : '−'}${Math.abs(state.hoverDistance).toFixed(1)}`;
    const text = entry => `${entry === nearest ? '▸ ' : ''}${DV.COLORS[entry.item.profile.shape.slot][1]} ${channelLabel(entry.item.key)}: ${entry.value}`;
    readout.textContent = readout.title = `${offsetText} м от точки совмещения · яркость ${[...(nearest ? [nearest] : []), ...values.filter(entry => entry !== nearest)].map(text).join(' · ') || '—'}`; drawChart(); view.redraw();
  });
  chart.addEventListener('mouseleave', () => {clearHover(); markNearestCell(); readout.textContent = idleReadout; drawChart(); view.redraw()});
  chart.addEventListener('wheel', event => {
    if (!state.profiles.length) return;
    event.preventDefault();
    const full = fullProfileDomain(), current = profileDomain(), rect = chart.getBoundingClientRect(), fraction = Math.max(0, Math.min(1, (event.clientX - rect.left - pad.l) / Math.max(1, rect.width - pad.l - pad.r))), centre = current.minimum + fraction * (current.maximum - current.minimum), factor = event.deltaY < 0 ? .72 : 1 / .72, minimumSpan = Math.max((full.maximum - full.minimum) / 100, .5), span = Math.max(minimumSpan, Math.min(full.maximum - full.minimum, (current.maximum - current.minimum) * factor));
    let minimum = centre - fraction * span, maximum = minimum + span;
    if (minimum < full.minimum) {maximum += full.minimum - minimum; minimum = full.minimum}
    if (maximum > full.maximum) {minimum -= maximum - full.maximum; maximum = full.maximum}
    state.profileWindow = span >= full.maximum - full.minimum - 1e-6 ? null : {minimum, maximum};
    clearHover(); markNearestCell(); readout.textContent=state.profileWindow?`масштаб: ${(state.profileWindow.maximum-state.profileWindow.minimum).toFixed(1)} м`:'вся длина'; drawChart();
  }, {passive:false});
  new ResizeObserver(() => drawChart()).observe(panel);

  function setVariant(nextVariant) {
    if (!nextVariant || nextVariant.path === variant.path) return;
    variant = nextVariant; entry = DV.annotationStore.entry(variant); shapes = entry.shapes;
    Object.assign(state, {mode: 'pan', slot: firstFreeSlot(0), drawing: null, cursor: null, selected: -1,
      down: null, profiles: [], profileWindow: null, profileYSpan: 255, hoverDistance: null, hoverProfile: null, hoverCurve: null,
      profileFocus: null, profilePin: null, syncTarget: null, syncStep: 1, syncChain: false, syncPreview: null});
    hideMenu(); updateProfilePanel(); renderList(); syncBar(); say('Разметка относится только к открытому кадру. Для сопоставления используйте тот же цвет на соседнем кадре.');
    api_.variant = variant; api_.shapes = shapes; view.redraw();
  }
  const api_ = {handlers, draw, syncOverlays, setVariant, variant, shapes,
    screenPoints: () => shapes.map(shape => shape.points.map(([e, n]) => view.imagePoint(e, n)))};
  DV.viewers.add(api_);
  // Shapes may be changed from outside (contour tracking writes to both frames of a comparison).
  const annotationsChanged = () => {if (!view.card.isConnected) {window.removeEventListener('dv:annotations-changed', annotationsChanged); return} renderList(); syncBar(); view.redraw()};
  window.addEventListener('dv:annotations-changed', annotationsChanged);
  syncBar(); renderList(); syncOverlays();
  return api_;
};

// Displacement of the marked objects between the two frames of a comparison: one button for everything.
// Every colour is an object and is handled on its own, so each gets its own vector.
//  * the same colour marked by hand on both frames: polylines are registered by shape
//    (/api/curve-displacement), closed contours by their two outlines;
//  * a colour marked on one frame only: the shape is searched for on the other frame by image matching
//    (dronevision/contour_tracking.py) and added there in the same colour (`track.auto`).
// `track.start`/`track.end` are the object's positions at the earlier/later time (ENU metres); `track.ghost` is
// the earlier shape carried by the estimated motion (on an automatically found shape: the earlier shape itself).
DV.motionShapes = variants => variants.flatMap(v => ((v && DV.annotationStore.images[v.path]) || {shapes: []}).shapes
  .filter(s => (s.type === 'polygon' ? s.points.length > 2 : s.points.length > 1) && !(s.track && s.track.auto)));
// An open line is a polyline or a two-point segment: both are registered as lines (dronevision/motion.py).
DV.isLine = shape => shape.type !== 'polygon';
DV.computeMotion = async (variants, button, results, panel) => {
  const [a, b] = variants, label = button.textContent;
  if (!a || !b) return;
  const flash = text => {button.textContent = text; setTimeout(() => button.textContent = label, 3000)};
  const store = DV.annotationStore, frames = [a, b], entries = [store.entry(a), store.entry(b)];
  const drawn = [DV.motionShapes([a]), DV.motionShapes([b])];
  if (!drawn[0].length && !drawn[1].length) {flash('Сначала отметьте объект: «✎ Ломаная» или «⬠ Контур»'); return}
  const kind = shape => shape.type === 'polygon' ? 'контур' : shape.type === 'segment' ? 'отрезок' : 'ломаная', rows = [], curves = [];
  // job: shapes of frame `from` followed to frame `to`; `other` = the hand-drawn contour of the same colour there
  const jobs = [{from: 0, to: 1, items: []}, {from: 1, to: 0, items: []}];
  drawn[0].forEach(shape => {
    const other = drawn[1].find(s => s.slot === shape.slot);
    if (other && DV.isLine(other) !== DV.isLine(shape)) rows.push({slot: shape.slot, kind: kind(shape), error: `на соседнем кадре этим цветом отмечен${kind(other) === 'контур' ? ' контур, а не линия' : 'а линия, а не контур'}; используйте фигуры одного типа`});
    else if (other && DV.isLine(shape)) curves.push({slot: shape.slot, a: shape, b: other});
    else jobs[0].items.push({shape, other});
  });
  drawn[1].filter(shape => !drawn[0].some(s => s.slot === shape.slot)).forEach(shape => jobs[1].items.push({shape}));
  button.disabled = true; button.textContent = 'Считаю смещение…';
  try {
    const dt = b.time_s - a.time_s;
    for (const pair of curves) {
      try {
        const value = await api('/api/curve-displacement', {method: 'POST', body: JSON.stringify({points_a: pair.a.points, points_b: pair.b.points, delta_time_s: dt})});
        DV.setPolylineTrack({...pair, dt}, value, [a.path, b.path]);
        rows.push({slot: pair.slot, kind: kind(pair.a), manual: true, ...pair.a.track, overlap_length_m: value.overlap_length_m, residual_m: value.residual_m, warning: value.warning});
      } catch (error) {delete pair.a.track; delete pair.b.track; rows.push({slot: pair.slot, kind: kind(pair.a), error: error.message})}
    }
    for (const job of jobs.filter(job => job.items.length)) {
      const response = await api('/api/track-contours', {method: 'POST', body: JSON.stringify({path_a: frames[job.from].path, path_b: frames[job.to].path,
        shapes: job.items.map(({shape, other}) => ({slot: shape.slot, type: DV.isLine(shape) ? 'polyline' : 'polygon', points: shape.points, ...(other ? {points_b: other.points} : {})}))})});
      const forward = response.delta_time_s > 0, sign = forward ? 1 : -1;
      response.results.forEach(r => {
        const item = job.items.find(x => x.shape.slot === r.slot), target = entries[job.to].shapes, old = target.findIndex(s => s.slot === r.slot);
        if (!item) return;
        const {shape, other} = item, closed = shape.type === 'polygon';
        if (r.error) {delete shape.track; if (other) delete other.track; else if (old >= 0 && target[old].track && target[old].track.auto) target.splice(old, 1); rows.push({slot: r.slot, kind: kind(shape), error: r.error}); return}
        const info = {east_m: sign * r.east_m, north_m: sign * r.north_m, displacement_m: r.displacement_m, speed_m_s: r.speed_m_s,
          azimuth_deg: forward ? r.azimuth_deg : (r.azimuth_deg + 180) % 360, rotation_deg: sign * r.rotation_deg, delta_time_s: Math.abs(r.delta_time_s),
          score: r.score, residual_m: r.residual_m, rotation_found: r.rotation_found, manual: !!r.manual, quality: r.quality, warning: r.warning || '',
          area_m2: r.area_m2, length_m: r.length_m, start: forward ? r.centroid_a : r.centroid_b, end: forward ? r.centroid_b : r.centroid_a, closed};
        shape.track = {...info, auto: false, ghost: r.points_b, other: frames[job.to].path};
        if (other) other.track = {...info, auto: false, ghost: r.points_b, other: frames[job.from].path};  // both are the user's: nothing is replaced
        else {
          const copy = {slot: r.slot, type: shape.type, points: r.points_b, name: shape.name, hidden: false, track: {...info, auto: true, ghost: shape.points, other: frames[job.from].path}};
          if (old >= 0) target.splice(old, 1, copy); else target.push(copy);
        }
        rows.push({slot: r.slot, kind: kind(shape), ...info});
      });
    }
    entries.forEach(entry => {entry.saved = Date.now()}); store.save();
    rows.sort((x, y) => x.slot - y.slot);
    const good = rows.filter(r => !r.error && r.quality !== 'low'), signed = v => `${v >= 0 ? '+' : ''}${v.toFixed(2)}`;
    const range = (key, scale = 1, digits = 1) => {const v = good.map(r => r[key] * scale); return `${Math.min(...v).toFixed(digits)}…${Math.max(...v).toFixed(digits)}`};
    const summary = `<small>${good.length > 1 ? `Каждый объект посчитан отдельно: скорость ${range('speed_m_s', 100)} см/с · азимут ${range('azimuth_deg', 1, 0)}°. ` : ''}Стрелка — вектор смещения; пунктир — фигура раннего кадра, перенесённая этим вектором: на позднем кадре она должна лечь на объект.</small>`;
    results.innerHTML = summary + rows.map(r => {
      const check = `<input type="checkbox" data-motion-export-slot="${r.slot}"${r.error ? ' disabled' : ' checked'} title="${r.error ? 'Этот вектор не рассчитан' : 'Сохранять этот вектор в TXT'}">`;
      const head = `<article style="--c:${DV.COLORS[r.slot][0]}"><label class="motion-vector-choice">${check}<small><i></i>${DV.COLORS[r.slot][1]} · ${r.kind}`;
      if (r.error) return `${head} · не посчитано</small></label><small class="motion-warning">${r.error}</small></article>`;
      const closed = r.kind === 'контур';
      const quality = r.manual ? (r.quality === 'good' ? (closed ? 'контуры согласуются' : 'надёжно') : r.quality === 'acceptable' ? (closed ? 'контуры согласуются приблизительно' : 'проверьте визуально') : (closed ? 'контуры сильно различаются по форме или площади' : 'низкая уверенность'))
        : (r.quality === 'good' ? 'надёжно' : r.quality === 'acceptable' ? 'проверьте по пунктиру на соседнем кадре' : 'низкая уверенность — фигура, вероятно, найдена неверно');
      const source = !r.manual ? `найдено автоматически на соседнем кадре · совпадение ${r.score.toFixed(2)}`
        : closed ? `по двум вашим контурам · невязка формы ${r.residual_m.toFixed(2)} м` : `кратчайший сдвиг, совмещающий две ваши линии · общий участок ${r.overlap_length_m.toFixed(1)} м · расхождение линий ${r.residual_m.toFixed(2)} м`;
      const rotation = !Number.isFinite(r.rotation_deg) ? '' : r.manual && !r.rotation_found ? ' · поворот по контурам не определяется' : ` · поворот ${signed(r.rotation_deg)}° (${r.rotation_deg >= 0 ? 'по часовой' : 'против часовой'})`;
      const size = closed ? `Площадь ${formatArea(r.area_m2)} · ` : Number.isFinite(r.length_m) ? `Длина ${formatLength(r.length_m)} · ` : '';
      return `${head} · ${r.displacement_m.toFixed(2)} м за ${r.delta_time_s.toFixed(2)} с</small></label><strong>${r.speed_m_s.toFixed(3)} м/с <small>(${(r.speed_m_s * 100).toFixed(1)} см/с)</small></strong><span>на восток ${signed(r.east_m)} м · на север ${signed(r.north_m)} м · азимут ${r.azimuth_deg.toFixed(1)}°${rotation}</span><small${r.quality === 'low' ? ' class="motion-warning"' : ''}>${size}${source} · ${quality}${r.warning ? ` · ${r.warning}` : ''}</small></article>`;
    }).join('');
    panel.classList.remove('empty');
    const save = panel.querySelector('[data-motion-save]'); if (save) save.hidden = !rows.some(r => !r.error);
  } catch (error) {results.innerHTML = `<p class="motion-warning">${error.message}</p>`; panel.classList.remove('empty')}
  finally {button.disabled = false; button.textContent = label}
};

// Save the vectors of the comparison (the result of DV.computeMotion) as a tab-separated TXT.
DV.exportMotion = async (variants, button, selectedSlots = null) => {
  const [a, b] = variants, label = button.textContent;
  if (!a || !b) return;
  const flash = text => {button.textContent = text; setTimeout(() => button.textContent = label, 4000)};
  const images = DV.annotationStore.images, early = (images[a.path] || {shapes: []}).shapes, late = (images[b.path] || {shapes: []}).shapes;
  // The earlier frame holds every tracked object: drawn there, or found there automatically from the later frame.
  const items = early.filter(s => s.track && s.track.other === b.path && (!selectedSlots || selectedSlots.includes(s.slot))).sort((x, y) => x.slot - y.slot).map(s => {
    const other = late.find(x => x.slot === s.slot), auto = s.track.auto || !!(other && other.track && other.track.auto);
    return {name: s.name || `obj${s.slot + 1}`, type: DV.isLine(s) ? 'polyline' : 'polygon', method: auto ? 'auto' : 'manual', points: s.points, start: s.track.start,
      east_m: s.track.east_m, north_m: s.track.north_m, rotation_deg: Number.isFinite(s.track.rotation_deg) && !(s.track.manual && s.track.rotation_found === false) ? s.track.rotation_deg : null, quality: s.track.quality};
  });
  if (!items.length) {flash(selectedSlots ? 'Отметьте хотя бы один вектор' : 'Сначала рассчитайте смещение'); return}
  button.disabled = true;
  try {
    const r = await api('/api/motion-export', {method: 'POST', body: JSON.stringify({path_a: a.path, path_b: b.path, items})});
    downloadBlob(new Blob([r.text], {type: 'text/plain;charset=utf-8'}), r.filename);
    button.title = `Последний файл: ${r.path}`; flash(`Сохранено векторов: ${r.count}`);
  } catch (error) {flash(error.message)} finally {button.disabled = false}
};

// Vector of a polyline pair (same colour on both frames) after "Рассчитать смещение и скорость": the earlier
// line shifted by the estimated displacement is drawn dashed on both frames, with the arrow at its middle.
DV.setPolylineTrack = (pair, value, paths) => {
  const forward = pair.dt >= 0, early = forward ? pair.a : pair.b, sign = forward ? 1 : -1;
  const east = sign * value.east_m, north = sign * value.north_m, total = shapeLength(early.points);
  let rest = total / 2, start = early.points[0];
  for (let i = 1; i < early.points.length; i++) {
    const p = early.points[i - 1], q = early.points[i], len = Math.hypot(q[0] - p[0], q[1] - p[1]);
    if (rest <= len || i === early.points.length - 1) {const t = len ? Math.min(1, rest / len) : 0; start = [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]; break}
    rest -= len;
  }
  const info = {east_m: east, north_m: north, displacement_m: value.displacement_m, speed_m_s: value.speed_m_s, azimuth_deg: forward ? value.azimuth_deg : (value.azimuth_deg + 180) % 360,
    delta_time_s: Math.abs(value.delta_time_s), quality: value.quality, manual: true, auto: false, closed: false,
    start, end: [start[0] + east, start[1] + north], ghost: early.points.map(([e, n]) => [e + east, n + north])};
  pair.a.track = {...info, other: paths[1]}; pair.b.track = {...info, other: paths[0]};
};

// "Открыть исходный PNG" + GeoTIFF export row under a projection/mosaic viewer.
DV.exportRow = (variant, link) => {
  const row = document.createElement('div'), select = document.createElement('select'), button = document.createElement('button'), note = document.createElement('span');
  row.className = 'export-row';
  select.innerHTML = '<option value="utm">UTM (метры)</option><option value="wgs84">WGS84 (градусы)</option>';
  select.title = 'Система координат GeoTIFF';
  button.type = 'button'; button.textContent = 'Экспортировать в GeoTIFF';
  button.title = 'Растр без сетки, с текущей цветовой коррекцией проекции; прозрачность там, где нет кадра';
  button.onclick = async () => {
    button.disabled = true; note.textContent = 'Пересчитываю в выбранную проекцию…';
    try {
      const response = await fetch(`/api/export-geotiff?path=${encodeURIComponent(variant.path)}&crs=${select.value}`);
      if (!response.ok) {let message = 'Ошибка экспорта'; try {message = (await response.json()).error || message} catch (e) {} throw Error(message)}
      const path = decodeURIComponent(response.headers.get('X-DroneVision-Path') || ''), crs = decodeURIComponent(response.headers.get('X-DroneVision-CRS') || '');
      downloadBlob(await response.blob(), path.split(/[/\\]/).pop() || 'projection.tif');
      note.textContent = `${crs} · ${path}`;
    } catch (e) {note.textContent = e.message} finally {button.disabled = false}
  };
  row.append(link, button, select, note);
  return row;
};
