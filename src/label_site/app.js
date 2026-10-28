/* Label globe: where the So2Sat and WUDAPT labels are.
 *
 * Two halves. `LabelCore` is pure (typed arrays in, typed arrays out) so it
 * runs under QuickJS in tests/test_label_site_client_js.py; everything below
 * it needs a browser, MapLibre and deck.gl.
 *
 * Each dataset owns one map. "Both" shows the two maps side by side with their
 * cameras locked together -- the datasets are never overlaid on one map.
 */
(function (root) {
  'use strict';

  var ITEMSIZE = { f4: 4, u4: 4, u2: 2, u1: 1 };
  var CTOR = { f4: Float32Array, u4: Uint32Array, u2: Uint16Array, u1: Uint8Array };
  var U2_MISSING = 65535;
  var SLIDER_STEPS = 1000;

  // ── Pure core ─────────────────────────────────────────────────────────────
  var Core = {
    /* One typed array per field, laid end to end; offsets are n * itemsize. */
    decodeBlock: function (buffer, fields, n) {
      var out = {}, offset = 0;
      for (var i = 0; i < fields.length; i++) {
        var f = fields[i];
        out[f.name] = new CTOR[f.dtype](buffer, offset, n);
        offset += n * ITEMSIZE[f.dtype];
      }
      return out;
    },

    /* Bbox centre of every label: the globe-scale dot. */
    centres: function (rows, n) {
      var c = new Float32Array(2 * n), h = 65535 / 2;
      for (var i = 0; i < n; i++) {
        c[2 * i] = rows.x0[i] + rows.sx[i] * h;
        c[2 * i + 1] = rows.y0[i] + rows.sy[i] * h;
      }
      return c;
    },

    /* uint16 ring offsets -> closed float rings + startIndices (vertex units). */
    decodeRings: function (buffer, rings, rows, n) {
      var counts = new Uint16Array(buffer, 0, n);
      var q = new Uint16Array(buffer, rings.vertex_offset, 2 * rings.n_vertices);
      var starts = new Uint32Array(n + 1);
      var pos = new Float32Array(2 * (rings.n_vertices + n));
      var src = 0, dst = 0;
      for (var i = 0; i < n; i++) {
        starts[i] = dst / 2;
        var x0 = rows.x0[i], y0 = rows.y0[i], sx = rows.sx[i], sy = rows.sy[i];
        var first = dst;
        for (var k = 0; k < counts[i]; k++) {
          pos[dst++] = x0 + q[src++] * sx;
          pos[dst++] = y0 + q[src++] * sy;
        }
        pos[dst++] = pos[first];               // close the ring again
        pos[dst++] = pos[first + 1];
      }
      starts[n] = dst / 2;
      return { positions: pos, startIndices: starts };
    },

    /* The value of column `name` for every row (submission columns expanded). */
    columnFor: function (ds, name) {
      var col = ds.manifest.columns[name];
      if (col.table === 'rows') return ds.rows[name];
      var sub = ds.subs[name], key = ds.rows[ds.manifest.subs.key];
      var out = new sub.constructor(ds.n);
      for (var i = 0; i < ds.n; i++) out[i] = sub[key[i]];
      return out;
    },

    isMissing: function (col, v) {
      return col.dtype === 'u2' ? v === U2_MISSING : !(v === v);   // NaN
    },

    sliderToValue: function (col, t) {
      var u = t / SLIDER_STEPS;
      if (col.scale === 'log') {
        var a = Math.log(col.min), b = Math.log(col.max);
        return Math.exp(a + u * (b - a));
      }
      return col.min + u * (col.max - col.min);
    },

    /* Inverse of sliderToValue, clamped to the track and snapped to a step.
       The number boxes need it to turn a typed value back into slider state. */
    valueToSlider: function (col, v) {
      var u;
      if (col.scale === 'log') {
        var a = Math.log(col.min), b = Math.log(col.max);
        u = b === a ? 0 : (Math.log(Math.max(v, Number.MIN_VALUE)) - a) / (b - a);
      } else {
        u = col.max === col.min ? 0 : (v - col.min) / (col.max - col.min);
      }
      if (!(u === u)) u = 0;                                    // NaN from a bad parse
      return Math.max(0, Math.min(SLIDER_STEPS, Math.round(u * SLIDER_STEPS)));
    },

    /* The value a slider step actually filters at.
       Integer columns are rounded HERE rather than only when displayed. The two
       year filters span ~35 values over 1000 steps, so a step lands on 2016.34;
       showing "2016" while testing `v >= 2016.34` dropped every 2016 submission
       the label claimed to include. Display and filter now read the same number
       because they call the same function. */
    thresholdFor: function (col, t) {
      var v = Core.sliderToValue(col, t);
      return col.format === 'd' ? Math.round(v) : v;
    },

    /* Step for the typed boxes: integers for 'd', fine for a ratio, free on a
       log axis where the span covers many orders of magnitude. */
    stepFor: function (col) {
      if (col.format === 'd') return '1';
      if (col.scale === 'log' || col.format === '.3g') return 'any';
      return '0.01';
    },

    /* Default, i.e. inactive, filter state for a column. */
    initialState: function (col) {
      if (col.kind === 'range') return { lo: 0, hi: SLIDER_STEPS, missing: true };
      var n = col.levels.length, sel = new Uint8Array(n);
      for (var i = 0; i < n; i++) sel[i] = 1;
      return { sel: sel };
    },

    /* Code -> level index for the LCZ column (codes are 1..19, not 0-based). */
    lczIndex: function (col) {
      var idx = new Int16Array(256);
      for (var i = 0; i < 256; i++) idx[i] = -1;
      col.levels.forEach(function (lv, j) { idx[lv.code] = j; });
      return idx;
    },

    /* A per-table-row pass array for one filter, or null when it filters nothing. */
    passFor: function (ds, name, st) {
      var col = ds.manifest.columns[name];
      var vals = col.table === 'rows' ? ds.rows[name] : ds.subs[name];
      var m = vals.length, pass = new Uint8Array(m), i;
      if (col.kind === 'range') {
        if (st.lo === 0 && st.hi === SLIDER_STEPS && st.missing) return null;
        var lo = st.lo === 0 ? -Infinity : Core.thresholdFor(col, st.lo);
        var hi = st.hi === SLIDER_STEPS ? Infinity : Core.thresholdFor(col, st.hi);
        for (i = 0; i < m; i++) {
          var v = vals[i];
          pass[i] = Core.isMissing(col, v) ? (st.missing ? 1 : 0)
            : (v >= lo && v <= hi ? 1 : 0);
        }
        return pass;
      }
      var all = true;
      for (i = 0; i < st.sel.length; i++) if (!st.sel[i]) { all = false; break; }
      if (all) return null;
      var lookup = col.kind === 'lcz' ? Core.lczIndex(col) : null;
      for (i = 0; i < m; i++) {
        var c = lookup ? lookup[vals[i]] : vals[i];
        pass[i] = c >= 0 && st.sel[c] ? 1 : 0;
      }
      return pass;
    },

    /* fails[i] has bit k set when row i fails filter k; mask = fails === 0. */
    evaluate: function (ds, state) {
      var n = ds.n, fails = new Uint32Array(n), names = ds.manifest.filters;
      var key = ds.manifest.subs ? ds.rows[ds.manifest.subs.key] : null;
      for (var k = 0; k < names.length; k++) {
        var pass = Core.passFor(ds, names[k], state[names[k]]);
        if (!pass) continue;
        var bit = 1 << k, bySub = ds.manifest.columns[names[k]].table === 'subs';
        for (var i = 0; i < n; i++) {
          if (!pass[bySub ? key[i] : i]) fails[i] |= bit;
        }
      }
      var mask = new Uint8Array(n), shown = 0;
      for (var j = 0; j < n; j++) if (fails[j] === 0) { mask[j] = 1; shown++; }
      return { fails: fails, mask: mask, shown: shown };
    },

    /* Rows per level of filter k among rows passing every OTHER filter. */
    levelCounts: function (ds, fails, k) {
      var name = ds.manifest.filters[k], col = ds.manifest.columns[name];
      var counts = new Float64Array(col.levels.length);
      var vals = col.table === 'rows' ? ds.rows[name] : ds.subs[name];
      var key = col.table === 'subs' ? ds.rows[ds.manifest.subs.key] : null;
      var lookup = col.kind === 'lcz' ? Core.lczIndex(col) : null;
      var others = ~(1 << k) >>> 0;
      for (var i = 0; i < ds.n; i++) {
        if ((fails[i] & others) !== 0) continue;
        var v = vals[key ? key[i] : i];
        var c = lookup ? lookup[v] : v;
        if (c >= 0) counts[c]++;
      }
      return counts;
    },

    formatValue: function (col, v) {
      if (col.kind === 'lcz') {
        var lv = col.levels.filter(function (l) { return l.code === v; })[0];
        return lv ? (v <= 17 ? 'LCZ ' + v + ' · ' : '') + lv.name : String(v);
      }
      if (col.kind === 'categorical') return col.levels[v];
      if (col.kind === 'id') return String(v);
      if (Core.isMissing(col, v)) return '—';
      if (col.format === 'd') return String(Math.round(v));
      if (col.format === '.3g') return Number(v).toPrecision(3);
      return Number(v).toFixed(2);
    },
  };
  root.LabelCore = Core;

  if (typeof document === 'undefined' || !root.maplibregl || !root.deck) return;

  // ── Browser half ──────────────────────────────────────────────────────────
  var SWITCH_ZOOM = 10;       // dots below, polygons from here (320 m ≈ 4 px)
  var MANIFEST = null;
  var DS = {};                // key -> loaded dataset
  var MAPS = {};              // key -> { map, overlay, el }
  var UI = { mode: 'so2sat', panel: 'so2sat', basemap: 'map',
             overlays: { guppd: false, demuzere: false }, demuzereOpacity: 0.7 };
  var GUPPD = null;           // overlay outlines, loaded on first toggle
  var syncing = false;
  root.labelGlobe = { maps: MAPS, data: DS, ui: UI,             // console handle
                      guppd: function () { return GUPPD; } };

  function $(sel, el) { return (el || document).querySelector(sel); }
  function h(tag, attrs, kids) {
    var e = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === 'text') e.textContent = attrs[k];
      else if (k === 'class') e.className = attrs[k];
      else if (k.slice(0, 2) === 'on') e.addEventListener(k.slice(2), attrs[k]);
      else e.setAttribute(k, attrs[k]);
    });
    (kids || []).forEach(function (c) { if (c) e.appendChild(c); });
    return e;
  }
  function fmt(n) { return Math.round(n).toLocaleString('en-GB'); }

  /* A failure has to be visible where the controls are.
   *
   * Everything here is wired from one fetch, so a single failed request used to
   * leave a page that looks complete and answers nothing: the checkboxes still
   * render and still tick, they just drive nothing. The banner goes at the top
   * of the sidebar, not in the map pane, because the sidebar is what you are
   * looking at when you press a filter. */
  function showError(msg) {
    var bar = $('#load-error');
    if (!bar) {
      bar = h('p', { class: 'load-error', id: 'load-error' });
      var side = $('#sidebar');
      if (side) side.insertBefore(bar, side.firstChild);
      else document.body.appendChild(bar);
    }
    bar.textContent = msg;
    bar.hidden = false;
  }
  if (root.addEventListener) {           // absent under the offline JS runner
    root.addEventListener('error', function (e) {
      showError('Script error: ' + ((e && e.message) || e) +
                ' — the controls below may do nothing.');
    });
    root.addEventListener('unhandledrejection', function (e) {
      var r = e && e.reason;
      showError('Failed: ' + ((r && r.message) || r) +
                ' — the controls below may do nothing.');
    });
  }
  function hexRgb(hex) {
    var v = parseInt(hex.slice(1), 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  }
  function fetchBuf(path) {
    return fetch(path).then(function (r) {
      if (!r.ok) throw new Error(path + ': HTTP ' + r.status);
      return r.arrayBuffer();
    });
  }

  function basemapStyle() {
    var dem = MANIFEST.overlays.demuzere;
    var style = {
      version: 8,
      projection: { type: 'globe' },
      sources: {
        // Esri's Light Gray Canvas: key-free (CARTO's raster tiles now answer
        // with an "API key required" placeholder).
        grey: {
          type: 'raster', tileSize: 256, maxzoom: 16,
          tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'],
          attribution: 'Basemap © Esri, HERE, Garmin, © OpenStreetMap contributors',
        },
        esri: {
          type: 'raster', tileSize: 256, maxzoom: 19,
          tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
          attribution: 'Imagery © Esri, Maxar, Earthstar Geographics',
        },
      },
      layers: [
        { id: 'bg', type: 'background', paint: { 'background-color': '#dfe6ea' } },
        { id: 'grey', type: 'raster', source: 'grey' },
        { id: 'esri', type: 'raster', source: 'esri', layout: { visibility: 'none' } },
      ],
    };
    if (dem && dem.url) {
      // One PMTiles archive read by range requests (build_demuzere_pmtiles.py).
      style.sources.demuzere = { type: 'raster', url: 'pmtiles://' + dem.url,
                                 tileSize: 256, attribution: dem.attribution };
      style.layers.push({ id: 'demuzere', type: 'raster', source: 'demuzere',
        layout: { visibility: UI.overlays.demuzere ? 'visible' : 'none' },
        paint: { 'raster-opacity': UI.demuzereOpacity, 'raster-resampling': 'nearest',
                 'raster-fade-duration': 0 } });
    }
    return style;
  }

  // ── Data ──────────────────────────────────────────────────────────────────
  function loadDataset(key) {
    if (DS[key]) return DS[key].ready;
    var m = MANIFEST.datasets[key];
    var ds = { key: key, manifest: m, n: m.n, rings: null, ringsLoading: null,
               state: {}, version: 0, result: null };
    m.filters.forEach(function (f) { ds.state[f] = Core.initialState(m.columns[f]); });
    DS[key] = ds;
    var jobs = [fetchBuf(m.rows.file)];
    if (m.subs) jobs.push(fetchBuf(m.subs.file));
    ds.ready = Promise.all(jobs).then(function (bufs) {
      ds.rows = Core.decodeBlock(bufs[0], m.rows.fields, m.n);
      if (m.subs) ds.subs = Core.decodeBlock(bufs[1], m.subs.fields, m.subs.n);
      ds.centres = Core.centres(ds.rows, m.n);
      var lcz = m.columns.lcz, rgb = {};
      lcz.levels.forEach(function (lv) { rgb[lv.code] = hexRgb(lv.colour); });
      ds.colour = new Uint8Array(4 * m.n);
      for (var i = 0; i < m.n; i++) {
        var c = rgb[ds.rows.lcz[i]] || [0, 0, 0];
        ds.colour.set([c[0], c[1], c[2], 225], 4 * i);
      }
      ds.result = Core.evaluate(ds, ds.state);
      return ds;
    });
    return ds.ready;
  }

  function loadRings(ds) {
    if (ds.ringsLoading) return ds.ringsLoading;
    setStatus(ds.key, 'loading polygons…');
    ds.ringsLoading = fetchBuf(ds.manifest.rings.file).then(function (buf) {
      ds.rings = Core.decodeRings(buf, ds.manifest.rings, ds.rows, ds.n);
      setStatus(ds.key, '');
      renderLayers(ds.key);
    }).catch(function (e) { setStatus(ds.key, 'polygons failed: ' + e.message); });
    return ds.ringsLoading;
  }

  function loadGuppd() {
    if (GUPPD) return GUPPD.ready;
    var m = MANIFEST.overlays.guppd;
    GUPPD = { manifest: m, n: m.n, version: 0 };
    GUPPD.levelOn = new Uint8Array(m.columns.level.levels.length).fill(1);
    setStatus('guppd', 'loading…');
    GUPPD.ready = Promise.all([fetchBuf(m.rows.file), fetchBuf(m.rings.file),
                               fetch(m.names).then(function (r) { return r.text(); })])
      .then(function (res) {
        GUPPD.rows = Core.decodeBlock(res[0], m.rows.fields, m.n);
        GUPPD.rings = Core.decodeRings(res[1], m.rings, GUPPD.rows, m.n);
        GUPPD.names = res[2].split('\n');
        GUPPD.colour = m.columns.level.colours.map(hexRgb);
        setStatus('guppd', '');
        Object.keys(MAPS).forEach(renderLayers);
      }).catch(function (e) { setStatus('guppd', 'failed: ' + e.message); });
    return GUPPD.ready;
  }

  function guppdLayer(key) {
    if (!UI.overlays.guppd || !GUPPD || !GUPPD.rings) return null;
    var g = GUPPD, level = g.rows.level, on = g.levelOn;
    return new deck.PathLayer({
      id: key + '-guppd',
      data: { length: g.n, startIndices: g.rings.startIndices,
              attributes: { getPath: { value: g.rings.positions, size: 2 } } },
      _pathType: 'open',
      getColor: function (_, o) { return g.colour[level[o.index]] || [80, 80, 80]; },
      widthUnits: 'pixels', getWidth: 1.4, widthMinPixels: 1,
      extensions: [new deck.DataFilterExtension({ filterSize: 1 })],
      getFilterValue: function (_, o) { return on[level[o.index]]; },
      filterRange: [0.5, 1.5], updateTriggers: { getFilterValue: g.version },
      pickable: true, autoHighlight: true, highlightColor: [20, 20, 20, 255],
    });
  }

  // ── Layers ────────────────────────────────────────────────────────────────
  function layersFor(key) {
    var labels = labelLayers(key), g = guppdLayer(key);
    return g ? labels.concat([g]) : labels;
  }

  function labelLayers(key) {
    var ds = DS[key], mp = MAPS[key];
    if (!ds || !ds.result) return [];
    var zoom = mp.map.getZoom(), near = zoom >= SWITCH_ZOOM;
    var mask = ds.result.mask, colour = ds.colour;
    var filter = new deck.DataFilterExtension({ filterSize: 1 });
    var getFilterValue = function (_, o) { return mask[o.index]; };
    var getFillColor = function (_, o) {
      var j = 4 * o.index;
      return [colour[j], colour[j + 1], colour[j + 2], colour[j + 3]];
    };
    var common = {
      pickable: true, extensions: [filter], filterRange: [0.5, 1.5],
      getFilterValue: getFilterValue, getFillColor: getFillColor,
      updateTriggers: { getFilterValue: ds.version },
      autoHighlight: true, highlightColor: [20, 20, 20, 200],
    };
    if (near && !ds.rings) loadRings(ds);
    if (near && ds.rings) {
      return [new deck.SolidPolygonLayer(Object.assign({
        id: key + '-poly',
        data: { length: ds.n, startIndices: ds.rings.startIndices,
                attributes: { getPolygon: { value: ds.rings.positions, size: 2 } } },
        _normalize: false, _windingOrder: 'CCW',
      }, common))];
    }
    return [new deck.ScatterplotLayer(Object.assign({
      id: key + '-dots',
      data: { length: ds.n,
              attributes: { getPosition: { value: ds.centres, size: 2 } } },
      radiusUnits: 'meters', getRadius: key === 'so2sat' ? 160 : 120,
      radiusMinPixels: 1.8, radiusMaxPixels: 12, stroked: false,
    }, common))];
  }

  function renderLayers(key) {
    var mp = MAPS[key];
    if (mp && mp.overlay) mp.overlay.setProps({ layers: layersFor(key) });
  }

  function guppdTooltip(i) {
    var g = GUPPD, m = g.manifest, name = g.names[g.rows.feat[i]] || '';
    var rowsHtml = m.tooltip.map(function (c) {
      return '<tr><th>' + m.columns[c].label + '</th><td>' +
        escapeHtml(Core.formatValue(m.columns[c], g.rows[c][i])) + '</td></tr>';
    }).join('');
    return { html: '<b>' + escapeHtml(name === 'N/A' || !name ? 'Unnamed settlement' : name) +
             '</b><table>' + rowsHtml + '</table>', className: 'tip' };
  }

  function tooltipFor(key) {
    return function (info) {
      if (info.index == null || info.index < 0) return null;
      if (info.layer && /-guppd$/.test(info.layer.id)) return guppdTooltip(info.index);
      if (!DS[key]) return null;
      var ds = DS[key], m = ds.manifest, i = info.index;
      var rowsHtml = m.tooltip.map(function (name) {
        var col = m.columns[name];
        var v = col.table === 'rows' ? ds.rows[name][i]
          : ds.subs[name][ds.rows[m.subs.key][i]];
        return '<tr><th>' + col.label + '</th><td>' +
          escapeHtml(Core.formatValue(col, v)) + '</td></tr>';
      }).join('');
      return { html: '<table>' + rowsHtml + '</table>', className: 'tip' };
    };
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  // ── Maps ──────────────────────────────────────────────────────────────────
  function makeMap(key) {
    var el = $('#map-' + key);
    var map = new maplibregl.Map({
      container: el, style: basemapStyle(), center: [10, 20], zoom: 1.8,
      attributionControl: { compact: true }, maxPitch: 0,
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
    var overlay = new deck.MapboxOverlay({ interleaved: true, layers: [],
                                           getTooltip: tooltipFor(key) });
    map.addControl(overlay);
    var lastNear = null;
    map.on('zoom', function () {
      var near = map.getZoom() >= SWITCH_ZOOM;
      if (near !== lastNear) { lastNear = near; renderLayers(key); }
    });
    map.on('move', function () {
      if (syncing || UI.mode !== 'both') return;
      var other = MAPS[key === 'so2sat' ? 'wudapt' : 'so2sat'];
      if (!other) return;
      syncing = true;
      other.map.jumpTo({ center: map.getCenter(), zoom: map.getZoom(),
                         bearing: map.getBearing(), pitch: map.getPitch() });
      syncing = false;
    });
    MAPS[key] = { map: map, overlay: overlay, el: el };
    return MAPS[key];
  }

  function setBasemap(which) {
    UI.basemap = which;
    Object.keys(MAPS).forEach(function (k) {
      var map = MAPS[k].map;
      var apply = function () {
        map.setLayoutProperty('grey', 'visibility', which === 'map' ? 'visible' : 'none');
        map.setLayoutProperty('esri', 'visibility', which === 'satellite' ? 'visible' : 'none');
      };
      if (map.isStyleLoaded()) apply(); else map.once('load', apply);
    });
    document.querySelectorAll('[data-basemap]').forEach(function (b) {
      b.setAttribute('aria-pressed', b.dataset.basemap === which ? 'true' : 'false');
    });
  }

  function setOverlay(name, on) {
    UI.overlays[name] = on;
    if (name === 'guppd') {
      if (on) loadGuppd();
      Object.keys(MAPS).forEach(renderLayers);
    } else {
      Object.keys(MAPS).forEach(function (k) {
        var map = MAPS[k].map;
        var apply = function () {
          if (map.getLayer('demuzere')) {
            map.setLayoutProperty('demuzere', 'visibility', on ? 'visible' : 'none');
          }
        };
        if (map.isStyleLoaded()) apply(); else map.once('load', apply);
      });
    }
  }

  function setDemuzereOpacity(v) {
    UI.demuzereOpacity = v;
    Object.keys(MAPS).forEach(function (k) {
      var map = MAPS[k].map;
      if (map.getLayer('demuzere')) map.setPaintProperty('demuzere', 'raster-opacity', v);
    });
  }

  function setMode(mode) {
    var prev = UI.mode === 'both' ? UI.panel : UI.mode;
    UI.mode = mode;
    var shown = mode === 'both' ? ['so2sat', 'wudapt'] : [mode];
    document.body.dataset.mode = mode;
    document.querySelectorAll('button[data-mode]').forEach(function (b) {
      b.setAttribute('aria-pressed', b.dataset.mode === mode ? 'true' : 'false');
    });
    var from = MAPS[prev] ? MAPS[prev].map : null;
    var cam = from ? { center: from.getCenter(), zoom: from.getZoom() } : null;
    shown.forEach(function (key) {
      var mp = MAPS[key] || makeMap(key);
      mp.el.hidden = false;
      mp.map.resize();
      if (cam && key !== prev) mp.map.jumpTo(cam);
      loadDataset(key).then(function (ds) {
        buildPanel(ds);
        renderLayers(key);
      }).catch(function (e) { setStatus(key, 'failed to load: ' + e.message); });
    });
    Object.keys(MAPS).forEach(function (k) {
      if (shown.indexOf(k) < 0) MAPS[k].el.hidden = true;
      else MAPS[k].map.resize();
    });
    setPanel(mode === 'both' ? (shown.indexOf(UI.panel) >= 0 ? UI.panel : 'so2sat') : mode);
    try { localStorage.setItem('labelglobe-mode', mode); } catch (e) { /* ignore */ }
  }

  function setPanel(key) {
    UI.panel = key;
    document.querySelectorAll('.panel').forEach(function (p) {
      p.hidden = p.dataset.key !== key;
    });
    document.querySelectorAll('[data-panel]').forEach(function (b) {
      b.setAttribute('aria-selected', b.dataset.panel === key ? 'true' : 'false');
    });
  }

  function setStatus(key, text) {
    var el = $('#status-' + key);
    if (el) el.textContent = text;
  }

  // ── Filter panel ──────────────────────────────────────────────────────────
  var pending = {};
  /* Editing a filter no longer redraws the map.
   *
   * Re-evaluating 630k rings and rebuilding the deck layers is a synchronous
   * burst, so filtering live meant the page froze on every checkbox with
   * nothing on screen to say whether it was working or had died. Edits now
   * collect in ds.state and the map only moves when Apply is pressed, which
   * also gives one honest place to show "working". */
  function markDirty(ds) {
    ds.dirty = true;
    paintApply(ds);
  }

  function paintApply(ds) {
    var p = $('.panel[data-key="' + ds.key + '"]');
    if (!p) return;
    var btn = $('.apply', p), note = $('.pending-note', p);
    if (btn) {
      btn.disabled = ds.busy || !ds.dirty;
      btn.textContent = ds.busy ? 'Filtering…' : 'Apply';
      btn.classList.toggle('dirty', !!ds.dirty && !ds.busy);
    }
    if (note) note.classList.toggle('on', !!ds.dirty && !ds.busy);
    p.classList.toggle('busy', !!ds.busy);
  }

  function applyFilters(ds) {
    if (ds.busy) return;
    ds.busy = true;
    paintApply(ds);
    // Two hops before the work: rAF gets us to a paint, the timeout lets that
    // paint actually land. Calling evaluate() straight away would block the
    // thread before "Filtering…" was ever drawn, which is the whole problem.
    requestAnimationFrame(function () {
      setTimeout(function () {
        try {
          ds.result = Core.evaluate(ds, ds.state);
          ds.version++;
          renderLayers(ds.key);
          updateCounts(ds);
          ds.dirty = false;
        } finally {
          ds.busy = false;
          paintApply(ds);
        }
      }, 0);
    });
  }

  function updateCounts(ds) {
    var p = $('.panel[data-key="' + ds.key + '"]');
    if (!p) return;
    $('.shown', p).textContent = fmt(ds.result.shown) + ' of ' + fmt(ds.n) + ' labels shown';
    ds.manifest.filters.forEach(function (name, k) {
      var col = ds.manifest.columns[name];
      if (col.kind === 'range') return;
      var counts = Core.levelCounts(ds, ds.result.fails, k);
      var box = $('[data-filter="' + name + '"]', p);
      box._counts = counts;
      box._render();
    });
  }

  function buildPanel(ds) {
    var host = $('.panel[data-key="' + ds.key + '"] .filters');
    if (host.dataset.built) return updateCounts(ds);
    host.dataset.built = '1';
    ds.manifest.filters.forEach(function (name) {
      var col = ds.manifest.columns[name];
      host.appendChild(col.kind === 'range' ? rangeFilter(ds, name, col)
        : listFilter(ds, name, col));
    });
    updateCounts(ds);
  }

  function listFilter(ds, name, col) {
    var st = ds.state[name], isLcz = col.kind === 'lcz';
    var list = h('ul', { class: isLcz ? 'levels lcz' : 'levels' });
    var search = col.search ? h('input', { type: 'search', placeholder: 'Search ' +
      col.levels.length + ' ' + col.label.toLowerCase().split(' ')[0] + ' values…' }) : null;
    var box = h('details', { class: 'filter', 'data-filter': name },
      [h('summary', {}, [h('span', { text: col.label }), h('span', { class: 'active' })]),
       h('div', { class: 'bulk' }, [
         h('button', { type: 'button', text: 'All', onclick: function () { setAll(1); } }),
         h('button', { type: 'button', text: 'None', onclick: function () { setAll(0); } }),
       ]), search, list]);
    if (isLcz || col.levels.length <= 8) box.open = true;
    if (search) search.addEventListener('input', function () { box._render(); });

    function setAll(v) {
      if (search && search.value) {       // only what the search is showing
        visible().forEach(function (j) { st.sel[j] = v; });
      } else {
        for (var j = 0; j < st.sel.length; j++) st.sel[j] = v;
      }
      box._render();            // the ticks are the state; show it immediately
      markDirty(ds);
    }
    function visible() {
      var q = search ? search.value.trim().toLowerCase() : '';
      var idx = [];
      col.levels.forEach(function (lv, j) {
        var label = isLcz ? lv.name : lv;
        if (!q || String(label).toLowerCase().indexOf(q) >= 0) idx.push(j);
      });
      if (col.search && box._counts) {      // long lists: biggest first
        idx.sort(function (a, b) { return box._counts[b] - box._counts[a]; });
      }
      return idx;
    }
    box._render = function () {
      var counts = box._counts || [], idx = visible(), cap = col.search ? 200 : idx.length;
      list.textContent = '';
      idx.slice(0, cap).forEach(function (j) {
        var lv = col.levels[j];
        var cb = h('input', { type: 'checkbox' });
        cb.checked = !!st.sel[j];
        cb.addEventListener('change', function () { st.sel[j] = cb.checked ? 1 : 0; markDirty(ds); });
        var label = isLcz ? (lv.code <= 17 ? lv.code + ' ' : '') + lv.name : lv;
        list.appendChild(h('li', { class: counts[j] ? '' : 'empty' }, [
          h('label', {}, [cb,
            isLcz ? h('span', { class: 'swatch', style: 'background:' + lv.colour }) : null,
            h('span', { class: 'name', text: label }),
            h('span', { class: 'n', text: fmt(counts[j] || 0) })])]));
      });
      if (idx.length > cap) {
        list.appendChild(h('li', { class: 'more',
          text: (idx.length - cap) + ' more — type to narrow the list' }));
      }
      var off = 0;
      for (var j = 0; j < st.sel.length; j++) if (!st.sel[j]) off++;
      $('.active', box).textContent = off ? (st.sel.length - off) + ' of ' + st.sel.length : '';
    };
    return box;
  }

  /* One track with two thumbs, flanked by typed boxes.
   *
   * Two stacked <input type=range> used to mean two separate bars and no way to
   * enter an exact number. Here both inputs sit on ONE track: the inputs are
   * transparent and pointer-transparent, only their thumbs take the pointer, and
   * a fill div between the two values draws the selected span. Native inputs are
   * kept rather than a custom drag implementation so the control stays
   * keyboard-operable and announced correctly.
   */
  function rangeFilter(ds, name, col) {
    var st = ds.state[name];
    var step = Core.stepFor(col);
    var loR = h('input', { type: 'range', class: 'thumb lo', min: 0, max: SLIDER_STEPS,
                           value: st.lo, 'aria-label': col.label + ' minimum' });
    var hiR = h('input', { type: 'range', class: 'thumb hi', min: 0, max: SLIDER_STEPS,
                           value: st.hi, 'aria-label': col.label + ' maximum' });
    var fill = h('div', { class: 'fill' });
    var track = h('div', { class: 'track' }, [h('div', { class: 'rail' }), fill, loR, hiR]);

    var loN = h('input', { type: 'number', class: 'num', step: step,
                           'aria-label': col.label + ' minimum, type a value' });
    var hiN = h('input', { type: 'number', class: 'num', step: step,
                           'aria-label': col.label + ' maximum, type a value' });

    var miss = null;
    if (col.n_missing) {
      miss = h('input', { type: 'checkbox' });
      miss.checked = st.missing;
      miss.addEventListener('change', function () { st.missing = miss.checked; apply(); });
    }

    var box = h('details', { class: 'filter range', 'data-filter': name },
      [h('summary', {}, [h('span', { text: col.label }), h('span', { class: 'active' })]),
       h('div', { class: 'range-row' }, [loN, track, hiN]),
       miss ? h('label', { class: 'missing' }, [miss,
         h('span', { text: 'include not recorded (' + fmt(col.n_missing) + ' ' +
                           (col.table === 'subs' ? 'submissions' : 'rows') + ')' })]) : null]);

    /* Redraw every part from st, without firing the handlers that read them. */
    function paint() {
      loR.value = st.lo; hiR.value = st.hi;
      fill.style.left = (100 * st.lo / SLIDER_STEPS) + '%';
      fill.style.width = (100 * (st.hi - st.lo) / SLIDER_STEPS) + '%';
      // An open end filters nothing, so it shows the data bound greyed out
      // rather than a number that looks like a cut.
      // The column's own format, not the raw float: an unrounded threshold
      // renders as 0.000004076830285786492 in a 4.6 em box. Exponential output
      // from '.3g' is still a valid value for type=number.
      loN.value = Core.formatValue(col, Core.thresholdFor(col, st.lo));
      hiN.value = Core.formatValue(col, Core.thresholdFor(col, st.hi));
      loN.classList.toggle('open', st.lo === 0);
      hiN.classList.toggle('open', st.hi === SLIDER_STEPS);
      loN.title = st.lo === 0 ? 'Open end — no lower limit' : '';
      hiN.title = st.hi === SLIDER_STEPS ? 'Open end — no upper limit' : '';
      var on = st.lo > 0 || st.hi < SLIDER_STEPS || !st.missing;
      $('.active', box).textContent = on ? 'on' : '';
    }
    function apply() { paint(); markDirty(ds); }

    function setSteps(a, b) {
      st.lo = Math.max(0, Math.min(SLIDER_STEPS, a));
      st.hi = Math.max(0, Math.min(SLIDER_STEPS, b));
      if (st.lo > st.hi) { var t = st.lo; st.lo = st.hi; st.hi = t; }
      apply();
    }

    loR.addEventListener('input', function () {
      // Push rather than swap while dragging: a thumb dragged past its partner
      // should carry it, not teleport the user to the other handle.
      setSteps(+loR.value, Math.max(+loR.value, st.hi));
    });
    hiR.addEventListener('input', function () {
      setSteps(Math.min(+hiR.value, st.lo), +hiR.value);
    });

    function commit(input, isLo) {
      var raw = input.value.trim();
      if (raw === '') { paint(); return; }           // cleared: put the value back
      var v = Number(raw);
      if (!(v === v)) { paint(); return; }           // unparseable: same
      var t = Core.valueToSlider(col, v);
      if (isLo) setSteps(t, Math.max(t, st.hi)); else setSteps(Math.min(t, st.lo), t);
    }
    // `change`, not `input`: committing per keystroke makes "2016" filter on 2
    // and then 20 on the way through.
    loN.addEventListener('change', function () { commit(loN, true); });
    hiN.addEventListener('change', function () { commit(hiN, false); });
    [loN, hiN].forEach(function (el) {
      el.addEventListener('keydown', function (e) { if (e.key === 'Enter') el.blur(); });
    });

    /* Whichever thumb is nearer the pointer goes on top. Without this the two
     * thumbs become unreachable once they coincide, because the one painted
     * last always wins the hit test. */
    track.addEventListener('pointerdown', function (e) {
      var r = track.getBoundingClientRect();
      if (!r.width) return;
      var t = (e.clientX - r.left) / r.width * SLIDER_STEPS;
      var dLo = Math.abs(t - st.lo), dHi = Math.abs(t - st.hi);
      // Equidistant means the thumbs coincide. Then the side the pointer is on
      // decides, otherwise the low thumb would always win and the high one
      // could never be picked up again.
      var nearLo = dLo === dHi ? t < st.lo : dLo < dHi;
      loR.style.zIndex = nearLo ? 4 : 3;
      hiR.style.zIndex = nearLo ? 3 : 4;
    });

    paint();
    return box;
  }

  function resetFilters(key) {
    var ds = DS[key];
    if (!ds) return;
    ds.manifest.filters.forEach(function (f) {
      ds.state[f] = Core.initialState(ds.manifest.columns[f]);
    });
    var host = $('.panel[data-key="' + key + '"] .filters');
    host.textContent = '';
    delete host.dataset.built;
    buildPanel(ds);
    ds.dirty = true;
    applyFilters(ds);
  }

  // ── Boot ──────────────────────────────────────────────────────────────────
  /* Wired before the data arrives, and guarded on MANIFEST rather than only
   * registered once it has. A control that exists in the markup should either
   * work or say why, never sit there inert. */
  function wireChrome() {
    function on(sel, ev, fn) {
      document.querySelectorAll(sel).forEach(function (el) {
        el.addEventListener(ev, function (e) {
          if (!MANIFEST) {
            showError('Still loading data/manifest.json — the filters are not live yet.');
            return;
          }
          fn(el, e);
        });
      });
    }
    // `button[...]`, not `[data-mode]`: <body data-mode> carries the current mode
    // for CSS, so the bare selector also bound this to the body. Every click in
    // the page then re-ran setMode -> buildPanel -> _render, which wiped the
    // checkbox list while the click that caused it was still being dispatched.
    // The input ended up detached before its activation behaviour ran, so it
    // never fired `change` and the tick silently reverted.
    on('button[data-mode]', 'click', function (b) { setMode(b.dataset.mode); });
    on('[data-basemap]', 'click', function (b) { setBasemap(b.dataset.basemap); });
    on('[data-panel]', 'click', function (b) { setPanel(b.dataset.panel); });
    on('[data-reset]', 'click', function (b) { resetFilters(b.dataset.reset); });
    on('[data-apply]', 'click', function (b) {
      var ds = DS[b.dataset.apply];
      if (ds) applyFilters(ds);
    });
    document.querySelectorAll('[data-overlay]').forEach(function (cb) {
      cb.addEventListener('click', function (e) { e.stopPropagation(); });
    });
    on('[data-overlay]', 'change', function (cb) { setOverlay(cb.dataset.overlay, cb.checked); });
    on('[data-guppd-level]', 'change', function (cb) {
      loadGuppd().then(function () {
        GUPPD.levelOn[+cb.dataset.guppdLevel] = cb.checked ? 1 : 0;
        GUPPD.version++;
        Object.keys(MAPS).forEach(renderLayers);
      });
    });
    on('#demuzere-opacity', 'input', function (op) { setDemuzereOpacity(op.value / 100); });
    var tp = $('#toggle-panel');            // pure markup, works with no data
    if (tp) tp.addEventListener('click', function () {
      document.body.classList.toggle('panel-open');
    });
  }

  function boot() {
    wireChrome();
    fetch('data/manifest.json').then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(function (m) {
      MANIFEST = m;
      if (root.pmtiles && MANIFEST.overlays.demuzere && MANIFEST.overlays.demuzere.url) {
        maplibregl.addProtocol('pmtiles', new pmtiles.Protocol().tile);
      }
      var mode = 'so2sat';
      try { mode = localStorage.getItem('labelglobe-mode') || mode; } catch (e) { /* */ }
      setMode(MANIFEST.datasets[mode] || mode === 'both' ? mode : 'so2sat');
      setBasemap('map');
    }).catch(function (e) {
      showError('Could not load data/manifest.json (' + e.message + '). ' +
                'The filters below will not do anything until it loads — ' +
                'if you opened this page as a file, serve the folder over HTTP.');
      $('#maps').textContent = 'Could not load data/manifest.json: ' + e.message;
    });
  }
  // Exposed so the offline suite can build and drive a filter widget without a
  // browser (tests/test_label_site_client_js.py); nothing in the page uses it.
  root.LabelUI = {
    rangeFilter: rangeFilter,
    listFilter: listFilter,
    markDirty: function (ds) { markDirty(ds); },
    wireChrome: wireChrome,
    showError: showError,
    applyFilters: function (ds) { applyFilters(ds); },
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})(typeof globalThis !== 'undefined' ? globalThis : this);
