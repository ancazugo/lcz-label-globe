"""Run LabelCore (src/label_site/app.js) in QuickJS against a fixture payload.

There is no browser or node on this host. The browser half of app.js returns
early when there is no `document`, so the pure core -- block decoding, ring
decoding, the filter mask and the cross-filter level counts -- can be checked
against numpy on the same bytes the builder wrote.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

quickjs = pytest.importorskip("quickjs")

from test_build_label_site import _decode, build_fixture_site  # noqa: E402

APP_JS = Path(__file__).resolve().parents[1] / "src" / "label_site" / "app.js"

SHIM = r"""
function atobBytes(b64) {
  var chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  var n = b64.length, pad = b64.endsWith('==') ? 2 : (b64.endsWith('=') ? 1 : 0);
  var out = new Uint8Array((n / 4) * 3 - pad), p = 0;
  for (var i = 0; i < n; i += 4) {
    var a = chars.indexOf(b64[i]), b = chars.indexOf(b64[i + 1]);
    var c = chars.indexOf(b64[i + 2]), d = chars.indexOf(b64[i + 3]);
    out[p++] = (a << 2) | (b >> 4);
    if (p < out.length) out[p++] = ((b & 15) << 4) | (c >> 2);
    if (p < out.length) out[p++] = ((c & 3) << 6) | d;
  }
  return out.buffer;
}
function loadDs(m, rowsB64, subsB64) {
  var ds = { manifest: m, n: m.n };
  ds.rows = LabelCore.decodeBlock(atobBytes(rowsB64), m.rows.fields, m.n);
  if (m.subs) ds.subs = LabelCore.decodeBlock(atobBytes(subsB64), m.subs.fields, m.subs.n);
  return ds;
}
function initial(m) {
  var st = {};
  m.filters.forEach(function (f) { st[f] = LabelCore.initialState(m.columns[f]); });
  return st;
}
"""


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    out, manifest = build_fixture_site(tmp_path_factory.mktemp("label_js"))
    c = quickjs.Context()
    c.eval(APP_JS.read_text())
    c.eval(SHIM)
    assert c.eval("typeof LabelCore") == "object"
    return c, out, manifest


def _b64(out: Path, block) -> str:
    return base64.b64encode((out / block["file"]).read_bytes()).decode() if block else ""


def _load(c, out, m, name):
    ds = m["datasets"][name]
    c.eval(f"var M = {json.dumps(ds)};")
    c.eval(f"var DS = loadDs(M, '{_b64(out, ds['rows'])}', '{_b64(out, ds['subs'])}');")
    return ds


def test_unfiltered_shows_everything(ctx):
    c, out, m = ctx
    for name in ("so2sat", "wudapt"):
        ds = _load(c, out, m, name)
        assert c.eval("LabelCore.evaluate(DS, initial(M)).shown") == ds["n"]


def test_categorical_and_submission_level_filters(ctx):
    c, out, m = ctx
    ds = _load(c, out, m, "wudapt")
    # QC step 1 = Pass only, and OA (a per-submission column) above slider step 600
    c.eval("var st = initial(M); st.qc1.sel[0] = 0; st.oa.lo = 600;"
           "var r = LabelCore.evaluate(DS, st);")
    mask = json.loads(c.eval("JSON.stringify(Array.from(r.mask))"))
    rows = _decode(out, ds["rows"], ds["n"])
    subs = _decode(out, ds["subs"], ds["subs"]["n"])
    thr = c.eval("LabelCore.sliderToValue(M.columns.oa, 600)")
    assert ds["columns"]["oa"]["min"] < thr < ds["columns"]["oa"]["max"]
    oa_ok = subs["oa"][rows["sub"]] >= thr
    expect = (rows["qc1"] == 1) & oa_ok
    assert 0 < expect.sum() < ds["n"]
    assert mask == expect.astype(int).tolist()

    # counts for qc1 ignore qc1's own filter but honour the OA one
    k = ds["filters"].index("qc1")
    counts = json.loads(c.eval(f"JSON.stringify(Array.from(LabelCore.levelCounts(DS, r.fails, {k})))"))
    assert counts == [float(((rows["qc1"] == v) & oa_ok).sum()) for v in (0, 1)]


def test_range_missing_values_follow_the_checkbox(ctx):
    c, out, m = ctx
    ds = _load(c, out, m, "wudapt")
    rows = _decode(out, ds["rows"], ds["n"])
    subs = _decode(out, ds["subs"], ds["subs"]["n"])
    missing = np.isnan(subs["oau"][rows["sub"]])
    c.eval("var st = initial(M); st.oau.missing = false;")
    assert c.eval("LabelCore.evaluate(DS, st).shown") == int((~missing).sum())


def test_lcz_filter_uses_codes_not_positions(ctx):
    c, out, m = ctx
    ds = _load(c, out, m, "so2sat")
    rows = _decode(out, ds["rows"], ds["n"])
    levels = [lv["code"] for lv in ds["columns"]["lcz"]["levels"]]
    j = levels.index(17)
    c.eval(f"var st = initial(M); for (var i = 0; i < st.lcz.sel.length; i++) st.lcz.sel[i] = 0;"
           f"st.lcz.sel[{j}] = 1;")
    assert c.eval("LabelCore.evaluate(DS, st).shown") == int((rows["lcz"] == 17).sum())


def test_rings_decode_closed_and_in_place(ctx):
    c, out, m = ctx
    ds = _load(c, out, m, "so2sat")
    b64 = _b64(out, ds["rings"])
    c.eval(f"var R = LabelCore.decodeRings(atobBytes('{b64}'), M.rings, DS.rows, M.n);")
    starts = json.loads(c.eval("JSON.stringify(Array.from(R.startIndices))"))
    pos = np.array(json.loads(c.eval("JSON.stringify(Array.from(R.positions))"))).reshape(-1, 2)
    assert starts == [0, 5, 10, 15, 20]
    first = pos[0:5]
    assert np.allclose(first[0], first[-1])                # closed again
    assert np.allclose(first.min(0), [36.8, -1.3], atol=1e-5)
    assert np.allclose(first.max(0), [36.803, -1.297], atol=1e-5)


# ── Range filters: the slider/value map ──────────────────────────────────────
YEAR = {"label": "Representative year", "kind": "range", "table": "subs",
        "dtype": "u2", "min": 1990.0, "max": 2024.0, "scale": "linear",
        "format": "d", "n_missing": 3}
OA = {"label": "OA", "kind": "range", "table": "subs", "dtype": "f4",
      "min": 0.0, "max": 1.0, "scale": "linear", "format": ".2f", "n_missing": 0}
AREA = {"label": "Area", "kind": "range", "table": "rows", "dtype": "f4",
        "min": 0.001, "max": 1000.0, "scale": "log", "format": ".3g",
        "n_missing": 0}


@pytest.fixture(scope="module")
def core():
    ctx = quickjs.Context()
    ctx.eval(APP_JS.read_text())
    for name, col in (("YEAR", YEAR), ("OA", OA), ("AREA", AREA)):
        ctx.eval(f"var {name} = {json.dumps(col)};")
    return ctx


def test_value_to_slider_inverts_slider_to_value(core):
    for col in ("YEAR", "OA", "AREA"):
        worst = core.eval(f"""(function(){{
          var w = 0;
          for (var t = 0; t <= 1000; t += 7) {{
            var back = LabelCore.valueToSlider({col}, LabelCore.sliderToValue({col}, t));
            w = Math.max(w, Math.abs(back - t));
          }}
          return w;
        }})()""")
        assert worst <= 1, f"{col} round-trips {worst} steps off"


def test_value_to_slider_clamps_outside_the_data_range(core):
    assert core.eval("LabelCore.valueToSlider(YEAR, 1500)") == 0
    assert core.eval("LabelCore.valueToSlider(YEAR, 2999)") == 1000
    assert core.eval("LabelCore.valueToSlider(AREA, -5)") == 0, "log of a negative"
    assert core.eval("LabelCore.valueToSlider(OA, NaN)") == 0, "unparseable box"


def test_an_integer_column_filters_at_the_number_it_displays(core):
    """The bug this replaced: the readout said 2016 and the filter used 2016.34,
    so every 2016 submission the label promised was dropped."""
    got = core.eval("""(function(){
      var bad = [];
      for (var t = 1; t < 1000; t++) {
        var shown = LabelCore.formatValue(YEAR, LabelCore.thresholdFor(YEAR, t));
        var used = LabelCore.thresholdFor(YEAR, t);
        if (String(used) !== shown) bad.push(t);
      }
      return bad.length;
    })()""")
    assert got == 0, f"{got} steps where the shown and applied thresholds differ"
    assert core.eval("LabelCore.thresholdFor(YEAR, 765) % 1") == 0
    # A continuous column must NOT be rounded.
    assert core.eval("LabelCore.thresholdFor(OA, 333) % 1") != 0


def test_typing_a_year_keeps_that_year_in_range(core):
    kept = core.eval("""(function(){
      var t = LabelCore.valueToSlider(YEAR, 2016);
      return LabelCore.thresholdFor(YEAR, t) <= 2016;
    })()""")
    assert kept is True, "typing 2016 as a minimum must not exclude 2016"


def test_step_matches_the_column_format(core):
    assert core.eval("LabelCore.stepFor(YEAR)") == "1"
    assert core.eval("LabelCore.stepFor(OA)") == "0.01"
    assert core.eval("LabelCore.stepFor(AREA)") == "any"


# ── Range filters: the widget ────────────────────────────────────────────────
# A DOM small enough to build the control and drive it. `refilter` defers behind
# requestAnimationFrame, so a no-op there makes it inert and the widget can be
# exercised without maps, deck.gl or a render pass.
DOM_SHIM = r"""
function El(tag) {
  var self = this;
  this.tagName = tag; this.children = []; this.className = ''; this.textContent = '';
  this.style = {}; this.title = ''; this.type = ''; this.hidden = false;
  // A real input stringifies whatever is assigned, and the widget calls
  // .trim() on it; a stub that kept numbers would hide a TypeError.
  var _v = '';
  Object.defineProperty(this, 'value', {
    get: function () { return _v; },
    set: function (x) { _v = String(x); },
  });
  this.checked = false; this.attrs = {}; this._l = {};
  this.setAttribute = function (k, v) {
    self.attrs[k] = String(v);
    if (k === 'value' || k === 'type' || k === 'step') self[k] = String(v);
  };
  this.getAttribute = function (k) { return k in self.attrs ? self.attrs[k] : null; };
  this.appendChild = function (c) { self.children.push(c); return c; };
  this.insertBefore = function (c, ref) {
    var i = ref ? self.children.indexOf(ref) : -1;
    if (i < 0) self.children.push(c); else self.children.splice(i, 0, c);
    return c;
  };
  Object.defineProperty(this, 'firstChild', {
    get: function () { return self.children[0] || null; },
  });
  this.addEventListener = function (ev, fn) { (self._l[ev] = self._l[ev] || []).push(fn); };
  this.fire = function (ev, d) {
    var e = { target: self, key: (d || {}).key, clientX: (d || {}).clientX };
    (self._l[ev] || []).forEach(function (f) { f(e); });
  };
  this.blur = function () {};
  this.getBoundingClientRect = function () { return { left: 0, width: 100 }; };
  this.classList = {
    toggle: function (c, on) {
      var has = (' ' + self.className + ' ').indexOf(' ' + c + ' ') >= 0;
      if (on && !has) self.className = (self.className + ' ' + c).trim();
      if (!on && has) {
        self.className = (' ' + self.className + ' ').split(' ' + c + ' ').join(' ').trim();
      }
    },
  };
  this.querySelector = function (sel) {
    var want = sel.replace('.', ''), out = null;
    (function walk(n) {
      n.children.forEach(function (c) {
        if (out) return;
        if ((' ' + c.className + ' ').indexOf(' ' + want + ' ') >= 0) { out = c; return; }
        walk(c);
      });
    })(self);
    return out;
  };
}
globalThis.document = {
  readyState: 'loading',                       // keeps boot() from running
  createElement: function (t) { return new El(t); },
  querySelector: function () { return null; },
  addEventListener: function () {},
  body: { classList: { toggle: function () {} } },
};
globalThis.FRAMES = [];
globalThis.requestAnimationFrame = function (fn) { FRAMES.push(fn); };
globalThis.setTimeout = function (fn) { FRAMES.push(fn); };
globalThis.flushFrames = function (n) {
  for (var i = 0; i < (n || 4) && FRAMES.length; i++) FRAMES.shift()();
};
globalThis.maplibregl = {}; globalThis.deck = {};
globalThis.localStorage = { getItem: function () { return null; },
                            setItem: function () {} };

function findAll(root, cls, out) {
  out = out || [];
  root.children.forEach(function (c) {
    if ((' ' + c.className + ' ').indexOf(' ' + cls + ' ') >= 0) out.push(c);
    findAll(c, cls, out);
  });
  return out;
}
function widget(col) {
  var ds = { key: 'wudapt', state: { f: LabelCore.initialState(col) },
             manifest: { columns: { f: col }, filters: ['f'] } };
  var box = LabelUI.rangeFilter(ds, 'f', col);
  var nums = findAll(box, 'num');
  return { ds: ds, box: box, st: ds.state.f,
           track: findAll(box, 'track')[0], fill: findAll(box, 'fill')[0],
           loR: findAll(box, 'lo')[0], hiR: findAll(box, 'hi')[0],
           loN: nums[0], hiN: nums[1] };
}
"""


@pytest.fixture(scope="module")
def ui():
    ctx = quickjs.Context()
    ctx.eval(DOM_SHIM.replace("globalThis.document", "globalThis.document", 1))
    ctx.eval(APP_JS.read_text())
    ctx.eval(DOM_SHIM)                     # re-run: helpers only, document kept
    for name, col in (("YEAR", YEAR), ("OA", OA), ("AREA", AREA)):
        ctx.eval(f"var {name} = {json.dumps(col)};")
    return ctx


def test_one_track_carries_both_thumbs_and_two_typed_boxes(ui):
    got = ui.eval("""(function(){
      var w = widget(YEAR);
      return JSON.stringify({
        tracks: findAll(w.box, 'track').length,
        thumbs: findAll(w.track, 'thumb').length,
        boxes: findAll(w.box, 'num').length,
        boxType: w.loN.type, step: w.loN.getAttribute('step'),
      });
    })()""")
    r = json.loads(got)
    assert r["tracks"] == 1 and r["thumbs"] == 2, r
    assert r["boxes"] == 2 and r["boxType"] == "number"
    assert r["step"] == "1", "a year box should step in whole years"


def test_it_opens_showing_the_full_range_and_filtering_nothing(ui):
    got = json.loads(ui.eval("""(function(){
      var w = widget(YEAR);
      return JSON.stringify({lo: w.loN.value, hi: w.hiN.value, active: w.box.querySelector('.active').textContent,
                             loOpen: w.loN.className.indexOf('open') >= 0,
                             hiOpen: w.hiN.className.indexOf('open') >= 0,
                             width: w.fill.style.width});
    })()"""))
    assert got["lo"] == "1990" and got["hi"] == "2024"
    assert got["active"] == "", "an untouched filter must not read as active"
    assert got["loOpen"] and got["hiOpen"], "both ends start open"
    assert got["width"] == "100%"


def test_typing_a_bound_moves_the_thumb_and_the_fill(ui):
    got = json.loads(ui.eval("""(function(){
      var w = widget(YEAR);
      w.loN.value = '2010'; w.loN.fire('change');
      return JSON.stringify({lo: w.st.lo, hi: w.st.hi, shown: w.loN.value,
                             left: w.fill.style.left,
                             thumb: w.loR.value,
                             active: w.box.querySelector('.active').textContent,
                             open: w.loN.className.indexOf('open') >= 0});
    })()"""))
    assert got["shown"] == "2010", "the box must echo the committed value"
    assert got["thumb"] == str(got["lo"]), "the thumb follows the typed number"
    assert got["left"] != "0%" and got["active"] == "on"
    assert got["open"] is False, "a typed bound is a real cut, not an open end"


def test_a_typed_minimum_above_the_maximum_carries_it_up(ui):
    got = json.loads(ui.eval("""(function(){
      var w = widget(YEAR);
      w.hiN.value = '2000'; w.hiN.fire('change');
      w.loN.value = '2015'; w.loN.fire('change');
      return JSON.stringify({lo: w.st.lo, hi: w.st.hi,
                             loV: w.loN.value, hiV: w.hiN.value});
    })()"""))
    assert got["lo"] <= got["hi"], "the range must never invert"
    assert got["loV"] == "2015" and got["hiV"] == "2015"


def test_dragging_a_thumb_past_its_partner_pushes_rather_than_swaps(ui):
    got = json.loads(ui.eval("""(function(){
      var w = widget(OA);
      w.hiR.value = 400; w.hiR.fire('input');
      w.loR.value = 700; w.loR.fire('input');
      return JSON.stringify({lo: w.st.lo, hi: w.st.hi});
    })()"""))
    assert got["lo"] == 700 and got["hi"] == 700


def test_out_of_range_and_junk_typing_is_handled(ui):
    got = json.loads(ui.eval("""(function(){
      var w = widget(YEAR), out = {};
      w.loN.value = '1200'; w.loN.fire('change'); out.tooLow = w.st.lo;
      w.hiN.value = '9999'; w.hiN.fire('change'); out.tooHigh = w.st.hi;
      w.loN.value = '2005'; w.loN.fire('change');
      w.loN.value = 'abc';  w.loN.fire('change'); out.junk = w.loN.value;
      w.loN.value = '';     w.loN.fire('change'); out.blank = w.loN.value;
      return JSON.stringify(out);
    })()"""))
    assert got["tooLow"] == 0, "below the data clamps to the open end"
    assert got["tooHigh"] == 1000
    assert got["junk"] == "2005", "junk is rejected, the previous value restored"
    assert got["blank"] == "2005", "clearing the box restores rather than wipes"


def test_the_log_column_spaces_its_boxes_logarithmically(ui):
    got = json.loads(ui.eval("""(function(){
      var w = widget(AREA);
      w.loN.value = '1'; w.loN.fire('change');
      return JSON.stringify({step: w.st.lo, back: Number(w.loN.value)});
    })()"""))
    # 0.001 -> 1000 spans six decades and 1 km2 is three of them up, so a log
    # axis puts it at the midpoint; a linear axis would put it at step 1.
    assert got["step"] == 500, got["step"]
    assert abs(got["back"] - 1.0) < 0.02


def test_the_nearer_thumb_is_raised_so_coincident_handles_stay_reachable(ui):
    got = json.loads(ui.eval("""(function(){
      var w = widget(OA);
      w.loN.value = '0.50'; w.loN.fire('change');
      w.hiN.value = '0.50'; w.hiN.fire('change');
      w.track.fire('pointerdown', { clientX: 10 });     // far left, nearer lo
      var a = [w.loR.style.zIndex, w.hiR.style.zIndex];
      w.track.fire('pointerdown', { clientX: 95 });     // far right, nearer hi
      return JSON.stringify({left: a, right: [w.loR.style.zIndex, w.hiR.style.zIndex]});
    })()"""))
    assert got["left"][0] > got["left"][1], "a click left of both raises the low thumb"
    assert got["right"][1] > got["right"][0]


def test_the_boxes_round_to_the_column_format(ui):
    """Raw thresholds render as 0.000004076830285786492 in a 4.6 em box."""
    got = json.loads(ui.eval("""(function(){
      var y = widget(YEAR), o = widget(OA), a = widget(AREA);
      return JSON.stringify({year: y.hiN.value, oa: o.loN.value, area: a.hiN.value});
    })()"""))
    assert got["year"] == "2024", "a year is a whole number"
    assert got["oa"] == "0.00", "two decimals for a ratio"
    assert len(got["area"]) <= 8, f"log column should be short, got {got['area']}"
    assert float(got["area"]) == pytest.approx(1000, rel=0.01)


# ── Apply-on-press ───────────────────────────────────────────────────────────
def test_editing_a_filter_marks_it_dirty_without_redrawing(ui):
    """The point of the change: a checkbox must not kick off a 630k-ring
    re-evaluation, because that froze the page with nothing on screen."""
    got = json.loads(ui.eval("""(function(){
      FRAMES.length = 0;
      var w = widget(YEAR);
      var before = { dirty: !!w.ds.dirty, result: w.ds.result || null, frames: FRAMES.length };
      w.loN.value = '2000'; w.loN.fire('change');
      return JSON.stringify({ before: before, dirty: !!w.ds.dirty,
                              result: w.ds.result || null, frames: FRAMES.length,
                              lo: w.ds.state.f.lo });
    })()"""))
    assert got["before"]["dirty"] is False
    assert got["dirty"] is True, "the edit must be recorded as pending"
    assert got["lo"] > 0, "the draft state still updates immediately"
    assert got["result"] is None, "but nothing was re-evaluated"
    assert got["frames"] == 0, "and no frame was scheduled"


def test_dragging_a_thumb_also_only_marks_dirty(ui):
    got = json.loads(ui.eval("""(function(){
      FRAMES.length = 0;
      var w = widget(OA);
      w.hiR.value = 600; w.hiR.fire('input');
      return JSON.stringify({ dirty: !!w.ds.dirty, frames: FRAMES.length, hi: w.ds.state.f.hi });
    })()"""))
    assert got["dirty"] is True and got["hi"] == 600
    assert got["frames"] == 0


def test_apply_defers_the_work_so_the_busy_state_can_paint(ui):
    """Calling evaluate() straight away would block before 'Filtering…' was
    ever drawn, which is the failure mode being fixed."""
    got = json.loads(ui.eval("""(function(){
      FRAMES.length = 0;
      var w = widget(YEAR);
      w.loN.value = '2000'; w.loN.fire('change');
      LabelUI.applyFilters(w.ds);
      return JSON.stringify({ busy: !!w.ds.busy, frames: FRAMES.length,
                              stillDirty: !!w.ds.dirty });
    })()"""))
    assert got["busy"] is True, "it must enter the busy state immediately"
    assert got["frames"] >= 1, "and hand the heavy work to a later frame"
    assert got["stillDirty"] is True, "dirty only clears once the work has run"


def test_apply_is_ignored_while_already_busy(ui):
    got = json.loads(ui.eval("""(function(){
      FRAMES.length = 0;
      var w = widget(YEAR);
      w.loN.value = '2000'; w.loN.fire('change');
      LabelUI.applyFilters(w.ds);
      var after_one = FRAMES.length;
      LabelUI.applyFilters(w.ds);
      return JSON.stringify({ one: after_one, two: FRAMES.length });
    })()"""))
    assert got["two"] == got["one"], "a second press must not queue a second pass"


# ── Wiring survives a failed data load ───────────────────────────────────────
# Every listener used to be registered inside the manifest fetch's .then(), so
# one failed request left a page whose checkboxes rendered, ticked, and drove
# nothing — indistinguishable from a bug in the filters themselves.
CHROME_SHIM = r"""
function mkDoc(els) {
  var side = new El('aside');
  globalThis.document = {
    readyState: 'loading',
    createElement: function (t) { return new El(t); },
    querySelectorAll: function (sel) { return els[sel] || []; },
    querySelector: function (sel) {
      if (sel === '#sidebar') return side;
      if (sel === '#load-error') return side.children[0] || null;
      return (els[sel] || [])[0] || null;
    },
    addEventListener: function () {},
    body: { classList: { toggle: function () {} } },
  };
  document.sidebar = side;
  return document;
}
"""


@pytest.fixture()
def chrome():
    ctx = quickjs.Context()
    ctx.eval(DOM_SHIM)
    ctx.eval(APP_JS.read_text())
    ctx.eval(DOM_SHIM)
    ctx.eval(CHROME_SHIM)
    return ctx


def test_controls_are_wired_before_the_manifest_arrives(chrome):
    got = json.loads(chrome.eval("""(function(){
      var apply = new El('button'); apply.attrs['data-apply'] = 'so2sat';
      apply.dataset = { apply: 'so2sat' };
      var lev = new El('input'); lev.dataset = { guppdLevel: '0' };
      mkDoc({ '[data-apply]': [apply], '[data-guppd-level]': [lev],
              '[data-mode]': [], '[data-basemap]': [], '[data-panel]': [],
              '[data-reset]': [], '[data-overlay]': [], '#demuzere-opacity': [] });
      LabelUI.wireChrome();
      return JSON.stringify({ apply: (apply._l.click || []).length,
                              level: (lev._l.change || []).length });
    })()"""))
    assert got["apply"] == 1, "Apply must be wired without waiting for the data"
    assert got["level"] == 1, "so must the layer checkboxes"


def test_pressing_a_control_with_no_data_says_so_instead_of_going_quiet(chrome):
    got = json.loads(chrome.eval("""(function(){
      var apply = new El('button'); apply.dataset = { apply: 'so2sat' };
      var doc = mkDoc({ '[data-apply]': [apply], '[data-guppd-level]': [],
                        '[data-mode]': [], '[data-basemap]': [], '[data-panel]': [],
                        '[data-reset]': [], '[data-overlay]': [], '#demuzere-opacity': [] });
      LabelUI.wireChrome();
      apply.fire('click');
      var bar = doc.sidebar.children[0];
      return JSON.stringify({ banner: bar ? bar.textContent : null,
                              hidden: bar ? !!bar.hidden : null });
    })()"""))
    assert got["banner"], "a dead control must explain itself"
    assert "manifest" in got["banner"], got["banner"]
    assert got["hidden"] is False


def test_the_banner_is_reused_rather_than_stacked(chrome):
    got = json.loads(chrome.eval("""(function(){
      var doc = mkDoc({});
      LabelUI.showError('first');
      LabelUI.showError('second');
      return JSON.stringify({ n: doc.sidebar.children.length,
                              text: doc.sidebar.children[0].textContent });
    })()"""))
    assert got["n"] == 1, "repeated failures must not pile up banners"
    assert got["text"] == "second"


def test_mode_buttons_are_wired_without_catching_the_body(chrome):
    """<body data-mode> must not be treated as a mode button.

    It was: the bare `[data-mode]` selector bound setMode to the body too, so
    every click in the page re-ran it, and setMode rebuilds the filter list.
    The rebuild landed on the microtask checkpoint that runs the moment the
    body's listener returns -- before the clicked checkbox's activation
    behaviour -- so the input was detached, never fired `change`, and the tick
    silently reverted. All/None survived because they act from a listener that
    has already run by then.
    """
    got = json.loads(chrome.eval("""(function(){
      var btn = new El('button'); btn.dataset = { mode: 'wudapt' };
      var body = new El('body');  body.dataset = { mode: 'so2sat' };
      var seen = { 'button[data-mode]': [btn], '[data-mode]': [btn, body] };
      mkDoc({ 'button[data-mode]': seen['button[data-mode]'],
              '[data-mode]': seen['[data-mode]'],
              '[data-basemap]': [], '[data-panel]': [], '[data-reset]': [],
              '[data-apply]': [], '[data-overlay]': [], '[data-guppd-level]': [],
              '#demuzere-opacity': [] });
      LabelUI.wireChrome();
      return JSON.stringify({ button: (btn._l.click || []).length,
                              body: (body._l.click || []).length });
    })()"""))
    assert got["button"] == 1, "the real mode buttons must still be wired"
    assert got["body"] == 0, "a click anywhere must not re-run setMode"


def test_the_source_never_selects_data_mode_unscoped():
    """Belt and braces: the bug is a selector, so pin the selector."""
    src = APP_JS.read_text()
    assert "querySelectorAll('[data-mode]')" not in src
    assert "on('[data-mode]'" not in src
    assert "button[data-mode]" in src
