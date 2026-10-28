"""Offline tests for the label-globe site builder (src/build_label_site.py).

Everything runs on tiny fixture GeoPackages written into tmp_path. What is
pinned down: WUDAPT's personal-data columns never reach the output, the
quantised rings decode back to the source geometry, the release's messy
fields (T/F flags, year 2323, class 18) are normalised as documented, and the
zip carries every file the manifest names.
"""

from __future__ import annotations

import hashlib

import json
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_label_site as L  # noqa: E402

SECRET = "ZZ_PERSONAL_SENTINEL"


def _square(x, y, d=0.003):
    return box(x, y, x + d, y + d)


def build_fixture_site(root: Path) -> tuple[Path, dict]:
    """Two So2Sat cities (one of them the merged Guangzhou) and 5 WUDAPT rows."""
    bounds = pd.DataFrame({
        "SMOD_ID": ["a", "b"], "JRC_NAME_MAIN": ["Nairobi", "Guangzhou"],
        "minx": [36.6, 113.0], "miny": [-1.5, 22.3], "maxx": [37.1, 114.4], "maxy": [-1.1, 23.4],
    })
    bounds_csv = root / "bounds.csv"
    bounds.to_csv(bounds_csv, index=False)

    so2sat = gpd.GeoDataFrame({
        "patch_id": [1, 2, 3, 4],
        "dataset": ["training", "testing", "training", "validation"],
        "LCZ_class": [6, 17, 1, 14],
    }, geometry=[_square(36.8, -1.3), _square(36.9, -1.25),
                 _square(114.0, 22.55), _square(113.25, 23.1)], crs="EPSG:4326")
    so2sat_gpkg = root / "so2sat.gpkg"
    so2sat.to_file(so2sat_gpkg, driver="GPKG")

    ring = [(10.0, 50.0, 0.0), (10.02, 50.0, 0.0), (10.02, 50.01, 0.0),
            (10.01, 50.015, 0.0), (10.0, 50.01, 0.0)]
    wudapt = gpd.GeoDataFrame({
        "submission_id": ["s1", "s1", "s2", "s2", "s3"],
        "submission_date": pd.to_datetime(["2021-05-01"] * 2 + ["2023-01-02"] * 2 +
                                          ["2024-03-03"], utc=True),
        "city": ["Bonn", "Bonn", "Lagos", "Lagos", "Lima"],
        "reference": [SECRET] * 5, "remarks": [SECRET] * 5,
        "representative_date": ["2019-06-01"] * 2 + ["2323-01-01"] * 2 + ["2020-01-01"],
        "firstname": [SECRET] * 5, "lastname": [SECRET] * 5,
        "license": ["CC BY-SA"] * 4 + ["CC BY-NC-SA 4.0"],
        "cite_as": [SECRET] * 5,
        "version": ["1.0.0", "1.0.0", "2.1.5", "2.1.5", "1.10.0"],
        "class": [2, 11, 18, 6, 17],
        "area": [0.5, 1.0, 0.01, 2.0, 0.2],
        "qc_step1": ["True", "F", "T", "False", "True"],
        "qc_step2": ["True"] * 5, "qc_step3": ["False"] * 5,
        "oa": [0.7, 0.7, 0.5, 0.5, 0.9], "oau": [0.6, 0.6, np.nan, np.nan, 0.8],
        "oabu": [0.9] * 5, "oaw": [0.9] * 5,
    }, geometry=[Polygon([(x + i, y, z) for x, y, z in ring]) for i in range(5)],
        crs="EPSG:4326")
    wudapt_gpkg = root / "LCZ-Generator_training_areas_fixture.gpkg"
    wudapt.to_file(wudapt_gpkg, layer=wudapt_gpkg.stem, driver="GPKG")

    guppd = gpd.GeoDataFrame({
        "SMOD_LEVEL": ["30", "23"], "CNTRY_NAME": ["Kenya", "Peru"],
        "JRC_NAME_MAIN": ["Nairobi", "N/A"], "AREA_SQKM": [700.0, 3.0],
        L.GUPPD_POP: [4.4e6, 1200.0],
    }, geometry=[MultiPolygon([_square(36.7, -1.4, 0.2), _square(37.0, -1.4, 0.05)]),
                 _square(-77.0, -12.0, 0.02)], crs="EPSG:4326")
    guppd_gpkg = root / "guppd.gpkg"
    guppd.to_file(guppd_gpkg, layer=L.GUPPD_LAYER, driver="GPKG")

    out = root / "site"
    manifest = L.build_site(out, so2sat_gpkg, wudapt_gpkg, bounds_csv, guppd_gpkg,
                            "https://example.org/demuzere.pmtiles")
    return out, manifest


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    return build_fixture_site(tmp_path_factory.mktemp("label_site"))


def _decode(site_dir: Path, block: dict, n: int) -> dict:
    buf = (site_dir / block["file"]).read_bytes()
    out, off = {}, 0
    for f in block["fields"]:
        dt = np.dtype("<" + f["dtype"])
        out[f["name"]] = np.frombuffer(buf, dt, count=n, offset=off)
        off += n * dt.itemsize
    assert off == len(buf)
    return out


# ── Personal data ────────────────────────────────────────────────────────────
def test_personal_columns_never_reach_the_output(site):
    out, manifest = site
    for p in out.rglob("*"):
        if p.is_file():
            assert SECRET.encode() not in p.read_bytes(), p
    text = json.dumps(manifest)
    for col in L.EXCLUDED_COLUMNS:
        assert f'"{col}"' not in text


def test_whitelist_excludes_personal_columns():
    assert not set(L.WUDAPT_COLUMNS) & set(L.EXCLUDED_COLUMNS)


# ── Geometry ─────────────────────────────────────────────────────────────────
def test_rings_round_trip_within_quantisation_error():
    geoms = np.array([_square(-170.0, 60.0, 0.5), Polygon([(0, 0), (2, 0), (1, 1.5)])])
    frame, counts, q = L.quantize_rings(geoms)
    assert counts.tolist() == [4, 3]                       # closing vertex dropped
    starts = np.r_[0, np.cumsum(counts)]
    for i, g in enumerate(geoms):
        v = q[starts[i]:starts[i + 1]].astype(float)
        xy = np.c_[frame.x0[i] + v[:, 0] * frame.sx[i], frame.y0[i] + v[:, 1] * frame.sy[i]]
        rebuilt = Polygon(xy)
        assert rebuilt.exterior.is_ccw
        assert shapely.hausdorff_distance(rebuilt, g) < 2 * max(frame.sx[i], frame.sy[i])


def test_rings_file_layout(site):
    out, manifest = site
    for ds in manifest["datasets"].values():
        buf = (out / ds["rings"]["file"]).read_bytes()
        n, r = ds["n"], ds["rings"]
        assert r["vertex_offset"] % 4 == 0 and r["vertex_offset"] >= 2 * n
        assert len(buf) == r["vertex_offset"] + 4 * r["n_vertices"]
        counts = np.frombuffer(buf, "<u2", count=n)
        assert counts.sum() == r["n_vertices"]


# ── So2Sat ───────────────────────────────────────────────────────────────────
def test_so2sat_columns_and_guangzhou_split(site):
    out, manifest = site
    ds = manifest["datasets"]["so2sat"]
    rows = _decode(out, ds["rows"], ds["n"])
    cols = ds["columns"]
    city = [cols["city"]["levels"][c] for c in rows["city"]]
    # training patches of the merged GUPPD polygon are Shenzhen
    assert city == ["Nairobi", "Nairobi", "Shenzhen", "Guangzhou"]
    assert [cols["split"]["levels"][c] for c in rows["split"]] == \
        ["Training", "Testing", "Training", "Validation"]
    assert rows["lcz"].tolist() == [6, 17, 1, 14]
    assert rows["patch_id"].tolist() == [1, 2, 3, 4]
    assert ds["filters"] == ["lcz", "split", "city"]


# ── WUDAPT ───────────────────────────────────────────────────────────────────
def test_wudapt_rows_and_submissions(site):
    out, manifest = site
    ds = manifest["datasets"]["wudapt"]
    cols = ds["columns"]
    rows = _decode(out, ds["rows"], ds["n"])
    subs = _decode(out, ds["subs"], ds["subs"]["n"])
    assert ds["subs"]["n"] == 3
    assert rows["sub"].tolist() == [0, 0, 1, 1, 2]
    # T/F and True/False normalise to the same two levels
    assert [cols["qc1"]["levels"][c] for c in rows["qc1"]] == \
        ["Pass", "Fail", "Pass", "Fail", "Pass"]
    # year 2323 is not a year
    assert subs["rep_year"].tolist() == [2019, L.U2_MISSING, 2020]
    assert cols["rep_year"]["n_missing"] == 1
    assert subs["sub_year"].tolist() == [2021, 2023, 2024]
    assert np.isnan(subs["oau"][1]) and cols["oau"]["n_missing"] == 1
    assert [cols["city"]["levels"][c] for c in subs["city"]] == ["Bonn", "Lagos", "Lima"]
    # versions sort numerically, not as strings
    assert cols["version"]["levels"] == ["1.0.0", "1.10.0", "2.1.5"]
    codes = [lv["code"] for lv in cols["lcz"]["levels"]]
    assert 18 in codes
    assert cols["lcz"]["levels"][codes.index(18)]["name"].startswith("Undocumented")


# ── Overlays ─────────────────────────────────────────────────────────────────
def test_guppd_multipolygons_explode_to_rows(site):
    out, manifest = site
    g = manifest["overlays"]["guppd"]
    assert g["n"] == 3 and g["n_features"] == 2
    rows = _decode(out, g["rows"], g["n"])
    assert rows["feat"].tolist() == [0, 0, 1]
    levels = g["columns"]["level"]["levels"]
    assert [levels[c] for c in rows["level"]] == \
        ["Urban centre", "Urban centre", "Semi-dense urban cluster"]
    assert rows["pop"][2] == pytest.approx(1200.0)
    names = (out / g["names"]).read_text(encoding="utf-8").split("\n")
    assert names == ["Nairobi", "N/A"]


def test_demuzere_is_a_url_not_a_payload(site):
    out, manifest = site
    d = manifest["overlays"]["demuzere"]
    assert d["url"] == "https://example.org/demuzere.pmtiles"
    assert 'data-overlay="demuzere"' in (out / "index.html").read_text()
    assert not list(out.rglob("*.pmtiles"))


def test_zip_has_every_manifest_file_at_the_root(site, tmp_path):
    out, manifest = site
    dest = tmp_path / "label_site.zip"
    L.write_zip(out, dest)
    with zipfile.ZipFile(dest) as z:
        names = set(z.namelist())
    assert L.manifest_files(manifest) <= names
    assert "netlify.toml" in names
    assert not any(n.startswith("site/") for n in names)


def test_asset_urls_are_versioned_by_content(site):
    """A fresh index.html paired with a cached app.js is silently broken: the
    markup has an Apply button the old script never wires up, so every filter
    looks dead. The URL has to change when the bytes do."""
    import re
    out, _ = site
    html = (out / "index.html").read_text()
    refs = re.findall(r'assets/app\.(?:js|css)\?v=([0-9a-f]{12})', html)
    assert len(refs) == 2, f"both assets must be versioned, found {refs}"
    assert len(set(refs)) == 1, "one build, one version"
    expect = hashlib.sha256(
        b"".join((out / "assets" / f).read_bytes() for f in ("app.js", "app.css"))
    ).hexdigest()[:12]
    assert refs[0] == expect, "the version must be the hash of the served bytes"


def test_editing_an_asset_would_change_the_version(site):
    """The property that matters: the hash is over the asset bytes, so any edit
    to app.js lands on a different URL."""
    out, _ = site
    base = (out / "assets" / "app.js").read_bytes()
    before = hashlib.sha256(base + (out / "assets" / "app.css").read_bytes()).hexdigest()[:12]
    after = hashlib.sha256(base + b"\n// edit\n"
                           + (out / "assets" / "app.css").read_bytes()).hexdigest()[:12]
    assert before != after
