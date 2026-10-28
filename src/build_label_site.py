#!/usr/bin/env python
"""Build the static label-globe site: where the So2Sat and WUDAPT labels are.

    source /maps/acz25/envs/eo_fm-env/bin/activate
    export SQLITE_USE_OGR_VFS=YES NUMPY_MADVISE_HUGEPAGE=0
    python src/build_label_site.py --output-dir label_site \
        --zip /scratch/acz25/eo_fm/label_site.zip
    python -m http.server -d label_site 8001

A globe (MapLibre + deck.gl, both from jsDelivr) with every label polygon drawn
in its LCZ colour: 400,673 So2Sat patches and 630,311 WUDAPT training areas.
The sidebar's filters are read from the manifest, so each dataset brings its
own (So2Sat: split / class / city; WUDAPT: class, city, dates, QC steps,
submission accuracies, area, generator version, licence). Both datasets
together are only ever shown as two side-by-side maps with a locked camera.

Two optional layers sit under/over both maps, toggled from the sidebar:
GUPPD settlement outlines (all 123,034 polygons of every SMOD level, fetched on
first toggle; --no-guppd leaves them out) and the Demuzere et al. (2022) global
LCZ map. The latter is GBs of raster tiles, so it is NOT part of this build:
make it with src/build_demuzere_pmtiles.py, host the .pmtiles file somewhere
that answers range requests with CORS (Cloudflare R2), and pass its URL as
--demuzere-url. Without the URL the toggle is shown disabled.

WHY THE GEOMETRY IS QUANTISED AND SPLIT
---------------------------------------
Every polygon is stored as a bbox anchor + per-axis scale (float32, in the
rows block) and its ring as uint16 offsets inside that bbox (the rings block),
with the closing vertex dropped. That halves the float32 size, and it means
the globe view -- one dot per label at the bbox centre -- needs only the rows
block; the rings block is fetched the first time someone zooms in far enough
to see a polygon. Quantisation error is bbox/65535: 5 mm on a 320 m patch,
3 m on the largest (2 degree) WUDAPT area.

WUDAPT fields that are constant within a submission (city, dates, accuracies,
version, licence) ship once per submission in a separate block and are looked
up through the row's `sub` code; qc_step1 is NOT constant within a submission,
so the three QC flags stay per row.

PERSONAL DATA
-------------
WUDAPT is read with an explicit column whitelist (`WUDAPT_COLUMNS`). The
submitter's `firstname`, `lastname` and `cite_as` are never read, and neither
are `remarks` and `reference`, which are free text that can carry names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.blocks import encode_categorical, write_block  # noqa: E402
from utils.constants import lcz_dict  # noqa: E402
from utils.geo_lookup import assign_city  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_label_site")

TEMPLATE_DIR = Path(__file__).resolve().parent / "label_site"
DATA_DIR = Path(os.environ.get("DATA_DIR", "/maps/acz25/phd-thesis-data"))
# The _rxr file, not patches_reference.gpkg: the latter places some patches in
# the Southern Ocean and has none at all in eight cities (Karachi, Nanjing ...).
SO2SAT_GPKG = DATA_DIR / "input/So2Sat-LCZ42/v4/patches_reference_rxr.gpkg"
WUDAPT_GPKG = DATA_DIR / "input/WUDAPT/LCZ-Generator_training_areas_2024-10-01.gpkg"
BOUNDS_CSV = Path("data/so2sat_guppd_bounds.csv")
GUPPD_GPKG = DATA_DIR / "input/NASA/GUPPD/urbanspatial-guppd-v1-gpkg.gpkg"
GUPPD_LAYER = "urbanspatial_guppd_v1_polygons"
# GHS-SMOD degree of urbanisation, as the GUPPD documentation names the levels.
GUPPD_LEVELS = [("30", "Urban centre", "#7b2d8e"),
                ("22", "Dense urban cluster", "#d1495b"),
                ("23", "Semi-dense urban cluster", "#edae49")]
GUPPD_POP = "GHSPOP_R23_2020"

# The only WUDAPT columns this script ever asks the driver for. Anything not
# named here -- in particular firstname / lastname / cite_as / remarks /
# reference -- never enters the process, so it cannot leak into the payload.
WUDAPT_COLUMNS = ["submission_id", "submission_date", "city", "representative_date",
                  "license", "version", "class", "area", "qc_step1", "qc_step2",
                  "qc_step3", "oa", "oau", "oabu", "oaw"]
EXCLUDED_COLUMNS = ["firstname", "lastname", "cite_as", "remarks", "reference"]

SPLIT_LEVELS = [("training", "Training"), ("validation", "Validation"),
                ("testing", "Testing")]
# The 2024-10-01 release carries 580 rows of class 18 and 53 of class 19, which
# the Stewart & Oke scheme does not define (lcz_wudapt drops them). The site's
# job is to show every label, so they are drawn, in grey, under an honest name.
EXTRA_CLASSES = {18: ("Undocumented code 18", "#9e9e9e"),
                 19: ("Undocumented code 19", "#cfcfcf")}
YEAR_RANGE = (1980, 2025)          # representative_date also holds 0202, 2117, 2323
MISSING_LEVEL = "(not recorded)"
SIMPLIFY_DEG = 1e-5                # ~1 m; trims the 12k-vertex outliers
U2_MISSING = 65535

NETLIFY_TOML_FOR_ZIP = """\
# Dropped into Netlify as a zip; the archive root is the site root.
[[headers]]
  for = "/data/*.bin"
  [headers.values]
    Cache-Control = "public, max-age=0, must-revalidate"

[[headers]]
  for = "/assets/*"
  [headers.values]
    Cache-Control = "public, max-age=86400"
"""


# ── Geometry ─────────────────────────────────────────────────────────────────
def quantize_rings(geoms: np.ndarray, simplify: float = 0.0):
    """Exterior rings -> (anchor/scale frame, per-polygon counts, uint16 xy).

    Holes are dropped (1,456 of 630,311 WUDAPT polygons have any; at a scale
    where the question is "where are the labels" they do not change the
    answer). Rings are oriented counter-clockwise and lose their closing
    vertex, which the client puts back when it decodes.
    """
    geoms = shapely.force_2d(np.asarray(geoms))
    polys = shapely.polygons(shapely.get_exterior_ring(geoms))
    if simplify:
        polys = shapely.simplify(polys, simplify, preserve_topology=True)
    polys = shapely.orient_polygons(polys)          # exterior CCW
    ring = shapely.get_exterior_ring(polys)
    xy, idx = shapely.get_coordinates(ring, return_index=True)
    n = len(geoms)
    counts = np.bincount(idx, minlength=n)
    last = np.cumsum(counts) - 1
    keep = np.ones(len(xy), bool)
    keep[last[counts > 0]] = False                  # the closing vertex
    xy, idx = xy[keep], idx[keep]
    counts = np.bincount(idx, minlength=n)
    if (counts < 3).any():
        raise ValueError(f"{int((counts < 3).sum())} rings with <3 vertices")
    if counts.max() >= 65535:
        raise ValueError(f"a ring has {counts.max()} vertices; counts are uint16")

    b = shapely.bounds(ring)
    x0, y0 = b[:, 0], b[:, 1]
    sx = np.maximum(b[:, 2] - b[:, 0], 1e-9) / 65535.0
    sy = np.maximum(b[:, 3] - b[:, 1], 1e-9) / 65535.0
    q = np.empty((len(xy), 2), np.uint16)
    q[:, 0] = np.clip(np.rint((xy[:, 0] - x0[idx]) / sx[idx]), 0, 65535)
    q[:, 1] = np.clip(np.rint((xy[:, 1] - y0[idx]) / sy[idx]), 0, 65535)
    frame = pd.DataFrame({"x0": x0, "y0": y0, "sx": sx, "sy": sy})
    return frame, counts.astype(np.uint16), q


def write_rings(path: Path, counts: np.ndarray, q: np.ndarray) -> dict:
    """counts u2[n], zero-padded to 4 bytes, then interleaved xy u2[2V]."""
    head = counts.astype("<u2").tobytes()
    pad = (-len(head)) % 4
    body = q.astype("<u2").tobytes()
    path.write_bytes(head + b"\0" * pad + body)
    logger.info(f"  {path.name}: {len(counts):,} rings, {len(q):,} vertices, "
                f"{path.stat().st_size / 1e6:.1f} MB")
    return {"file": f"data/{path.name}", "n_vertices": int(len(q)),
            "vertex_offset": len(head) + pad}


# ── Columns ──────────────────────────────────────────────────────────────────
def categorical(values: pd.Series, order: list[str] | None = None):
    """Codes + levels, with missing turned into a real, selectable level."""
    codes, levels = encode_categorical(values, order)
    sentinel = 65535 if len(levels) > 254 else 255
    if (codes == sentinel).any():
        codes = np.where(codes == sentinel, len(levels), codes)
        levels = list(levels) + [MISSING_LEVEL]
    return codes, [str(x) for x in levels]


def year_codes(years: pd.Series) -> np.ndarray:
    y = pd.to_numeric(years, errors="coerce")
    ok = y.between(*YEAR_RANGE)
    return np.where(ok, y.fillna(0), U2_MISSING).astype(np.int64)


def qc_flag(values: pd.Series) -> pd.Series:
    """The release mixes True/False with T/F; anything else is missing."""
    s = values.astype("string").str.strip().str.upper()
    return s.map({"TRUE": "Pass", "T": "Pass", "FALSE": "Fail", "F": "Fail"})


def lcz_legend(codes) -> list[dict]:
    out = []
    for c in sorted(int(x) for x in codes):
        if c in lcz_dict:
            out.append({"code": c, "name": lcz_dict[c]["name"],
                        "colour": lcz_dict[c]["color"]})
        elif c in EXTRA_CLASSES:
            name, colour = EXTRA_CLASSES[c]
            out.append({"code": c, "name": name, "colour": colour})
        else:
            raise ValueError(f"unknown LCZ code {c}")
    return out


def range_column(label, table, v, *, dtype, scale="linear", fmt=".2f"):
    """Range-filter metadata; bounds come from the finite, non-sentinel values."""
    arr = np.asarray(v, dtype="float64")
    ok = np.isfinite(arr) & (arr != U2_MISSING if dtype == "u2" else True)
    if scale == "log":
        ok &= arr > 0
    if not ok.any():
        lo = hi = 0.0
    elif scale == "log":
        # The extremes (a 3e-11 km2 sliver, a 22,000 km2 region) would squeeze
        # every real polygon into a few slider steps; the slider's ends are
        # open anyway, so nothing outside the 0.1-99.9% range is lost.
        lo, hi = (float(x) for x in np.percentile(arr[ok], [0.1, 99.9]))
    else:
        lo, hi = float(arr[ok].min()), float(arr[ok].max())
    return {"label": label, "kind": "range", "table": table, "dtype": dtype,
            "min": lo, "max": hi, "scale": scale, "format": fmt,
            "n_missing": int((~ok).sum())}


# ── Datasets ─────────────────────────────────────────────────────────────────
def build_so2sat(gpkg: Path, bounds_csv: Path, out: Path) -> dict:
    df = pyogrio.read_dataframe(gpkg, columns=["LCZ_class", "patch_id", "dataset"])
    logger.info(f"So2Sat: {len(df):,} patches from {gpkg.name}")
    frame, counts, q = quantize_rings(df.geometry.values)

    cen = shapely.centroid(df.geometry.values)
    city = pd.Series(assign_city(np.column_stack([shapely.get_x(cen), shapely.get_y(cen)]),
                                 bounds_csv))
    # GUPPD merges Guangzhou and Shenzhen into one polygon; the training
    # patches of that polygon are Shenzhen (see the so2sat-52-cities note).
    city[(city == "Guangzhou") & (df["dataset"] == "training").to_numpy()] = "Shenzhen"
    city = city.replace({"东营区": "Dongying"})

    split_code, split_levels = categorical(df["dataset"], [k for k, _ in SPLIT_LEVELS])
    split_levels = [dict(SPLIT_LEVELS).get(s, s) for s in split_levels]
    city_code, city_levels = categorical(city)
    lcz = df["LCZ_class"].astype(int)

    rows = frame.assign(patch_id=df["patch_id"].astype(np.int64).to_numpy(),
                        lcz=lcz.to_numpy(), split=split_code, city=city_code)
    fields = [("x0", "f4"), ("y0", "f4"), ("sx", "f4"), ("sy", "f4"),
              ("patch_id", "u4"), ("lcz", "u1"), ("split", "u1"), ("city", "u1")]
    layout = write_block(out / "so2sat_rows.bin", rows, fields)
    rings = write_rings(out / "so2sat_rings.bin", counts, q)
    return {
        "key": "so2sat", "label": "So2Sat LCZ42",
        "blurb": "320 m patches labelled by experts in 52 cities (Zhu et al. 2020).",
        "source": gpkg.name, "n": int(len(df)),
        "rows": {"file": "data/so2sat_rows.bin", "fields": layout},
        "subs": None, "rings": rings,
        "columns": {
            "lcz": {"label": "LCZ class", "kind": "lcz", "table": "rows",
                    "levels": lcz_legend(lcz.unique())},
            "split": {"label": "Split (dataset)", "kind": "categorical",
                      "table": "rows", "levels": split_levels},
            "city": {"label": "City", "kind": "categorical", "table": "rows",
                     "levels": city_levels, "search": True},
            "patch_id": {"label": "Patch id", "kind": "id", "table": "rows"},
        },
        "filters": ["lcz", "split", "city"],
        "tooltip": ["lcz", "split", "city", "patch_id"],
    }


def build_wudapt(gpkg: Path, out: Path) -> dict:
    layer = gpkg.stem
    df = pyogrio.read_dataframe(gpkg, layer=layer, columns=WUDAPT_COLUMNS)
    leaked = set(df.columns) & set(EXCLUDED_COLUMNS)
    assert not leaked, f"personal-data columns were read: {leaked}"
    logger.info(f"WUDAPT: {len(df):,} polygons, "
                f"{df['submission_id'].nunique():,} submissions")
    frame, counts, q = quantize_rings(df.geometry.values, simplify=SIMPLIFY_DEG)

    # ── Per-submission table ────────────────────────────────────────────────
    sub_keys, sub_code = np.unique(df["submission_id"].astype(str), return_inverse=True)
    first = df.groupby("submission_id", sort=True).first()
    assert (first.index.astype(str) == sub_keys).all()
    s_city, city_levels = categorical(first["city"].astype(str).str.strip())
    s_version, version_levels = categorical(
        first["version"], sorted(first["version"].dropna().astype(str).unique(),
                                 key=lambda v: tuple(int(p) if p.isdigit() else 0
                                                     for p in v.split("."))))
    s_license, license_levels = categorical(first["license"])
    subs = pd.DataFrame({
        "oa": first["oa"].to_numpy(), "oau": first["oau"].to_numpy(),
        "oabu": first["oabu"].to_numpy(), "oaw": first["oaw"].to_numpy(),
        "city": s_city,
        "rep_year": year_codes(first["representative_date"].astype(str).str[:4]),
        "sub_year": year_codes(pd.to_datetime(first["submission_date"], utc=True,
                                              errors="coerce").dt.year),
        "version": s_version, "license": s_license,
    })
    sub_fields = [("oa", "f4"), ("oau", "f4"), ("oabu", "f4"), ("oaw", "f4"),
                  ("city", "u2"), ("rep_year", "u2"), ("sub_year", "u2"),
                  ("version", "u1"), ("license", "u1")]
    sub_layout = write_block(out / "wudapt_subs.bin", subs, sub_fields)

    # ── Per-row table ───────────────────────────────────────────────────────
    qc = {}
    for i in (1, 2, 3):
        qc[f"qc{i}"], qc_levels = categorical(qc_flag(df[f"qc_step{i}"]), ["Fail", "Pass"])
        assert qc_levels[:2] == ["Fail", "Pass"]
    lcz = df["class"].astype(int)
    rows = frame.assign(area=df["area"].to_numpy(), sub=sub_code,
                        lcz=lcz.to_numpy(), **qc)
    fields = [("x0", "f4"), ("y0", "f4"), ("sx", "f4"), ("sy", "f4"), ("area", "f4"),
              ("sub", "u2"), ("lcz", "u1"), ("qc1", "u1"), ("qc2", "u1"), ("qc3", "u1")]
    layout = write_block(out / "wudapt_rows.bin", rows, fields)
    rings = write_rings(out / "wudapt_rings.bin", counts, q)

    def qc_col(i, what):
        levels = ["Fail", "Pass"] + ([MISSING_LEVEL] if qc[f"qc{i}"].max() > 1 else [])
        return {"label": f"QC step {i}: {what}", "kind": "categorical",
                "table": "rows", "levels": levels}

    return {
        "key": "wudapt", "label": "WUDAPT LCZ Generator",
        "blurb": "Training areas digitised by volunteers and submitted to the "
                 "LCZ Generator (Demuzere et al. 2021), release 2024-10-01.",
        "source": gpkg.name, "n": int(len(df)),
        "rows": {"file": "data/wudapt_rows.bin", "fields": layout},
        "subs": {"file": "data/wudapt_subs.bin", "fields": sub_layout,
                 "n": int(len(subs)), "key": "sub"},
        "rings": rings,
        "columns": {
            "lcz": {"label": "LCZ class", "kind": "lcz", "table": "rows",
                    "levels": lcz_legend(lcz.unique())},
            "city": {"label": "City (as submitted)", "kind": "categorical",
                     "table": "subs", "levels": city_levels, "search": True},
            "rep_year": range_column("Representative year", "subs", subs["rep_year"],
                                     dtype="u2", fmt="d"),
            "sub_year": range_column("Submission year", "subs", subs["sub_year"],
                                     dtype="u2", fmt="d"),
            "qc1": qc_col(1, "polygon area"),
            "qc2": qc_col(2, "polygon count"),
            "qc3": qc_col(3, "class count"),
            "oa": range_column("Overall accuracy (OA)", "subs", subs["oa"], dtype="f4"),
            "oau": range_column("OA, urban classes", "subs", subs["oau"], dtype="f4"),
            "oabu": range_column("OA, built vs natural", "subs", subs["oabu"], dtype="f4"),
            "oaw": range_column("Weighted OA", "subs", subs["oaw"], dtype="f4"),
            "area": range_column("Polygon area (km²)", "rows", rows["area"],
                                 dtype="f4", scale="log", fmt=".3g"),
            "version": {"label": "Generator version", "kind": "categorical",
                        "table": "subs", "levels": version_levels},
            "license": {"label": "Licence", "kind": "categorical", "table": "subs",
                        "levels": license_levels},
        },
        "filters": ["lcz", "city", "rep_year", "sub_year", "qc1", "qc2", "qc3",
                    "oa", "oau", "oabu", "oaw", "area", "version", "license"],
        "tooltip": ["lcz", "city", "rep_year", "sub_year", "qc1", "qc2", "qc3",
                    "oa", "area", "version", "license"],
    }


def build_guppd(gpkg: Path, out: Path) -> dict:
    """GUPPD settlement outlines, as is: every polygon of every SMOD level.

    Multipolygons are exploded so each ring is one row (`feat` points back at
    the settlement); names ship as one newline-separated text file indexed by
    `feat`, since 82,637 distinct names do not belong in the manifest.
    """
    df = pyogrio.read_dataframe(gpkg, layer=GUPPD_LAYER,
                                columns=["SMOD_LEVEL", "CNTRY_NAME", "JRC_NAME_MAIN",
                                         "AREA_SQKM", GUPPD_POP])
    parts, feat = shapely.get_parts(df.geometry.values, return_index=True)
    logger.info(f"GUPPD: {len(df):,} settlements, {len(parts):,} polygon parts")
    frame, counts, q = quantize_rings(parts)

    level_code, level_levels = categorical(df["SMOD_LEVEL"].astype(str),
                                           [k for k, _, _ in GUPPD_LEVELS])
    country_code, country_levels = categorical(df["CNTRY_NAME"])
    rows = frame.assign(area=df["AREA_SQKM"].to_numpy()[feat],
                        pop=df[GUPPD_POP].to_numpy()[feat],
                        feat=feat, level=level_code[feat], country=country_code[feat])
    fields = [("x0", "f4"), ("y0", "f4"), ("sx", "f4"), ("sy", "f4"), ("area", "f4"),
              ("pop", "f4"), ("feat", "u4"), ("level", "u1"), ("country", "u1")]
    layout = write_block(out / "guppd_rows.bin", rows, fields)
    rings = write_rings(out / "guppd_rings.bin", counts, q)
    names = df["JRC_NAME_MAIN"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True)
    (out / "guppd_names.txt").write_text("\n".join(names), encoding="utf-8")
    label = dict((k, (lab, col)) for k, lab, col in GUPPD_LEVELS)
    return {
        "key": "guppd", "label": "GUPPD settlements",
        "blurb": "Global Urban Polygons and Points Dataset v1 (CIESIN): "
                 "GHS-SMOD settlement outlines.",
        "source": gpkg.name, "n": int(len(parts)), "n_features": int(len(df)),
        "rows": {"file": "data/guppd_rows.bin", "fields": layout},
        "rings": rings, "names": "data/guppd_names.txt",
        "columns": {
            "level": {"label": "Settlement type", "kind": "categorical", "table": "rows",
                      "levels": [label.get(k, (k, ""))[0] for k in level_levels],
                      "colours": [label.get(k, (k, "#888888"))[1] for k in level_levels]},
            "country": {"label": "Country", "kind": "categorical", "table": "rows",
                        "levels": country_levels},
            "area": {"label": "Area (km²)", "kind": "range", "table": "rows",
                     "dtype": "f4", "format": ".3g"},
            "pop": {"label": "Population 2020 (GHS-POP)", "kind": "range",
                    "table": "rows", "dtype": "f4", "format": "d"},
        },
        "tooltip": ["level", "country", "area", "pop"],
    }


# ── Packaging ────────────────────────────────────────────────────────────────
def manifest_files(manifest: dict) -> set[str]:
    need = {"index.html", "assets/app.js", "assets/app.css", "data/manifest.json"}
    for ds in manifest["datasets"].values():
        need.add(ds["rows"]["file"])
        need.add(ds["rings"]["file"])
        if ds["subs"]:
            need.add(ds["subs"]["file"])
    guppd = manifest.get("overlays", {}).get("guppd")
    if guppd:
        need.update([guppd["rows"]["file"], guppd["rings"]["file"], guppd["names"]])
    return need


def write_zip(site: Path, dest: Path) -> None:
    """Zip for a Netlify drop: files at the ARCHIVE ROOT, then verified."""
    files = sorted(p for p in site.rglob("*") if p.is_file())
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, p.relative_to(site).as_posix())
        z.writestr("netlify.toml", NETLIFY_TOML_FOR_ZIP)
    with zipfile.ZipFile(dest) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"corrupt entry in {dest}: {bad}")
        names = set(z.namelist())
        missing = sorted(manifest_files(json.loads(z.read("data/manifest.json"))) - names)
        if missing:
            raise RuntimeError(f"{dest} is missing: {missing}")
    raw = sum(p.stat().st_size for p in files)
    logger.info(f"zip → {dest}  ({dest.stat().st_size / 1e6:.1f} MB "
                f"from {raw / 1e6:.1f} MB, {len(files) + 1} entries)")
    logger.info("  deploy: drag this file onto https://app.netlify.com/drop")


def build_site(out: Path, so2sat_gpkg: Path, wudapt_gpkg: Path,
               bounds_csv: Path, guppd_gpkg: Path | None = None,
               demuzere_url: str | None = None) -> dict:
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": {
            "so2sat": build_so2sat(so2sat_gpkg, bounds_csv, out / "data"),
            "wudapt": build_wudapt(wudapt_gpkg, out / "data"),
        },
        "overlays": {
            "guppd": build_guppd(guppd_gpkg, out / "data") if guppd_gpkg else None,
            # Tiles live off-site (R2): the archive is GBs and is read by range
            # requests, so only its URL is part of the build.
            "demuzere": {
                "label": "Demuzere et al. (2022) LCZ map",
                "url": demuzere_url, "max_zoom": 11,
                "attribution": "LCZ map © Demuzere et al. 2022, CC BY 4.0",
            },
        },
    }
    for f in ("app.js", "app.css"):
        shutil.copy(TEMPLATE_DIR / f, out / "assets" / f)
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR),
                      autoescape=select_autoescape(["html"]))
    # Cache-bust the assets by content, not by date.
    #
    # python -m http.server sends no Cache-Control, so browsers fall back to
    # heuristic caching and will happily pair a fresh index.html with a stale
    # app.js. That combination is silently broken rather than obviously
    # broken: the new markup has an Apply button the old script never wires
    # up, so every filter looks dead. Hashing the bytes means a changed asset
    # is a changed URL and the pairing cannot happen.
    asset_v = hashlib.sha256(
        b"".join((out / "assets" / f).read_bytes() for f in ("app.js", "app.css"))
    ).hexdigest()[:12]

    html = env.get_template("index.html.j2").render(
        asset_v=asset_v,
        datasets=manifest["datasets"], overlays=manifest["overlays"],
        generated=manifest["generated"][:10])
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "data" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    logger.info(f"site → {out}  ({total / 1e6:.1f} MB total)")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--so2sat-gpkg", type=Path, default=SO2SAT_GPKG)
    ap.add_argument("--wudapt-gpkg", type=Path, default=WUDAPT_GPKG)
    ap.add_argument("--bounds-csv", type=Path, default=BOUNDS_CSV)
    ap.add_argument("--guppd-gpkg", type=Path, default=GUPPD_GPKG)
    ap.add_argument("--no-guppd", action="store_true", help="leave the GUPPD layer out")
    ap.add_argument("--demuzere-url", default=None,
                    help="URL of the PMTiles archive from build_demuzere_pmtiles.py "
                         "(e.g. on Cloudflare R2); without it the layer is disabled")
    ap.add_argument("--output-dir", type=Path, default=Path("label_site"))
    ap.add_argument("--zip", type=Path, default=None,
                    help="also pack the site into this .zip for a Netlify drop")
    args = ap.parse_args(argv)
    build_site(args.output_dir, args.so2sat_gpkg, args.wudapt_gpkg, args.bounds_csv,
               None if args.no_guppd else args.guppd_gpkg, args.demuzere_url)
    logger.info(f"serve with:  python -m http.server -d {args.output_dir} 8001")
    if args.zip:
        write_zip(args.output_dir, args.zip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
