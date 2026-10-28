# LCZ label globe

A static web map of **where the Local Climate Zone training labels are**: all
400,673 [So2Sat-LCZ42](https://doi.org/10.1109/MGRS.2020.2964708) patches and
all 630,311 [WUDAPT LCZ Generator](https://doi.org/10.3389/fenvs.2021.637455)
training areas, each drawn in its LCZ colour on a MapLibre GL globe with
deck.gl.

No server, no API: the build writes typed-array binaries and the browser does
the rest.

![built from GeoPackages, deployed as static files](https://img.shields.io/badge/build-static-informational)

## What it does

- **Two datasets, one colour scheme.** So2Sat patches and WUDAPT training
  areas in the 17 Stewart & Oke LCZ colours. Shown side by side with a locked
  camera when you pick *Both*.
- **Filters declared by the data.** The sidebar is generated from the
  manifest, so it follows the dataset. So2Sat filters by LCZ class, split and
  city (52 of them — Guangzhou and Shenzhen are separated). WUDAPT adds
  representative and submission year, the three QC steps, the four submission
  accuracies (OA, OA-urban, OA-built/natural, weighted OA), polygon area,
  generator version and licence. Per-level counts honour whatever else is
  filtered.
- **Filters apply on a button press.** Re-evaluating 630k rings is one
  synchronous burst, so edits collect in a draft and the map only moves when
  you press *Apply* — with a visible busy state, instead of a frozen page you
  cannot distinguish from a crash.
- **Dots far out, polygons close in.** Centroids at globe scale; the real
  polygon rings are fetched on the first zoom past level 10.
- **Overlays.** GUPPD settlement outlines (123,034 polygons, all SMOD levels,
  ~11 MB, only downloaded when switched on) and the Demuzere et al. (2022)
  global LCZ map with an opacity slider.

## Privacy

WUDAPT is read through a **column whitelist**. `firstname`, `lastname`,
`cite_as` and the free-text `remarks` / `reference` fields never enter the
build, and a test asserts that no sentinel value from those columns reaches the
output.

## Build

The inputs are GeoPackages measured in hundreds of MB, so building needs access
to the data directory; the output is self-contained static files.

```bash
pip install -e ".[dev]"

export DATA_DIR=/maps/acz25/phd-thesis-data
export SQLITE_USE_OGR_VFS=YES NUMPY_MADVISE_HUGEPAGE=0

python src/build_label_site.py --output-dir label_site
python -m http.server -d label_site 8001
```

`SQLITE_USE_OGR_VFS=YES` is not optional in practice: GeoPackage reads fail
with `database is locked` without it when the filesystem is full.
`NUMPY_MADVISE_HUGEPAGE=0` avoids a ~40 s stall on large allocations on some
hosts.

Inputs, all under `$DATA_DIR`:

| File | Size |
|---|---|
| `input/So2Sat-LCZ42/v4/patches_reference_rxr.gpkg` | 85 MB |
| `input/WUDAPT/LCZ-Generator_training_areas_2024-10-01.gpkg` | 720 MB |
| `input/NASA/GUPPD/urbanspatial-guppd-v1-gpkg.gpkg` | 116 MB (`--no-guppd` skips it) |

The So2Sat source is `patches_reference_rxr.gpkg`, **not** the plain
`patches_reference.gpkg` — the latter puts some patches in the Southern Ocean
and has none at all in eight cities. `data/so2sat_guppd_bounds.csv` ships with
this repo and supplies the city centroids.

## Deploy

Source lives on `main`; the ~63 MB build output is published to a `deploy`
branch that is force-pushed, so its history never accumulates.

```bash
bash scripts/deploy.sh                        # build, then push to `deploy`
bash scripts/deploy.sh --demuzere-url https://<host>/demuzere_lcz_v3.pmtiles
```

In Netlify: connect the repo, set the production branch to `deploy`, publish
directory `.`, build command empty. `deploy/netlify.toml` is copied to the
branch root and sets the cache headers — the `.bin` payload must revalidate,
since it is content-addressed by rebuild rather than by filename.

`--zip <path>` instead produces a drag-and-drop archive with the files at the
archive root.

## The Demuzere overlay

The global LCZ map is 3.3 GB of tiles, so it is hosted separately and the build
records only its URL. Without `--demuzere-url` the toggle appears disabled.

```bash
pip install -e ".[demuzere]"

# 1. Tile the global map once (~6 min on 64 cores)
python src/build_demuzere_pmtiles.py \
    --input $DATA_DIR/input/Demuzere_2022_complete/lcz_filter_v3.tif \
    --output /scratch/demuzere_lcz_v3.pmtiles --max-zoom 11

# 2. Upload to Cloudflare R2. The dashboard caps uploads at 300 MB, so use the
#    S3 multipart API with an R2 API token:
aws s3 cp /scratch/demuzere_lcz_v3.pmtiles s3://<bucket>/ \
    --endpoint-url https://<account-id>.r2.cloudflarestorage.com

# 3. Enable public access (a custom domain for real traffic -- r2.dev is rate
#    limited) and add a CORS policy allowing GET/HEAD with the `range` header:
#    [{"AllowedOrigins": ["https://<your-site>"], "AllowedMethods": ["GET","HEAD"],
#      "AllowedHeaders": ["range","if-match"], "ExposeHeaders": ["etag"],
#      "MaxAgeSeconds": 3600}]
```

The browser reads the archive with HTTP range requests, so only the tiles on
screen are downloaded.

## Tests

```bash
pytest -q
```

The builder tests are fully offline: they synthesise tiny GeoPackages in a temp
directory and build a real site from them, so no data directory is needed. The
client tests run `src/label_site/app.js` in **QuickJS** against that payload —
there is no node here — and skip themselves if `quickjs` is not installed.

QuickJS stubs the DOM, which means it cannot see activation behaviour, hit
testing or layout shifts. Three real bugs hid in exactly that gap, so a real
browser is worth it for UI changes:

```bash
export PLAYWRIGHT_BROWSERS_PATH=/maps/acz25/.cache/ms-playwright
python -m playwright install chromium --only-shell
```

Drive it with the maps destroyed and `mouse.click` at explicit coordinates —
deck.gl software-rendering 400k polygons starves the main thread, and
Playwright's actionability checks then time out in ways that look exactly like
a broken page.

## Layout

```
src/build_label_site.py        the builder
src/label_site/                client: index.html.j2, app.js, app.css
src/build_demuzere_pmtiles.py  the overlay tiler (optional extra)
src/utils/                     lcz_dict, nearest-centroid city lookup,
                               typed-array block writer
data/so2sat_guppd_bounds.csv   city bounding boxes
deploy/netlify.toml            copied to the deploy branch root
scripts/deploy.sh              build + force-push to `deploy`
```

`app.js` keeps a pure core (`LabelCore`: block decoding, ring decoding, the
filter mask, cross-filter level counts) above the browser half, and returns
early when there is no `document` — that is what makes it testable headless.

## Credits

- So2Sat-LCZ42 — Zhu et al. (2020), [10.1109/MGRS.2020.2964708](https://doi.org/10.1109/MGRS.2020.2964708)
- WUDAPT LCZ Generator — Demuzere et al. (2021), [10.3389/fenvs.2021.637455](https://doi.org/10.3389/fenvs.2021.637455)
- Global LCZ map — Demuzere et al. (2022), [10.5194/essd-14-3835-2022](https://doi.org/10.5194/essd-14-3835-2022), CC BY 4.0
- LCZ scheme — Stewart & Oke (2012), [10.1175/BAMS-D-11-00019.1](https://doi.org/10.1175/BAMS-D-11-00019.1)
- GUPPD v1 — CIESIN, Columbia University
