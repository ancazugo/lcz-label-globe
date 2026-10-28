#!/usr/bin/env python
"""Tile the global Demuzere et al. (2022) LCZ map into one PMTiles archive.

    python src/build_demuzere_pmtiles.py \
        --input ${DATA_DIR}/input/Demuzere_2022_complete/lcz_filter_v3.tif \
        --output /scratch/acz25/eo_fm/demuzere_lcz_v3.pmtiles --max-zoom 11
    python src/build_demuzere_pmtiles.py ... --sample 24     # size estimate only

The archive is what the label globe (src/build_label_site.py, `--demuzere-url`)
draws as its "Demuzere 2022" layer. It is one file of raster tiles -- palette
PNGs in the LCZ colours, nodata transparent -- read by the browser with HTTP
range requests, so it must live on a host that honours `Range` and sends CORS
headers for the site's origin (Cloudflare R2 does; see the README).

HOW THE TILES ARE MADE
----------------------
Nearest-neighbour sampling straight from the source grid, no warping library:
each web-mercator output pixel takes the source cell under its centre. That is
the right resampling for a categorical map, and it means a tile can be made
from a plain numpy gather. Work is split into zoom-5 blocks (11.25 degrees
wide): a worker reads its block's source window once and cuts every tile from
zoom 5 to --max-zoom out of it. Zooms 0-4 are then mosaicked from the zoom-5
tiles by taking every other pixel, level by level.

The source is EPSG:4326 at 0.000898 degrees (~100 m). --max-zoom 11 gives
76 m pixels at the equator and ~49 m at 50 degrees, i.e. no visible loss of
the 100 m cells; MapLibre over-zooms beyond it.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.constants import lcz_dict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("demuzere_pmtiles")

TILE = 256
BLOCK_Z = 5


def palette() -> list[int]:
    pal = [0, 0, 0] * 256
    for code, v in lcz_dict.items():
        h = v["color"].lstrip("#")
        pal[3 * code:3 * code + 3] = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return pal


PALETTE = palette()


def encode(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr, mode="P")
    img.putpalette(PALETTE)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True, transparency=0)
    return buf.getvalue()


def tile_lonlat(z: int, x: int, y: int):
    """Pixel-centre longitudes (TILE,) and latitudes (TILE,) of one tile."""
    n = 2 ** z
    t = (np.arange(TILE) + 0.5) / TILE
    lon = -180.0 + (x + t) * 360.0 / n
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + t) / n))))
    return lon, lat


def tile_bounds(z: int, x: int, y: int):
    n = 2 ** z
    west, east = -180 + 360 * x / n, -180 + 360 * (x + 1) / n
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


def blocks_over(src_bounds) -> list[tuple[int, int]]:
    """Zoom-5 tiles that intersect the raster's bounds."""
    n = 2 ** BLOCK_Z
    out = []
    for x in range(n):
        for y in range(n):
            w, s, e, nn = tile_bounds(BLOCK_Z, x, y)
            if e > src_bounds.left and w < src_bounds.right and \
                    nn > src_bounds.bottom and s < src_bounds.top:
                out.append((x, y))
    return out


def render_block(path: str, bx: int, by: int, max_zoom: int):
    """All tiles z5..max_zoom inside zoom-5 tile (bx, by) -> [(z,x,y,bytes)], z5 array."""
    from pmtiles.tile import zxy_to_tileid
    with rasterio.open(path) as src:
        a, _, left, _, e, top = src.transform[:6]
        w, s, ee, n = tile_bounds(BLOCK_Z, bx, by)
        c0 = max(int(math.floor((w - left) / a)), 0)
        c1 = min(int(math.ceil((ee - left) / a)) + 1, src.width)
        r0 = max(int(math.floor((top - n) / -e)), 0)
        r1 = min(int(math.ceil((top - s) / -e)) + 1, src.height)
        if c1 <= c0 or r1 <= r0:
            return [], None
        data = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
    if not data.any():
        return [], None

    out, z5 = [], None
    for z in range(BLOCK_Z, max_zoom + 1):
        k = 2 ** (z - BLOCK_Z)
        for dx in range(k):
            for dy in range(k):
                x, y = bx * k + dx, by * k + dy
                lon, lat = tile_lonlat(z, x, y)
                col = np.floor((lon - left) / a).astype(np.int64) - c0
                row = np.floor((top - lat) / -e).astype(np.int64) - r0
                cok = (col >= 0) & (col < data.shape[1])
                rok = (row >= 0) & (row < data.shape[0])
                arr = np.zeros((TILE, TILE), np.uint8)
                if cok.any() and rok.any():
                    arr[np.ix_(rok, cok)] = data[np.ix_(row[rok], col[cok])]
                if z == BLOCK_Z:
                    z5 = arr
                if arr.any():
                    out.append((zxy_to_tileid(z, x, y), encode(arr)))
    return out, z5


def overview_tiles(z5: dict[tuple[int, int], np.ndarray]):
    """Zooms 0-4 from the zoom-5 arrays: mosaic four children, keep every other pixel."""
    from pmtiles.tile import zxy_to_tileid
    level, out = z5, []
    for z in range(BLOCK_Z - 1, -1, -1):
        nxt = {}
        for x in range(2 ** z):
            for y in range(2 ** z):
                kids = [level.get((2 * x + i, 2 * y + j)) for j in (0, 1) for i in (0, 1)]
                if all(k is None for k in kids):
                    continue
                kids = [k if k is not None else np.zeros((TILE, TILE), np.uint8) for k in kids]
                mosaic = np.block([[kids[0], kids[1]], [kids[2], kids[3]]])
                arr = mosaic[::2, ::2].copy()
                nxt[(x, y)] = arr
                if arr.any():
                    out.append((zxy_to_tileid(z, x, y), encode(arr)))
        level = nxt
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-zoom", type=int, default=11)
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    ap.add_argument("--sample", type=int, default=0,
                    help="render only N random land blocks and print a size estimate")
    args = ap.parse_args(argv)

    with rasterio.open(args.input) as src:
        blocks = blocks_over(src.bounds)
    logger.info(f"{len(blocks)} zoom-{BLOCK_Z} blocks over the raster, "
                f"zoom {BLOCK_Z}..{args.max_zoom} each, {args.workers} workers")
    if args.sample:
        rng = np.random.default_rng(0)
        blocks = [blocks[i] for i in rng.permutation(len(blocks))]

    tiles, z5, done, land = [], {}, 0, 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(render_block, str(args.input), bx, by, args.max_zoom): (bx, by)
                for bx, by in blocks}
        for f in as_completed(futs):
            out, arr = f.result()
            done += 1
            if arr is not None:
                land += 1
                z5[futs[f]] = arr
                tiles.extend(out)
            if args.sample and land >= args.sample:
                break
            if done % 50 == 0:
                mb = sum(len(t[1]) for t in tiles) / 1e6
                logger.info(f"  {done}/{len(blocks)} blocks, {len(tiles):,} tiles, {mb:,.0f} MB")
        if args.sample:
            for f in futs:
                f.cancel()

    if args.sample:
        per = sum(len(t[1]) for t in tiles) / max(land, 1)
        logger.info(f"sample: {land} land blocks, {len(tiles):,} tiles, "
                    f"{per / 1e6:.1f} MB per land block (before dedupe)")
        return 0

    tiles.extend(overview_tiles(z5))
    tiles.sort(key=lambda t: t[0])
    from pmtiles.tile import Compression, TileType
    from pmtiles.writer import Writer
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        w = Writer(f)
        for tid, data in tiles:
            w.write_tile(tid, data)
        w.finalize(
            {"tile_type": TileType.PNG, "tile_compression": Compression.NONE,
             "min_zoom": 0, "max_zoom": args.max_zoom,
             "min_lon_e7": -1800000000, "min_lat_e7": -600000000,
             "max_lon_e7": 1800000000, "max_lat_e7": 800000000,
             "center_zoom": 2, "center_lon_e7": 0, "center_lat_e7": 0},
            {"name": "Demuzere et al. (2022) global LCZ map",
             "attribution": "LCZ map © Demuzere et al. 2022 (CC BY 4.0)",
             "source": args.input.name,
             "legend": {str(k): {"name": v["name"], "colour": v["color"]}
                        for k, v in lcz_dict.items()}})
    logger.info(f"wrote {args.output}: {len(tiles):,} tiles, "
                f"{args.output.stat().st_size / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
