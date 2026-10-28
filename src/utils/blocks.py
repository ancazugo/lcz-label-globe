"""Typed-array block payloads: the on-disk format the client fetches.

Each block is one contiguous buffer holding one typed array per field, in
declaration order; the client slices it back apart using the field list in
the manifest. Offsets are NOT stored -- they are the cumulative
``n * itemsize``, which is why fields are declared widest-first, so every
Float32Array view lands naturally aligned.

Vendored from eo_fm's build_embedding_site.py, which the label globe used to
import directly. That import also dragged in pyarrow, loguru and a 484-line
embedding registry this site never touches.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("build_site")

ITEMSIZE = {"f4": 4, "u4": 4, "u2": 2, "u1": 1}


def write_block(path: Path, df: pd.DataFrame, fields) -> list[dict]:
    """Concatenate one typed array per field; return the field list.

    Offsets are NOT recorded: they are `cumulative n * itemsize`, so they differ
    per tier and the client derives them from the row count instead. What is
    asserted here is that every offset lands aligned -- a misaligned
    Float32Array view throws in the browser and succeeds silently here.
    """
    n = len(df)
    chunks, offset = [], 0
    for name, dt in fields:
        size = ITEMSIZE[dt]
        assert offset % size == 0, f"{name} lands at unaligned offset {offset}"
        arr = np.ascontiguousarray(df[name].to_numpy(), dtype=np.dtype("<" + dt))
        assert len(arr) == n, f"{name}: {len(arr)} rows, expected {n}"
        chunks.append(arr.tobytes())
        offset += size * n
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(chunks))
    logger.info(f"  {path.name}: {n:,} rows x {len(fields)} fields, "
                f"{path.stat().st_size / 1e6:.1f} MB")
    return [{"name": nm, "dtype": dt} for nm, dt in fields]


def encode_categorical(values: pd.Series, order: list[str] | None = None):
    """Dictionary-code a column; returns (codes, levels). 255/65535 = missing.

    Missing is a real state here, not an error: Koppen is a land-only raster so
    ~9.3% of patches sample nothing, and GUPPD is a settlement layer rather
    than a partition so ~49% fall outside every polygon.
    """
    s = values.astype("string")
    levels = order if order is not None else sorted(x for x in s.dropna().unique())
    idx = {lv: i for i, lv in enumerate(levels)}
    codes = s.map(idx).to_numpy(dtype="float64")
    sentinel = 65535 if len(levels) > 254 else 255
    codes = np.where(np.isnan(codes), sentinel, codes)
    return codes.astype(np.int64), levels
