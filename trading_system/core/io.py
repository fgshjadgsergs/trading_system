"""Parquet storage: append-only batches, hourly rotation, exchange/symbol/date partitions.

Layout: <root>/<stream>/exchange=<e>/symbol=<s>/date=<YYYY-MM-DD>/hour=<HH>/part-<n>.parquet
A unified reader scans this layout regardless of whether files came from the
live recorder or from a normalized data.binance.vision archive.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import polars as pl

from trading_system.core.schema import POLARS_SCHEMAS
from trading_system.core.timeutils import ns_to_dt


def partition_dir(root: Path, stream: str, exchange: str, symbol: str, ts: int) -> Path:
    dt = ns_to_dt(ts)
    return (
        root
        / stream
        / f"exchange={exchange}"
        / f"symbol={symbol}"
        / f"date={dt.strftime('%Y-%m-%d')}"
        / f"hour={dt.strftime('%H')}"
    )


def write_batch(root: Path, stream: str, frame: pl.DataFrame) -> list[Path]:
    """Append one batch, splitting rows across hourly partition directories."""
    if frame.is_empty():
        return []
    ts_col = "ts_event" if "ts_event" in frame.columns else "ts_open"
    written: list[Path] = []
    parts = frame.with_columns(
        (pl.col(ts_col) - pl.col(ts_col) % (3_600 * 1_000_000_000)).alias("_hour_bucket")
    ).partition_by("_hour_bucket", as_dict=False)
    for part in parts:
        first = part.row(0, named=True)
        d = partition_dir(root, stream, first["exchange"], first["symbol"], first[ts_col])
        d.mkdir(parents=True, exist_ok=True)
        n = len(list(d.glob("part-*.parquet")))
        path = d / f"part-{n:05d}.parquet"
        # write to a dot-prefixed tmp (invisible to part-*.parquet globs), then
        # atomically publish: a crash mid-write never leaves a readable partial
        tmp = d / f".{path.name}.tmp"
        part.drop("_hour_bucket").write_parquet(tmp)
        os.replace(tmp, path)
        written.append(path)
    return written


def read_stream(
    root: Path,
    stream: str,
    exchange: str | None = None,
    symbol: str | None = None,
    ts_from: int | None = None,
    ts_to: int | None = None,
) -> pl.DataFrame:
    """Unified reader over the partitioned layout; source (live/Vision) agnostic."""
    base = root / stream
    if not base.exists():
        return pl.DataFrame(schema=POLARS_SCHEMAS[stream])
    pattern = f"exchange={exchange or '*'}/symbol={symbol or '*'}/date=*/hour=*/part-*.parquet"
    files = sorted(base.glob(pattern))
    if not files:
        return pl.DataFrame(schema=POLARS_SCHEMAS[stream])
    lf = pl.scan_parquet(files)
    ts_col = "ts_event" if "ts_event" in POLARS_SCHEMAS[stream] else "ts_open"
    if ts_from is not None:
        lf = lf.filter(pl.col(ts_col) >= ts_from)
    if ts_to is not None:
        lf = lf.filter(pl.col(ts_col) < ts_to)
    return lf.sort(ts_col).collect()


def duckdb_query(root: Path, sql: str) -> pl.DataFrame:
    """Ad-hoc SQL over the parquet lake; use read_parquet('<root>/...') in SQL."""
    con = duckdb.connect()
    try:
        return con.execute(sql).pl()
    finally:
        con.close()
