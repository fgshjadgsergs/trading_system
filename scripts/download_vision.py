#!/usr/bin/env python3
"""Bulk download of data.binance.vision USDT-M archives into the parquet lake.

Plans the archive list, downloads zips with sha256 .CHECKSUM verification,
normalizes each kind into the unified schema and appends it to the lake, then
refreshes the dataset catalog. The transport is injectable (``main(argv,
fetch=...)``) so the whole flow is testable offline; the default fetch uses
urllib.

Example:
    python3 scripts/download_vision.py --lake data/lake \
        --symbols BTCUSDT SOLUSDT --kinds aggTrades klines fundingRate \
        --start 2024-01-01 --end 2024-01-07
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import urllib.request
from functools import partial
from pathlib import Path

import structlog

from trading_system.collectors.vision import (
    KINDS,
    VISION_BASE,
    DownloadResult,
    FetchFn,
    build_catalog,
    download,
    ingest_zip,
    plan_downloads,
    write_catalog,
)

log = structlog.get_logger("download_vision")


def urllib_fetch(url: str, timeout: float = 300.0) -> bytes:
    """Default network transport: chunked GET returning the response body.

    `timeout` is the per-socket-operation limit; reading in 1 MiB chunks keeps
    a slow-but-alive download of a multi-hundred-MB archive from tripping it.
    """
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        chunks: list[bytes] = []
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lake", type=Path, required=True, help="parquet lake root directory")
    p.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="where to keep the raw zips (default: <lake>/_archive)",
    )
    p.add_argument("--symbols", nargs="+", required=True, help="e.g. BTCUSDT SOLUSDT")
    p.add_argument("--kinds", nargs="+", choices=sorted(KINDS), required=True)
    p.add_argument(
        "--timeout", type=float, default=300.0, help="per-socket-read timeout, seconds"
    )
    p.add_argument("--start", type=dt.date.fromisoformat, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=dt.date.fromisoformat, required=True, help="YYYY-MM-DD")
    p.add_argument("--period", choices=("daily", "monthly"), default="daily")
    p.add_argument(
        "--intervals", nargs="+", default=["1m"], help="kline intervals for klines kinds"
    )
    p.add_argument("--base-url", default=VISION_BASE)
    p.add_argument("--no-verify", action="store_true", help="skip .CHECKSUM verification")
    p.add_argument("--no-normalize", action="store_true", help="download zips only")
    p.add_argument("--no-catalog", action="store_true", help="skip catalog refresh")
    p.add_argument("--redownload", action="store_true", help="do not skip existing archives")
    p.add_argument(
        "--reingest",
        action="store_true",
        help="also normalize archives that were already on disk (may duplicate lake rows)",
    )
    return p.parse_args(argv)


def _ingest(results: list[DownloadResult], lake_root: Path, *, reingest: bool) -> int:
    """Normalize downloaded archives into the lake; returns count ingested.

    Archives skipped as already-existing are re-ingested only with ``reingest``
    because the lake is append-only and would duplicate their rows.
    """
    n = 0
    for res in results:
        if res.path is None or (res.status == "skipped_existing" and not reingest):
            continue
        streams = ingest_zip(
            res.path.read_bytes(), res.item.kind, res.item.symbol, lake_root
        )
        n += 1
        log.info(
            "ingested",
            archive=res.path.name,
            streams={s: len(paths) for s, paths in streams.items()},
        )
    return n


def main(argv: list[str] | None = None, fetch: FetchFn | None = None) -> int:
    args = parse_args(argv)
    if fetch is None:
        fetch = partial(urllib_fetch, timeout=args.timeout)
    archive_dir = args.archive_dir or (args.lake / "_archive")
    items = plan_downloads(
        symbols=args.symbols,
        kinds=args.kinds,
        start=args.start,
        end=args.end,
        period=args.period,
        intervals=args.intervals,
        base=args.base_url,
    )
    log.info("plan", archives=len(items), dest=str(archive_dir))
    results = download(
        items,
        archive_dir,
        fetch,
        verify=not args.no_verify,
        skip_existing=not args.redownload,
    )
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    log.info("download_done", **by_status)
    if not args.no_normalize:
        n = _ingest(results, args.lake, reingest=args.reingest)
        log.info("normalize_done", archives=n)
    if not args.no_catalog:
        path = write_catalog(args.lake, build_catalog(args.lake))
        log.info("catalog_written", path=str(path))
    errors = sum(1 for r in results if not r.ok)
    if errors:
        log.error("finished_with_errors", errors=errors)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
