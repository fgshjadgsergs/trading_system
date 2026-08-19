"""Batch recording of normalized records and generic REST polling.

BatchWriter buffers unified-schema records per (stream, exchange, symbol) and
flushes through core.io.write_batch on max_rows or max_age (injectable clock),
so hourly partitioning and append-only part numbering come from core.io.

RestPoller drives periodic REST endpoints (openInterest every 5-10 s, the
three long/short ratios every 300 s) with injectable fetch/clock/sleep.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import structlog

from trading_system.core.io import write_batch
from trading_system.core.schema import STREAM_OF_TYPE, Record, records_to_frame
from trading_system.core.timeutils import NS_PER_S, now_ns

log = structlog.get_logger(__name__)

BufKey = tuple[str, str, str]  # (stream, exchange, symbol)


class BatchWriter:
    """Append-only parquet batcher: flush on max_rows OR max_age seconds.

    Buffers are keyed by (stream, exchange, symbol) so every flushed frame is
    homogeneous and core.io.write_batch partitions it correctly.
    """

    def __init__(
        self,
        root: Path,
        *,
        max_rows: int = 50_000,
        max_age_s: float = 60.0,
        clock: Callable[[], int] = now_ns,
    ) -> None:
        if max_rows < 1:
            raise ValueError("max_rows must be >= 1")
        self._root = Path(root)
        self._max_rows = max_rows
        self._max_age_ns = int(max_age_s * NS_PER_S)
        self._clock = clock
        self._buf: dict[BufKey, list[Record]] = {}
        self._first_add: dict[BufKey, int] = {}

    def add(self, rec: Record) -> list[Path]:
        """Buffer one record; returns paths written if a flush was triggered."""
        stream = STREAM_OF_TYPE[type(rec)]
        key = (stream, rec.exchange, rec.symbol)
        buf = self._buf.setdefault(key, [])
        if not buf:
            self._first_add[key] = self._clock()
        buf.append(rec)
        if len(buf) >= self._max_rows or self._aged(key):
            return self._flush_key(key)
        return []

    def poll(self) -> list[Path]:
        """Flush buffers that exceeded max_age even without new records."""
        written: list[Path] = []
        for key in [k for k in self._buf if self._aged(k)]:
            written.extend(self._flush_key(key))
        return written

    def flush_all(self) -> list[Path]:
        """Flush every non-empty buffer (shutdown path)."""
        written: list[Path] = []
        for key in list(self._buf):
            written.extend(self._flush_key(key))
        return written

    @property
    def buffered_rows(self) -> int:
        return sum(len(v) for v in self._buf.values())

    def _aged(self, key: BufKey) -> bool:
        first = self._first_add.get(key)
        return first is not None and self._clock() - first >= self._max_age_ns

    def _flush_key(self, key: BufKey) -> list[Path]:
        buf = self._buf.get(key)
        if not buf:
            return []
        stream = key[0]
        frame = records_to_frame(buf, stream)
        paths = write_batch(self._root, stream, frame)
        log.info("batch_flush", stream=stream, rows=len(buf), files=len(paths))
        self._buf[key] = []
        self._first_add.pop(key, None)
        return paths


class RestPoller:
    """Generic fixed-interval async poller: fetch -> normalize -> sink.

    normalizer(payload, ts_recv) returns unified records; sink receives each
    record (typically BatchWriter.add). Fetch, normalize and sink errors are
    all logged and skipped — polling never dies on a single bad response
    (Binance /futures/data endpoints can return error objects with HTTP 200).
    """

    def __init__(
        self,
        interval_s: float,
        fetch: Callable[[], Awaitable[Any]],
        normalizer: Callable[[Any, int], list[Record]],
        sink: Callable[[Record], object],
        *,
        clock: Callable[[], int] = now_ns,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self._interval_s = interval_s
        self._fetch = fetch
        self._normalizer = normalizer
        self._sink = sink
        self._clock = clock
        self._sleep = sleep
        self._on_error = on_error
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    async def run(self, n_polls: int | None = None) -> int:
        """Poll until stopped (or n_polls times); returns records emitted."""
        emitted = 0
        polls = 0
        while not self._stopped and (n_polls is None or polls < n_polls):
            try:
                payload = await self._fetch()
                ts_recv = self._clock()
                for rec in self._normalizer(payload, ts_recv):
                    self._sink(rec)
                    emitted += 1
            except Exception as exc:  # noqa: BLE001 - poller must survive bad responses
                log.warning("poll_error", error=str(exc))
                if self._on_error is not None:
                    self._on_error(exc)
            polls += 1
            if self._stopped or (n_polls is not None and polls >= n_polls):
                break
            await self._sleep(self._interval_s)
        return emitted
