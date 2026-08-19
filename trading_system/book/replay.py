"""Deterministic replay of a recorded (snapshot, diff stream) into book states.

Accepts core dataclasses directly or recorded polars frames in the
``book_snapshot`` / ``depth_diff`` schemas from ``core.schema.POLARS_SCHEMAS``.
All timestamps are UTC nanoseconds.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import polars as pl

from trading_system.book.order_book import OrderBook
from trading_system.core.schema import (
    BookSnapshot,
    DepthDiff,
    PriceLevel,
    records_to_frame,
)

GRID_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Int64,
    "side": pl.Utf8,
    "price": pl.Float64,
    "qty": pl.Float64,
}

METRICS_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Int64,
    "mid": pl.Float64,
    "spread": pl.Float64,
    "bid_depth": pl.Float64,
    "ask_depth": pl.Float64,
}


def _row_levels(levels: Iterable[dict[str, float]]) -> tuple[PriceLevel, ...]:
    return tuple((float(lv["price"]), float(lv["qty"])) for lv in levels)


def snapshot_from_row(row: dict[str, Any]) -> BookSnapshot:
    """One ``book_snapshot``-schema frame row -> BookSnapshot dataclass."""
    return BookSnapshot(
        exchange=row["exchange"],
        symbol=row["symbol"],
        ts_event=int(row["ts_event"]),
        ts_recv=int(row["ts_recv"]),
        last_update_id=int(row["last_update_id"]),
        bids=_row_levels(row["bids"]),
        asks=_row_levels(row["asks"]),
    )


def diff_from_row(row: dict[str, Any]) -> DepthDiff:
    """One ``depth_diff``-schema frame row -> DepthDiff dataclass."""
    return DepthDiff(
        exchange=row["exchange"],
        symbol=row["symbol"],
        ts_event=int(row["ts_event"]),
        ts_recv=int(row["ts_recv"]),
        first_update_id=int(row["first_update_id"]),
        final_update_id=int(row["final_update_id"]),
        prev_final_update_id=int(row["prev_final_update_id"]),
        bids=_row_levels(row["bids"]),
        asks=_row_levels(row["asks"]),
    )


class BookReplayer:
    """Replays one snapshot plus its diff stream; all queries re-run the replay
    from the snapshot, so every call is deterministic and side-effect free."""

    def __init__(self, snapshot: BookSnapshot, diffs: Iterable[DepthDiff]) -> None:
        self.snapshot = snapshot
        self.diffs: Sequence[DepthDiff] = tuple(diffs)

    @classmethod
    def from_frames(cls, snapshot_frame: pl.DataFrame, diffs_frame: pl.DataFrame) -> BookReplayer:
        """Build from recorded frames (schemas ``book_snapshot`` / ``depth_diff``).

        The first snapshot row is the sync point; diff rows are replayed in
        frame order — the replayer never reorders a recorded stream.
        """
        if snapshot_frame.is_empty():
            raise ValueError("snapshot frame is empty")
        snap = snapshot_from_row(snapshot_frame.row(0, named=True))
        diffs = [diff_from_row(r) for r in diffs_frame.iter_rows(named=True)]
        return cls(snap, diffs)

    def replay(self) -> Iterator[tuple[int, OrderBook]]:
        """Yield (ts_event, book) after each applied diff.

        Diffs dropped as pre-snapshot are not yielded. The same live OrderBook
        object is yielded each time — copy (e.g. ``book.top_n``) to keep state.
        """
        book = OrderBook()
        book.apply_snapshot(self.snapshot)
        for diff in self.diffs:
            if book.apply_diff(diff):
                yield diff.ts_event, book

    def final_book(self) -> OrderBook:
        """The book after the whole stream is applied."""
        book = OrderBook()
        book.apply_snapshot(self.snapshot)
        for diff in self.diffs:
            book.apply_diff(diff)
        return book

    def state_at(
        self, t: int, n: int = 50
    ) -> tuple[tuple[PriceLevel, ...], tuple[PriceLevel, ...]]:
        """Top-n (bids, asks) as of moment t (all events with ts_event <= t)."""
        if t < self.snapshot.ts_event:
            raise ValueError(f"t={t} precedes snapshot ts_event={self.snapshot.ts_event}")
        book = OrderBook()
        book.apply_snapshot(self.snapshot)
        for diff in self.diffs:
            if diff.ts_event > t:
                break
            book.apply_diff(diff)
        return book.top_n(n)

    def _walk_grid(self, interval_ns: int) -> Iterator[tuple[int, OrderBook]]:
        """Yield (grid_ts, book-as-of-grid_ts) on a regular grid from the
        snapshot time to the last diff time inclusive."""
        if interval_ns <= 0:
            raise ValueError("interval_ns must be positive")
        book = OrderBook()
        book.apply_snapshot(self.snapshot)
        t = self.snapshot.ts_event
        end = self.diffs[-1].ts_event if self.diffs else self.snapshot.ts_event
        for diff in self.diffs:
            while t < diff.ts_event and t <= end:
                yield t, book
                t += interval_ns
            book.apply_diff(diff)
        while t <= end:
            yield t, book
            t += interval_ns

    def sample_grid(self, interval_ns: int, n: int = 50) -> pl.DataFrame:
        """Top-n book states on a regular time grid, long format for heatmaps.

        Columns: ts (UTC ns, grid point), side ("bid"/"ask"), price, qty.
        """
        rows: list[tuple[int, str, float, float]] = []
        for t, book in self._walk_grid(interval_ns):
            bids, asks = book.top_n(n)
            rows.extend((t, "bid", p, q) for p, q in bids)
            rows.extend((t, "ask", p, q) for p, q in asks)
        return pl.DataFrame(rows, schema=GRID_SCHEMA, orient="row")

    def sample_metrics(self, interval_ns: int, depth_pct: float = 0.005) -> pl.DataFrame:
        """Mid, spread and depth within +/-depth_pct of mid on a regular grid."""
        rows: list[tuple[int, float | None, float | None, float, float]] = []
        for t, book in self._walk_grid(interval_ns):
            bid_depth, ask_depth = book.depth_within(depth_pct)
            rows.append((t, book.mid, book.spread, bid_depth, ask_depth))
        return pl.DataFrame(rows, schema=METRICS_SCHEMA, orient="row")


def stream_frames(
    snapshot: BookSnapshot, diffs: Sequence[DepthDiff]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Dataclass stream -> (snapshot frame, diffs frame) in the core schemas."""
    snap_df = records_to_frame([snapshot], "book_snapshot")
    diff_df = records_to_frame(list(diffs), "depth_diff")
    return snap_df, diff_df
