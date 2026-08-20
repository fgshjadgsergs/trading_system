"""M2 stress: deep books, single-level pulsation, absent-level removals,
adversarial crossing, empty/one-sided contracts, replay determinism and
throughput, extreme price/qty magnitudes.

Sizes scale with env STRESS_SCALE (default 1). Perf assertions are loose
sanity floors; measured figures are printed as [stress-perf] lines (-s).
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

import pytest

from trading_system.book import BookInvariantError, BookReplayer, NeedsResync, OrderBook
from trading_system.core.schema import BookSnapshot, DepthDiff
from trading_system.core.synth import synth_book_stream

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))

EX = "binance_usdm"
SYM = "BTCUSDT"
SNAP_ID = 1_000


def _scaled(n: int, lo: int = 100) -> int:
    return max(lo, int(n * SCALE))


def _perf(msg: str) -> None:
    print(f"[stress-perf] {msg}")


def snap(bids, asks, last_update_id=SNAP_ID, ts=1_000_000) -> BookSnapshot:
    return BookSnapshot(
        exchange=EX,
        symbol=SYM,
        ts_event=ts,
        ts_recv=ts + 1,
        last_update_id=last_update_id,
        bids=tuple(bids),
        asks=tuple(asks),
    )


def diff(pu, first, final, bids=(), asks=(), ts=2_000_000) -> DepthDiff:
    return DepthDiff(
        exchange=EX,
        symbol=SYM,
        ts_event=ts,
        ts_recv=ts + 1,
        first_update_id=first,
        final_update_id=final,
        prev_final_update_id=pu,
        bids=tuple(bids),
        asks=tuple(asks),
    )


class _Seq:
    """Sequential uid helper: each next() is a valid continuation diff."""

    def __init__(self, start: int = SNAP_ID) -> None:
        self.uid = start

    def next(self, bids=(), asks=(), ts=2_000_000) -> DepthDiff:
        d = diff(self.uid, self.uid + 1, self.uid + 1, bids=bids, asks=asks, ts=ts)
        self.uid += 1
        return d


class TestDeepBook:
    def test_100k_levels_insert_read_delete(self):
        n = _scaled(100_000, lo=2_000)
        batch = 1_000
        mid = 50_000.0
        book = OrderBook()
        book.apply_snapshot(snap(bids=((mid - 1.0, 1.0),), asks=((mid + 1.0, 1.0),)))
        seq = _Seq()

        def level_batches(qty: float):
            for start in range(0, n, batch):
                m = min(batch, n - start)
                dbids = tuple(
                    (mid - 1.0 - (start + j + 1) * 0.01, qty and qty + j % 7) for j in range(m)
                )
                dasks = tuple(
                    (mid + 1.0 + (start + j + 1) * 0.01, qty and qty + j % 7) for j in range(m)
                )
                yield dbids, dasks

        t0 = time.perf_counter()
        for dbids, dasks in level_batches(1.0):
            assert book.apply_diff(seq.next(bids=dbids, asks=dasks))
        dt_ins = time.perf_counter() - t0
        assert book.n_levels() == (n + 1, n + 1)

        # reads on the deep book: best/spread stay the snapshot's inner levels
        reads = 200
        t0 = time.perf_counter()
        for _ in range(reads):
            assert book.best_bid == mid - 1.0
            assert book.best_ask == mid + 1.0
            assert book.spread == 2.0
        dt_read = time.perf_counter() - t0

        bids10, asks10 = book.top_n(10)
        assert bids10[0] == (mid - 1.0, 1.0) and asks10[0] == (mid + 1.0, 1.0)
        assert all(bids10[i][0] > bids10[i + 1][0] for i in range(9))
        assert all(asks10[i][0] < asks10[i + 1][0] for i in range(9))
        bd, ad = book.depth_within(0.005)
        assert bd > 0 and ad > 0

        # memory bounded by book size: dict table well under 192 B / level
        table_bytes = sys.getsizeof(book._bids) + sys.getsizeof(book._asks)
        assert table_bytes < 192 * 2 * (n + 1) + (1 << 20)

        t0 = time.perf_counter()
        for dbids, dasks in level_batches(0.0):
            assert book.apply_diff(seq.next(bids=dbids, asks=dasks))
        dt_del = time.perf_counter() - t0
        assert book.n_levels() == (1, 1)
        assert book.spread == 2.0

        ins_rate = 2 * n / dt_ins
        del_rate = 2 * n / dt_del
        read_rate = reads / dt_read
        _perf(
            f"deep book n={n}/side: insert {ins_rate:,.0f} lvl/s, "
            f"delete {del_rate:,.0f} lvl/s, best/spread {read_rate:,.0f} reads/s, "
            f"dict tables {table_bytes / 1e6:.1f} MB"
        )
        # loose floors (best_* is an O(n) scan: reads degrade linearly with depth)
        assert ins_rate > 20_000 and del_rate > 20_000
        assert read_rate > 10

    def test_level_pulsation_keeps_structures_stable(self):
        cycles = _scaled(50_000, lo=1_000)
        base = 8
        bids = tuple((99.0 - i, 1.0) for i in range(base))
        asks = tuple((101.0 + i, 1.0) for i in range(base))
        book = OrderBook()
        book.apply_snapshot(snap(bids=bids, asks=asks))
        seq = _Seq()
        # warm cycles settle the dict into its steady-state size class
        # (CPython steps one class up on the first reinsert-after-delete)
        for _ in range(64):
            book.apply_diff(seq.next(bids=((100.0, 5.0),)))
            book.apply_diff(seq.next(bids=((100.0, 0.0),)))
        base_size = sys.getsizeof(book._bids)
        t0 = time.perf_counter()
        for _ in range(cycles):
            assert book.apply_diff(seq.next(bids=((100.0, 5.0),)))
            assert book.apply_diff(seq.next(bids=((100.0, 0.0),)))
        dt = time.perf_counter() - t0
        assert book.n_levels() == (base, base)
        assert sys.getsizeof(book._bids) <= base_size  # no accumulated garbage
        assert sys.getsizeof(book._bids) < 4_096  # and absolutely tiny
        assert book.best_bid == 99.0 and book.best_ask == 101.0
        rate = 2 * cycles / dt
        _perf(f"pulsation {cycles} add/remove cycles: {rate:,.0f} diffs/s")
        assert rate > 2_000

    def test_mass_remove_absent_levels_is_noop(self):
        n_diffs = _scaled(20_000, lo=500)
        bids = tuple((float(99 - i), 1.0) for i in range(5))
        asks = tuple((float(101 + i), 1.0) for i in range(5))
        book = OrderBook()
        book.apply_snapshot(snap(bids=bids, asks=asks))
        before = book.top_n(1_000)
        seq = _Seq()
        for i in range(n_diffs):
            # half-integer prices never exist in the book; both sides at once
            p = 50.5 + (i % 1000)
            assert book.apply_diff(seq.next(bids=((p, 0.0),), asks=((p + 2000.0, 0.0),)))
        assert book.synced
        assert book.top_n(1_000) == before
        assert book.last_update_id == seq.uid


class TestCrossingContract:
    """Actual contract: a diff crossing the book raises BookInvariantError,
    poisons the book (accessors keep the tainted crossed state, last_update_id
    does not advance) and every further diff raises until a fresh snapshot."""

    def _poisoned_book(self, cross_bids=(), cross_asks=()):
        book = OrderBook()
        book.apply_snapshot(snap(bids=((100.0, 1.0),), asks=((101.0, 1.0),)))
        with pytest.raises(BookInvariantError):
            book.apply_diff(diff(SNAP_ID, SNAP_ID + 1, SNAP_ID + 2,
                                 bids=cross_bids, asks=cross_asks))
        return book

    @pytest.mark.parametrize(
        "cross_bids,cross_asks",
        [
            (((101.5, 1.0),), ()),  # bid through the ask
            (((101.0, 1.0),), ()),  # bid equal to the ask
            ((), ((99.5, 1.0),)),  # ask under the bid
            (((100.5, 1.0),), ((100.5, 2.0),)),  # both meet inside the spread
        ],
    )
    def test_crossing_diff_poisons_book(self, cross_bids, cross_asks):
        book = self._poisoned_book(cross_bids, cross_asks)
        assert not book.synced
        assert book.last_update_id == SNAP_ID  # crossing diff never commits its id
        # tainted state stays readable (documented weakness: accessors do not
        # check synced), so callers must gate reads on book.synced themselves
        assert book.best_bid >= book.best_ask
        with pytest.raises(NeedsResync):
            book.apply_diff(diff(SNAP_ID + 2, SNAP_ID + 3, SNAP_ID + 4))

    def test_snapshot_recovers_after_crossing(self):
        book = self._poisoned_book(cross_bids=((101.5, 1.0),))
        book.apply_snapshot(snap(bids=((100.0, 1.0),), asks=((101.0, 1.0),), last_update_id=2000))
        assert book.synced and book.spread == 1.0
        assert book.apply_diff(diff(2000, 2001, 2001, bids=((99.5, 2.0),)))

    def test_many_crossing_attempts_never_corrupt_silently(self):
        # every attempt must raise; the book never absorbs a crossed diff quietly
        for k in range(_scaled(2_000, lo=100)):
            book = OrderBook()
            book.apply_snapshot(snap(bids=((100.0, 1.0),), asks=((101.0, 1.0),)))
            price = 101.0 + (k % 50) * 0.25
            with pytest.raises(BookInvariantError):
                book.apply_diff(diff(SNAP_ID, SNAP_ID + 1, SNAP_ID + 1, bids=((price, 1.0),)))
            assert not book.synced


class TestEmptyAndOneSided:
    """Contract: missing side -> None (never a fake number), zero depth,
    empty top_n; an empty snapshot is accepted and synced."""

    def test_empty_snapshot_contract(self):
        book = OrderBook()
        book.apply_snapshot(snap(bids=(), asks=()))
        assert book.synced
        assert book.best_bid is None and book.best_ask is None
        assert book.mid is None and book.spread is None
        assert book.top_n(10) == ((), ())
        assert book.depth_within(0.005) == (0.0, 0.0)
        assert book.n_levels() == (0, 0)

    @pytest.mark.parametrize("side", ["bids", "asks"])
    def test_one_sided_book_contract(self, side):
        levels = ((100.0, 1.0), (99.0, 2.0)) if side == "bids" else ((100.0, 1.0), (101.0, 2.0))
        book = OrderBook()
        book.apply_snapshot(snap(**{side: levels, ("asks" if side == "bids" else "bids"): ()}))
        assert book.synced
        assert (book.best_bid, book.best_ask)[side == "bids"] is None
        assert book.mid is None and book.spread is None
        assert book.depth_within(0.01) == (0.0, 0.0)  # no mid -> no band

    def test_diff_emptying_side_flips_contract(self):
        book = OrderBook()
        book.apply_snapshot(snap(bids=((100.0, 1.0),), asks=((101.0, 1.0), (102.0, 1.0))))
        assert book.apply_diff(
            diff(SNAP_ID, SNAP_ID + 1, SNAP_ID + 1, asks=((101.0, 0.0), (102.0, 0.0)))
        )
        assert book.best_ask is None and book.mid is None and book.spread is None
        # refilling the side restores numeric accessors
        assert book.apply_diff(diff(SNAP_ID + 1, SNAP_ID + 2, SNAP_ID + 2, asks=((103.0, 1.0),)))
        assert book.spread == 3.0


class TestReplayDeterminism:
    def test_100k_events_bit_identical_and_throughput(self):
        n = _scaled(100_000, lo=2_000)
        stream = synth_book_stream(n_diffs=n, seed=42)

        def run() -> tuple[str, int, float]:
            h = hashlib.sha256()
            t0 = time.perf_counter()
            count = 0
            book = None
            for ts, book in BookReplayer(stream.snapshot, stream.diffs).replay():
                h.update(
                    f"{ts}:{book.last_update_id}:{book.best_bid!r}:{book.best_ask!r}".encode()
                )
                count += 1
            dt = time.perf_counter() - t0
            bids, asks = book.top_n(10_000)
            h.update(repr((bids, asks)).encode())
            return h.hexdigest(), count, dt

        d1, c1, t1 = run()
        d2, c2, t2 = run()
        assert d1 == d2  # bit-for-bit across two full replays
        assert c1 == c2 == n
        rate = n / min(t1, t2)
        _perf(f"replay {n} diffs twice: {rate:,.0f} diffs/s (incl. per-step hashing)")
        assert rate > 10_000


class TestExtremeMagnitudes:
    def test_tiny_prices_keep_precision(self):
        tick, p0, n = 1e-10, 1e-8, 50
        bids = tuple((p0 - tick * (i + 1), 1.0) for i in range(n))
        asks = tuple((p0 + tick * (i + 1), 1.0) for i in range(n))
        book = OrderBook()
        book.apply_snapshot(snap(bids=bids, asks=asks))
        assert book.best_bid < p0 < book.best_ask
        assert book.spread > 0
        assert book.spread == pytest.approx(2 * tick, rel=1e-6)
        assert book.mid == pytest.approx(p0, rel=1e-12)
        # band wider than the whole ladder: every level must be counted once
        bd, ad = book.depth_within(0.9)
        assert (bd, ad) == (float(n), float(n))

    def test_huge_prices_and_quantities(self):
        p0, q = 1e12, 1e12
        bids = tuple((p0 - (i + 1), q) for i in range(100))
        asks = tuple((p0 + (i + 1), q) for i in range(100))
        book = OrderBook()
        book.apply_snapshot(snap(bids=bids, asks=asks))
        assert book.spread == 2.0  # integers below 2**53 stay exact
        assert book.mid == p0
        bd, ad = book.depth_within(0.005)
        assert bd == 100 * q and ad == 100 * q  # no overflow, exact sum

    def test_mixed_extreme_spread(self):
        book = OrderBook()
        book.apply_snapshot(snap(bids=((1e-8, 1e12),), asks=((1e12, 1e-8),)))
        assert book.spread == pytest.approx(1e12)
        assert book.spread > 0 and book.mid == pytest.approx(5e11)

    def test_nonfinite_levels_rejected(self):
        book = OrderBook()
        book.apply_snapshot(snap(bids=((100.0, 1.0),), asks=((101.0, 1.0),)))
        for bad_bids in (((float("inf"), 1.0),), ((float("nan"), 1.0),), ((99.0, float("nan")),)):
            with pytest.raises(BookInvariantError):
                book.apply_diff(diff(SNAP_ID, SNAP_ID + 1, SNAP_ID + 1, bids=bad_bids))
            book.apply_snapshot(snap(bids=((100.0, 1.0),), asks=((101.0, 1.0),)))
