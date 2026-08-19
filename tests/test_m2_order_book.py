"""M2 unit tests: OrderBook semantics, sequencing, invariants, accessors."""

from __future__ import annotations

import pytest

from trading_system.book import BookInvariantError, NeedsResync, OrderBook
from trading_system.core.schema import BookSnapshot, DepthDiff

EX = "binance_usdm"
SYM = "BTCUSDT"


def snap(
    bids=((100.0, 1.0), (99.0, 2.0)),
    asks=((101.0, 1.5), (102.0, 3.0)),
    last_update_id=1000,
    ts=1_000_000,
) -> BookSnapshot:
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


def synced_book() -> OrderBook:
    b = OrderBook()
    b.apply_snapshot(snap())
    # first diff continues the snapshot exactly (pu == last_update_id)
    assert b.apply_diff(diff(1000, 1001, 1005))
    return b


class TestSnapshot:
    def test_snapshot_sets_state(self):
        b = OrderBook()
        b.apply_snapshot(snap())
        assert b.synced
        assert b.last_update_id == 1000
        assert b.best_bid == 100.0
        assert b.best_ask == 101.0
        assert b.mid == 100.5
        assert b.spread == pytest.approx(1.0)
        assert b.n_levels() == (2, 2)

    def test_crossed_snapshot_rejected(self):
        b = OrderBook()
        with pytest.raises(BookInvariantError):
            b.apply_snapshot(snap(bids=((101.0, 1.0),), asks=((101.0, 1.0),)))

    def test_zero_qty_snapshot_level_rejected(self):
        b = OrderBook()
        with pytest.raises(BookInvariantError):
            b.apply_snapshot(snap(bids=((100.0, 0.0),)))

    def test_snapshot_clears_poisoned_state(self):
        b = synced_book()
        with pytest.raises(NeedsResync):
            b.apply_diff(diff(9999, 10000, 10001))
        assert not b.synced
        b.apply_snapshot(snap(last_update_id=5000))
        assert b.synced
        assert b.apply_diff(diff(5000, 5001, 5002, bids=((99.5, 4.0),)))


class TestSequencing:
    def test_diff_without_snapshot_raises(self):
        b = OrderBook()
        with pytest.raises(NeedsResync):
            b.apply_diff(diff(1000, 1001, 1002))

    def test_first_diff_straddle_accepted(self):
        b = OrderBook()
        b.apply_snapshot(snap())
        # U <= last_update_id <= u with pu != last: the futures straddle rule
        assert b.apply_diff(diff(990, 995, 1003, bids=((100.0, 5.0),)))
        assert b.last_update_id == 1003

    def test_first_diff_exact_continuation_accepted(self):
        b = OrderBook()
        b.apply_snapshot(snap())
        assert b.apply_diff(diff(1000, 1001, 1002))
        assert b.last_update_id == 1002

    def test_pre_snapshot_diff_dropped_silently(self):
        b = OrderBook()
        b.apply_snapshot(snap())
        assert b.apply_diff(diff(900, 901, 950, bids=((1.0, 1.0),))) is False
        assert b.best_bid == 100.0  # nothing applied
        assert b.synced
        # a later straddling diff still syncs
        assert b.apply_diff(diff(950, 951, 1001))

    def test_first_diff_gap_raises(self):
        b = OrderBook()
        b.apply_snapshot(snap())
        with pytest.raises(NeedsResync):
            b.apply_diff(diff(1005, 1006, 1010))
        assert not b.synced

    def test_next_diff_wrong_pu_raises(self):
        b = synced_book()
        with pytest.raises(NeedsResync):
            b.apply_diff(diff(1004, 1006, 1007))
        assert not b.synced

    def test_duplicate_diff_after_sync_raises(self):
        b = synced_book()
        with pytest.raises(NeedsResync):
            b.apply_diff(diff(1000, 1001, 1005))  # replayed event: pu != 1005

    def test_poisoned_book_refuses_further_diffs(self):
        b = synced_book()
        with pytest.raises(NeedsResync):
            b.apply_diff(diff(1, 2, 3))
        with pytest.raises(NeedsResync):
            b.apply_diff(diff(1005, 1006, 1007))  # would be valid on a healthy book


class TestAbsoluteQty:
    def test_set_overwrite_remove(self):
        b = synced_book()
        assert b.apply_diff(diff(1005, 1006, 1006, bids=((100.0, 7.0),)))
        assert dict(b.top_n(10)[0])[100.0] == 7.0
        assert b.apply_diff(diff(1006, 1007, 1007, bids=((100.0, 0.0),)))
        assert b.best_bid == 99.0
        # removing an absent level is a no-op, not an error
        assert b.apply_diff(diff(1007, 1008, 1008, bids=((55.0, 0.0),)))
        assert b.n_levels() == (1, 2)

    def test_negative_qty_raises(self):
        b = synced_book()
        with pytest.raises(BookInvariantError):
            b.apply_diff(diff(1005, 1006, 1006, asks=((101.0, -1.0),)))
        assert not b.synced

    def test_crossing_diff_raises(self):
        b = synced_book()
        with pytest.raises(BookInvariantError):
            b.apply_diff(diff(1005, 1006, 1006, bids=((101.5, 1.0),)))
        assert not b.synced

    def test_book_can_empty_one_side(self):
        b = synced_book()
        assert b.apply_diff(diff(1005, 1006, 1006, asks=((101.0, 0.0), (102.0, 0.0))))
        assert b.best_ask is None
        assert b.mid is None
        assert b.spread is None
        assert b.depth_within(0.005) == (0.0, 0.0)


class TestAccessors:
    def test_top_n_sorted_best_first(self):
        b = synced_book()
        b.apply_diff(diff(1005, 1006, 1006, bids=((98.0, 1.0),), asks=((103.0, 1.0),)))
        bids, asks = b.top_n(2)
        assert bids == ((100.0, 1.0), (99.0, 2.0))
        assert asks == ((101.0, 1.5), (102.0, 3.0))
        bids3, asks3 = b.top_n(10)
        assert len(bids3) == 3 and len(asks3) == 3

    def test_depth_within_band(self):
        b = OrderBook()
        b.apply_snapshot(
            snap(
                bids=((100.0, 1.0), (99.8, 2.0), (90.0, 100.0)),
                asks=((100.4, 3.0), (100.6, 4.0), (120.0, 100.0)),
            )
        )
        # mid = 100.2, +/-0.5% band = [99.699, 100.701]
        bid_qty, ask_qty = b.depth_within(0.005)
        assert bid_qty == pytest.approx(3.0)
        assert ask_qty == pytest.approx(7.0)
        # wide band captures everything
        assert b.depth_within(1.0) == (pytest.approx(103.0), pytest.approx(107.0))
