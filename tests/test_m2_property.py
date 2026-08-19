"""M2 property tests (hypothesis) and out-of-order fuzzing.

Two valid-sequence generators: core.synth wrapped with drawn parameters, and a
hand-rolled strategy that builds diffs level-by-level on an integer tick grid.
Fuzzing shuffles or drops diffs and asserts the book demands a resync instead
of continuing silently; resync tests verify the book recovers after a fresh
snapshot.
"""

from __future__ import annotations

import hashlib
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_system.book import (
    BookReplayer,
    NeedsResync,
    OrderBook,
    mean_reverting_book_stream,
)
from trading_system.core.schema import BookSnapshot, DepthDiff
from trading_system.core.synth import synth_book_stream

EX = "binance_usdm"
SYM = "BTCUSDT"
TICK = 0.25  # binary-exact tick so float price keys always match


def top_digest(bids, asks) -> str:
    payload = "|".join(
        f"{tag}:{p!r}:{q!r}"
        for tag, levels in (("B", bids), ("A", asks))
        for p, q in levels
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def assert_valid(book: OrderBook) -> None:
    bids, asks = book.top_n(10_000)
    assert all(q > 0 for _, q in bids)
    assert all(q > 0 for _, q in asks)
    if bids and asks:
        assert bids[0][0] < asks[0][0], "book crossed"


# -- strategy 1: core.synth with drawn parameters -----------------------------

synth_params = st.fixed_dictionaries(
    {
        "seed": st.integers(0, 2**31 - 1),
        "n_diffs": st.integers(0, 200),
        "n_levels": st.integers(3, 40),
        "mid0": st.floats(1_000.0, 100_000.0, allow_nan=False, allow_infinity=False),
    }
)


@settings(max_examples=40, deadline=None)
@given(params=synth_params)
def test_synth_streams_never_cross_never_negative(params):
    stream = synth_book_stream(**params)
    replayer = BookReplayer(stream.snapshot, stream.diffs)
    n_applied = 0
    for _, book in replayer.replay():
        assert_valid(book)
        n_applied += 1
    assert n_applied == len(stream.diffs)


@settings(max_examples=20, deadline=None)
@given(params=synth_params)
def test_replay_is_deterministic(params):
    stream = synth_book_stream(**params)
    replayer = BookReplayer(stream.snapshot, stream.diffs)
    d1 = top_digest(*replayer.final_book().top_n(50))
    d2 = top_digest(*replayer.final_book().top_n(50))
    stream_again = synth_book_stream(**params)
    d3 = top_digest(*BookReplayer(stream_again.snapshot, stream_again.diffs).final_book().top_n(50))
    assert d1 == d2 == d3


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), n_diffs=st.integers(0, 300))
def test_local_synth_streams_are_valid(seed, n_diffs):
    snapshot, diffs = mean_reverting_book_stream(n_diffs=n_diffs, seed=seed)
    replayer = BookReplayer(snapshot, diffs)
    n_applied = 0
    for _, book in replayer.replay():
        assert_valid(book)
        n_applied += 1
    assert n_applied == len(diffs)


# -- strategy 2: hand-built valid diff sequences on an integer tick grid ------


@st.composite
def valid_stream(draw) -> tuple[BookSnapshot, list[DepthDiff]]:
    mid_ticks = draw(st.integers(1_000, 100_000))
    n_lv = draw(st.integers(2, 10))
    bids = {TICK * (mid_ticks - 1 - i): 1.0 + i for i in range(n_lv)}
    asks = {TICK * (mid_ticks + 1 + i): 1.0 + i for i in range(n_lv)}
    snap = BookSnapshot(
        exchange=EX,
        symbol=SYM,
        ts_event=1_000_000_000,
        ts_recv=1_000_000_001,
        last_update_id=100,
        bids=tuple(sorted(bids.items(), key=lambda x: -x[0])),
        asks=tuple(sorted(asks.items())),
    )
    diffs: list[DepthDiff] = []
    prev_final = snap.last_update_id
    ts = snap.ts_event
    n_diffs = draw(st.integers(0, 30))
    for _ in range(n_diffs):
        ts += draw(st.integers(1, 10**9))
        first = prev_final + 1
        final = first + draw(st.integers(0, 5))
        dbids: list[tuple[float, float]] = []
        dasks: list[tuple[float, float]] = []
        for _ in range(draw(st.integers(0, 3))):
            side = draw(st.sampled_from(["bid", "ask"]))
            book_side = bids if side == "bid" else asks
            remove = draw(st.booleans()) and book_side
            if remove:
                price = draw(st.sampled_from(sorted(book_side)))
                qty = 0.0
                book_side.pop(price, None)
            else:
                # additions stay strictly inside the opposite best: never cross
                best_ask = min(asks) if asks else TICK * (mid_ticks + 10**6)
                best_bid = max(bids) if bids else TICK
                if side == "bid":
                    hi = int(round(best_ask / TICK)) - 1
                    k = draw(st.integers(max(1, hi - 20), hi))
                else:
                    lo = int(round(best_bid / TICK)) + 1
                    k = draw(st.integers(lo, lo + 20))
                price = TICK * k
                qty = draw(st.floats(0.001, 1e6, allow_nan=False, allow_infinity=False))
                book_side[price] = qty
            (dbids if side == "bid" else dasks).append((price, qty))
        diffs.append(
            DepthDiff(
                exchange=EX,
                symbol=SYM,
                ts_event=ts,
                ts_recv=ts + 1,
                first_update_id=first,
                final_update_id=final,
                prev_final_update_id=prev_final,
                bids=tuple(dbids),
                asks=tuple(dasks),
            )
        )
        prev_final = final
    return snap, diffs


@settings(max_examples=60, deadline=None)
@given(stream=valid_stream())
def test_hand_built_valid_sequences_keep_invariants(stream):
    snap, diffs = stream
    book = OrderBook()
    book.apply_snapshot(snap)
    for d in diffs:
        assert book.apply_diff(d)
        assert_valid(book)
    assert book.last_update_id == (diffs[-1].final_update_id if diffs else snap.last_update_id)


# -- fuzz: shuffled / dropped diffs must force a resync -----------------------


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    shuffle_seed=st.integers(0, 2**31 - 1),
)
def test_shuffled_diffs_raise_needs_resync(seed, shuffle_seed):
    stream = synth_book_stream(n_diffs=30, seed=seed)
    shuffled = stream.diffs.copy()
    random.Random(shuffle_seed).shuffle(shuffled)
    if [d.final_update_id for d in shuffled] == [d.final_update_id for d in stream.diffs]:
        return  # permutation happened to be identity: nothing to test
    book = OrderBook()
    book.apply_snapshot(stream.snapshot)
    with pytest.raises(NeedsResync):
        for d in shuffled:
            book.apply_diff(d)
    assert not book.synced


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    drop_at=st.integers(0, 28),
)
def test_dropped_diff_raises_needs_resync(seed, drop_at):
    stream = synth_book_stream(n_diffs=30, seed=seed)
    kept = [d for i, d in enumerate(stream.diffs) if i != drop_at]
    book = OrderBook()
    book.apply_snapshot(stream.snapshot)
    with pytest.raises(NeedsResync):
        for d in kept:
            book.apply_diff(d)
        raise AssertionError("gap was not detected")  # pragma: no cover
    assert not book.synced


def _shadow_state(stream, upto: int) -> tuple[dict, dict]:
    """Plain-dict state after applying diffs[0:upto] — independent of OrderBook."""
    bids = dict(stream.snapshot.bids)
    asks = dict(stream.snapshot.asks)
    for d in stream.diffs[:upto]:
        for side, levels in ((bids, d.bids), (asks, d.asks)):
            for p, q in levels:
                if q == 0.0:
                    side.pop(p, None)
                else:
                    side[p] = q
    return bids, asks


def _resync_snapshot(stream, upto: int, last_update_id: int) -> BookSnapshot:
    bids, asks = _shadow_state(stream, upto)
    ts = stream.diffs[upto - 1].ts_event
    return BookSnapshot(
        exchange=EX,
        symbol=SYM,
        ts_event=ts,
        ts_recv=ts + 1,
        last_update_id=last_update_id,
        bids=tuple(sorted(bids.items(), key=lambda x: -x[0])),
        asks=tuple(sorted(asks.items())),
    )


@pytest.mark.parametrize("mode", ["continuation", "straddle"])
def test_resync_after_gap_continues_correctly(mode):
    stream = synth_book_stream(n_diffs=200, seed=42)
    k = 100
    book = OrderBook()
    book.apply_snapshot(stream.snapshot)
    for d in stream.diffs[:k]:
        book.apply_diff(d)
    # a gap: diff k goes missing, diff k+1 arrives -> resync demanded
    with pytest.raises(NeedsResync):
        book.apply_diff(stream.diffs[k + 1])
    assert not book.synced
    # resync: fresh snapshot reflecting state after diffs[0:k+1]
    if mode == "continuation":
        last_id = stream.diffs[k].final_update_id  # next diff has pu == last_id
    else:
        last_id = stream.diffs[k + 1].first_update_id  # next diff straddles last_id
    book.apply_snapshot(_resync_snapshot(stream, k + 1, last_id))
    for d in stream.diffs[k + 1 :]:
        assert book.apply_diff(d)
        assert_valid(book)
    clean = BookReplayer(stream.snapshot, stream.diffs).final_book()
    assert book.top_n(10_000) == clean.top_n(10_000)


def test_resync_drops_stale_diffs_then_continues():
    stream = synth_book_stream(n_diffs=120, seed=7)
    j = 60
    snap = _resync_snapshot(stream, j + 1, stream.diffs[j].final_update_id)
    book = OrderBook()
    book.apply_snapshot(snap)
    applied = [book.apply_diff(d) for d in stream.diffs]
    # diffs entirely before the snapshot id are dropped; the boundary diff
    # (u == last_update_id) straddles and re-applies harmlessly (absolute qty)
    assert applied == [False] * j + [True] * (len(stream.diffs) - j)
    clean = BookReplayer(stream.snapshot, stream.diffs).final_book()
    assert book.top_n(10_000) == clean.top_n(10_000)
