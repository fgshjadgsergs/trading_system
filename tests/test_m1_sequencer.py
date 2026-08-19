"""M1: depth sequencer — snapshot sync, U/u/pu contiguity, gap + resync."""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from trading_system.collectors.sequencer import DepthSequencer, GapEvent
from trading_system.core.schema import BookSnapshot, DepthDiff
from trading_system.core.synth import SynthBookStream, synth_book_stream

SYMBOL = "BTCUSDT"


def replay_snapshot(stream: SynthBookStream, upto: int) -> BookSnapshot:
    """True book snapshot after applying diffs[0..upto] (test-local replay)."""
    bids = dict(stream.snapshot.bids)
    asks = dict(stream.snapshot.asks)
    for d in stream.diffs[: upto + 1]:
        for p, q in d.bids:
            if q == 0.0:
                bids.pop(p, None)
            else:
                bids[p] = q
        for p, q in d.asks:
            if q == 0.0:
                asks.pop(p, None)
            else:
                asks[p] = q
    last = stream.diffs[upto]
    return BookSnapshot(
        exchange=stream.snapshot.exchange,
        symbol=stream.snapshot.symbol,
        ts_event=last.ts_event,
        ts_recv=last.ts_recv,
        last_update_id=last.final_update_id,
        bids=tuple(sorted(bids.items(), key=lambda x: -x[0])),
        asks=tuple(sorted(asks.items())),
    )


def assert_contiguous(applied: list[DepthDiff], start_u: int) -> None:
    last = start_u
    for d in applied:
        assert d.prev_final_update_id == last or d.first_update_id <= last <= d.final_update_id
        last = d.final_update_id


# --------------------------------------------------------------------------- #
def test_buffered_then_snapshot_happy_path():
    stream = synth_book_stream(n_diffs=50, seed=1)
    seq = DepthSequencer(SYMBOL)
    for d in stream.diffs[:10]:
        assert seq.add_diff(d) == []  # buffering until a snapshot arrives
    assert not seq.synced and seq.buffered == 10

    ready = seq.set_snapshot(stream.snapshot)
    assert [d.final_update_id for d in ready] == [
        d.final_update_id for d in stream.diffs[:10]
    ]
    applied = list(ready)
    for d in stream.diffs[10:]:
        out = seq.add_diff(d)
        assert out == [d]
        applied.extend(out)
    assert seq.synced and seq.gaps == []
    assert_contiguous(applied, stream.snapshot.last_update_id)


def test_diffs_older_than_snapshot_are_dropped():
    stream = synth_book_stream(n_diffs=30, seed=2)
    seq = DepthSequencer(SYMBOL)
    for d in stream.diffs:
        seq.add_diff(d)
    snap = replay_snapshot(stream, upto=5)
    ready = seq.set_snapshot(snap)
    assert [d.final_update_id for d in ready] == [
        d.final_update_id for d in stream.diffs[6:]
    ]
    assert seq.synced and seq.gaps == []


def _diff(first: int, final: int, prev: int, ts: int = 1_000_000) -> DepthDiff:
    return DepthDiff(
        exchange="binance_usdm",
        symbol=SYMBOL,
        ts_event=ts,
        ts_recv=ts + 1,
        first_update_id=first,
        final_update_id=final,
        prev_final_update_id=prev,
        bids=((50_000.0, 1.0),),
        asks=((50_001.0, 1.0),),
    )


def test_first_event_straddling_snapshot_id_is_accepted():
    snap = BookSnapshot(
        exchange="binance_usdm",
        symbol=SYMBOL,
        ts_event=1_000_000,
        ts_recv=1_000_001,
        last_update_id=1005,
        bids=((50_000.0, 1.0),),
        asks=((50_001.0, 1.0),),
    )
    seq = DepthSequencer(SYMBOL)
    seq.set_snapshot(snap)
    assert seq.add_diff(_diff(990, 1003, 989)) == []  # fully covered -> dropped
    straddler = _diff(1001, 1010, 1000)  # U <= 1005 <= u
    assert seq.add_diff(straddler) == [straddler]
    nxt = _diff(1011, 1015, 1010)  # pu == previous u
    assert seq.add_diff(nxt) == [nxt]
    assert seq.gaps == [] and seq.synced


def test_dropped_diffs_flag_gap_and_resync_reestablishes_contiguity():
    stream = synth_book_stream(n_diffs=100, seed=4)
    gaps_seen: list[GapEvent] = []
    resyncs: list[int] = []
    seq = DepthSequencer(SYMBOL, on_gap=gaps_seen.append, on_resync=lambda: resyncs.append(1))
    seq.set_snapshot(stream.snapshot)

    j, m = 40, 5  # diffs j..j+m-1 lost in a simulated disconnect
    applied: list[DepthDiff] = []
    for i, d in enumerate(stream.diffs):
        if j <= i < j + m:
            continue
        applied.extend(seq.add_diff(d))

    assert len(seq.gaps) == 1 and gaps_seen == seq.gaps and resyncs == [1]
    gap = seq.gaps[0]
    assert gap.symbol == SYMBOL
    assert gap.expected == stream.diffs[j - 1].final_update_id
    assert gap.got == stream.diffs[j + m].prev_final_update_id
    assert not seq.synced
    assert [d.final_update_id for d in applied] == [
        d.final_update_id for d in stream.diffs[:j]
    ]  # nothing after the hole was released silently

    # resync: fresh snapshot that already includes the lost diffs
    ready = seq.set_snapshot(replay_snapshot(stream, upto=j + m - 1))
    assert seq.synced and len(seq.gaps) == 1  # no new gaps
    applied.extend(ready)
    assert [d.final_update_id for d in applied] == [
        d.final_update_id for i, d in enumerate(stream.diffs) if not (j <= i < j + m)
    ]
    assert_contiguous(applied[len(stream.diffs[:j]) :], stream.diffs[j + m - 1].final_update_id)


def test_too_old_snapshot_triggers_gap_and_second_resync_works():
    stream = synth_book_stream(n_diffs=40, seed=5)
    resyncs: list[int] = []
    seq = DepthSequencer(SYMBOL, on_resync=lambda: resyncs.append(1))
    for d in stream.diffs[10:20]:
        seq.add_diff(d)
    # original snapshot is too old: diffs 0..9 are missing between it and buffer
    seq.set_snapshot(stream.snapshot)
    assert not seq.synced
    assert len(seq.gaps) == 1 and resyncs == [1]
    assert seq.gaps[0].expected == stream.snapshot.last_update_id
    assert seq.gaps[0].got == stream.diffs[10].first_update_id

    ready = seq.set_snapshot(replay_snapshot(stream, upto=9))
    assert seq.synced
    assert [d.final_update_id for d in ready] == [
        d.final_update_id for d in stream.diffs[10:20]
    ]


@settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    j=st.integers(min_value=1, max_value=140),
    m=st.integers(min_value=1, max_value=25),
)
def test_property_random_drops_always_flag_gap_then_resync(seed: int, j: int, m: int):
    """Any dropped chunk is flagged exactly once and never silently skipped."""
    stream = synth_book_stream(n_diffs=170, seed=seed)
    seq = DepthSequencer(SYMBOL)
    seq.set_snapshot(stream.snapshot)
    applied: list[DepthDiff] = []
    for i, d in enumerate(stream.diffs):
        if j <= i < j + m:
            continue
        applied.extend(seq.add_diff(d))

    assert len(seq.gaps) == 1
    assert not seq.synced
    assert len(applied) == j  # everything before the hole, nothing after

    applied.extend(seq.set_snapshot(replay_snapshot(stream, upto=j + m - 1)))
    for d in stream.diffs[170:]:
        applied.extend(seq.add_diff(d))
    assert seq.synced and len(seq.gaps) == 1
    expected_ids = [
        d.final_update_id for i, d in enumerate(stream.diffs) if not (j <= i < j + m)
    ]
    assert [d.final_update_id for d in applied] == expected_ids
