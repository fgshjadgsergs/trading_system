"""Stress: M1 collectors — BatchWriter flood, adversarial sequencer, ws reconnect
storms, recorder e2e (sequencer+writer+gap log), quality report on pathological lakes.

Scale heavy sizes with env STRESS_SCALE (default 1).
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import time
import tracemalloc

import pytest
from structlog.testing import capture_logs

from trading_system.collectors.quality import (
    daily_quality_report,
    read_gaps,
    write_gap_events,
)
from trading_system.collectors.recorder import BatchWriter
from trading_system.collectors.sequencer import DepthSequencer, GapEvent
from trading_system.collectors.ws_client import (
    CONNECTED,
    DISCONNECTED,
    GAP_SUSPECTED,
    ReconnectingWSClient,
    TransportExhausted,
    backoff_delay,
)
from trading_system.core.io import read_stream
from trading_system.core.schema import (
    BookSnapshot,
    DepthDiff,
    Liquidation,
    MarkPrice,
    OpenInterest,
    Side,
    Trade,
)
from trading_system.core.timeutils import NS_PER_MS, NS_PER_S

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))
SEED = 7
EXCHANGE = "binance_usdm"
SYMBOL = "BTCUSDT"
HOUR_NS = 3_600 * NS_PER_S
DAY_NS = 24 * HOUR_NS
T0 = (1_755_600_000 * NS_PER_S // HOUR_NS) * HOUR_NS  # exact hour boundary


# --------------------------------------------------------------------------- #
# helpers (conventions of tests/test_m1_recorder.py / test_m1_ws_client.py)
# --------------------------------------------------------------------------- #
class FakeClock:
    def __init__(self, ts: int = T0):
        self.ts = ts

    def __call__(self) -> int:
        return self.ts


class TickingClock:
    def __init__(self, start: int = T0):
        self.ts = start

    def __call__(self) -> int:
        self.ts += NS_PER_MS
        return self.ts


class FakeSleep:
    def __init__(self, clock: TickingClock):
        self.calls: list[float] = []
        self.clock = clock

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.ts += int(seconds * NS_PER_S)


def make_trade(i: int, ts: int, symbol: str = SYMBOL) -> Trade:
    return Trade(
        exchange=EXCHANGE, symbol=symbol, ts_event=ts, ts_recv=ts + NS_PER_MS,
        price=50_000.0 + (i % 89), qty=0.01, qty_usd=500.0,
        side=Side.BUY if i % 2 == 0 else Side.SELL, trade_id=i,
    )


def make_mark(ts: int) -> MarkPrice:
    return MarkPrice(
        exchange=EXCHANGE, symbol=SYMBOL, ts_event=ts, ts_recv=ts + NS_PER_MS,
        mark_price=50_000.0, index_price=50_000.0, funding_rate=1e-4,
        next_funding_ts=ts + 8 * HOUR_NS,
    )


def make_liq(ts: int, symbol: str = SYMBOL) -> Liquidation:
    return Liquidation(
        exchange=EXCHANGE, symbol=symbol, ts_event=ts, ts_recv=ts + NS_PER_MS,
        price=50_000.0, qty=0.5, qty_usd=25_000.0, side=Side.SELL,
    )


def make_oi(ts: int) -> OpenInterest:
    return OpenInterest(
        exchange=EXCHANGE, symbol=SYMBOL, ts_event=ts, ts_recv=ts + NS_PER_MS,
        open_interest=80_000.0, open_interest_usd=80_000.0 * 50_000.0,
    )


def diff_chain(n: int, symbol: str = SYMBOL, start_u: int = 1_000, ts0: int = T0) -> list[DepthDiff]:
    """Fast contiguous U/u/pu chain (synth_book_stream is too slow for 100k)."""
    out: list[DepthDiff] = []
    prev = start_u
    ts = ts0
    for _ in range(n):
        ts += NS_PER_MS
        first = prev + 1
        final = first + 2
        out.append(
            DepthDiff(
                exchange=EXCHANGE, symbol=symbol, ts_event=ts, ts_recv=ts + NS_PER_MS,
                first_update_id=first, final_update_id=final, prev_final_update_id=prev,
                bids=((50_000.0, 1.0),), asks=((50_001.0, 1.0),),
            )
        )
        prev = final
    return out


def snap(symbol: str, last_id: int, ts: int = T0) -> BookSnapshot:
    return BookSnapshot(
        exchange=EXCHANGE, symbol=symbol, ts_event=ts, ts_recv=ts + NS_PER_MS,
        last_update_id=last_id, bids=((50_000.0, 1.0),), asks=((50_001.0, 1.0),),
    )


def make_client(factory, **kw) -> tuple[ReconnectingWSClient, FakeSleep]:
    clock = TickingClock()
    sleep = FakeSleep(clock)
    client = ReconnectingWSClient(
        "wss://test/stream", factory, rng=random.Random(SEED), clock=clock, sleep=sleep, **kw
    )
    return client, sleep


async def drain(client: ReconnectingWSClient) -> list:
    return [payload async for payload, _ts in client.messages()]


# --------------------------------------------------------------------------- #
# 2) BatchWriter flood
# --------------------------------------------------------------------------- #
def test_batchwriter_flood_mixed_types_bounded_memory(tmp_data):
    n = int(200_000 * SCALE)
    max_rows = 10_000
    writer = BatchWriter(tmp_data, max_rows=max_rows, max_age_s=3600.0, clock=FakeClock())
    counts = {"trade": 0, "mark_price": 0, "liquidation": 0, "open_interest": 0}
    step_ns = min(10 * NS_PER_MS, (HOUR_NS - NS_PER_S) // n)  # whole flood inside one hour
    tracemalloc.start()
    peak_buffered = 0
    t = time.perf_counter()
    for i in range(n):
        ts = T0 + i * step_ns
        r = i % 10
        if r < 7:
            writer.add(make_trade(counts["trade"], ts))
            counts["trade"] += 1
        elif r == 7:
            writer.add(make_mark(ts))
            counts["mark_price"] += 1
        elif r == 8:
            writer.add(make_liq(ts))
            counts["liquidation"] += 1
        else:
            writer.add(make_oi(ts))
            counts["open_interest"] += 1
        if i % 5_000 == 0:
            peak_buffered = max(peak_buffered, writer.buffered_rows)
    writer.flush_all()
    elapsed = time.perf_counter() - t
    _cur, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert writer.buffered_rows == 0
    assert peak_buffered <= max_rows * len(counts)  # buffer never hoards past max_rows/key
    assert peak_mem < 256 * 1024 * 1024  # generous: unbounded buffering would blow this

    expected_files = 0
    for stream, c in counts.items():
        frame = read_stream(tmp_data, stream, exchange=EXCHANGE, symbol=SYMBOL)
        assert len(frame) == c  # nothing lost, nothing duplicated
        expected_files += math.ceil(c / max_rows)
    trades = read_stream(tmp_data, "trade", exchange=EXCHANGE, symbol=SYMBOL)
    assert sorted(trades["trade_id"].to_list()) == list(range(counts["trade"]))
    files = list(tmp_data.glob("*/exchange=*/symbol=*/date=*/hour=*/part-*.parquet"))
    assert len(files) == expected_files  # rotation by max_rows, no file spam

    rate = n / elapsed
    assert rate > 5_000  # generous sanity bound
    print(f"\nbatchwriter flood: {n} recs in {elapsed:.2f}s = {rate:,.0f} rec/s, "
          f"peak tracemalloc {peak_mem / 1e6:.1f} MB, files {len(files)}")


def test_batchwriter_age_rotation_under_slow_flood(tmp_data):
    clock = FakeClock()
    writer = BatchWriter(tmp_data, max_rows=100_000, max_age_s=60.0, clock=clock)
    n = int(2_000 * SCALE)
    for i in range(n):
        clock.ts = T0 + i * NS_PER_S  # 1 rec/s: only max_age can rotate
        writer.add(make_trade(i, clock.ts))
    writer.flush_all()
    frame = read_stream(tmp_data, "trade", exchange=EXCHANGE, symbol=SYMBOL)
    assert len(frame) == n
    files = list(tmp_data.glob("trade/**/part-*.parquet"))
    assert n // 70 <= len(files) <= n // 50 + 1  # ~one file per max_age window


# --------------------------------------------------------------------------- #
# 3) sequencer: volume + adversarial
# --------------------------------------------------------------------------- #
def test_sequencer_100k_contiguous_chain_throughput():
    n = int(100_000 * SCALE)
    diffs = diff_chain(n)
    seq = DepthSequencer(SYMBOL)
    seq.set_snapshot(snap(SYMBOL, 1_000))
    released = 0
    t = time.perf_counter()
    for d in diffs:
        released += len(seq.add_diff(d))
    elapsed = time.perf_counter() - t
    assert released == n and seq.synced and seq.gaps == []
    rate = n / elapsed
    assert rate > 20_000  # generous sanity bound
    print(f"\nsequencer chain: {n} diffs in {elapsed:.2f}s = {rate:,.0f} diff/s")


def test_sequencer_duplicates_and_old_replay_dropped_silently():
    n = int(5_000 * SCALE)
    diffs = diff_chain(n)
    seq = DepthSequencer(SYMBOL)
    seq.set_snapshot(snap(SYMBOL, 1_000))
    released: list[DepthDiff] = []
    for d in diffs:
        released.extend(seq.add_diff(d))
        assert seq.add_diff(d) == []  # immediate duplicate
    for d in diffs[: n // 2]:
        assert seq.add_diff(d) == []  # replay of an old block
    assert [d.final_update_id for d in released] == [d.final_update_id for d in diffs]
    assert seq.gaps == [] and seq.synced


def test_sequencer_reordering_flags_gap_and_snapshot_recovers():
    diffs = diff_chain(100)
    seq = DepthSequencer(SYMBOL)
    seq.set_snapshot(snap(SYMBOL, 1_000))
    released: list[DepthDiff] = []
    with capture_logs():
        for d in diffs[:50]:
            released.extend(seq.add_diff(d))
        # swap 50/51: 51 first breaks pu contiguity, everything after buffers
        for d in [diffs[51], diffs[50], *diffs[52:]]:
            released.extend(seq.add_diff(d))
    assert len(seq.gaps) == 1 and not seq.synced
    assert len(released) == 50  # nothing out of order ever released
    # resync past the disturbance: both swapped diffs are covered and dropped
    released.extend(seq.set_snapshot(snap(SYMBOL, diffs[51].final_update_id)))
    assert seq.synced and len(seq.gaps) == 1
    assert [d.final_update_id for d in released] == [
        d.final_update_id for d in diffs[:50] + diffs[52:]
    ]


@pytest.mark.parametrize("gap_len", [1, int(10_000 * SCALE) or 1])
def test_sequencer_gap_sizes_flagged_once_then_resync(gap_len):
    head = tail = int(2_000 * SCALE) or 100
    diffs = diff_chain(head + gap_len + tail)
    seq = DepthSequencer(SYMBOL)
    seq.set_snapshot(snap(SYMBOL, 1_000))
    released: list[DepthDiff] = []
    with capture_logs():
        for d in diffs[:head]:
            released.extend(seq.add_diff(d))
        for d in diffs[head + gap_len :]:
            released.extend(seq.add_diff(d))
    assert len(seq.gaps) == 1  # one hole = exactly one event, any size
    gap = seq.gaps[0]
    assert gap.expected == diffs[head - 1].final_update_id
    assert gap.got == diffs[head + gap_len].prev_final_update_id
    assert len(released) == head and not seq.synced

    released.extend(seq.set_snapshot(snap(SYMBOL, diffs[head + gap_len - 1].final_update_id)))
    assert seq.synced and seq.buffered == 0 and len(seq.gaps) == 1
    assert [d.final_update_id for d in released] == [
        d.final_update_id for d in diffs[:head] + diffs[head + gap_len :]
    ]


def test_sequencer_three_symbols_interleaved_independent_chains():
    n = int(30_000 * SCALE)
    symbols = ["BTCUSDT", "SOLUSDT", "DOGEUSDT"]
    chains = {s: diff_chain(n, symbol=s, start_u=1_000 + k * 10_000_000) for k, s in enumerate(symbols)}
    seqs = {s: DepthSequencer(s) for s in symbols}
    released: dict[str, list[DepthDiff]] = {s: [] for s in symbols}
    for k, s in enumerate(symbols):
        seqs[s].set_snapshot(snap(s, 1_000 + k * 10_000_000))
    for i in range(n):  # strict interleave, one diff per symbol per step
        for s in symbols:
            for out in seqs[s].add_diff(chains[s][i]):
                assert out.symbol == s  # no cross-symbol leakage
                released[s].append(out)
    for s in symbols:
        assert seqs[s].synced and seqs[s].gaps == []
        assert [d.final_update_id for d in released[s]] == [
            d.final_update_id for d in chains[s]
        ]


def test_sequencer_resync_storm_state_never_stale():
    n = int(50_000 * SCALE)
    diffs = diff_chain(n)
    need_resync = False

    def on_resync() -> None:
        nonlocal need_resync
        need_resync = True

    seq = DepthSequencer(SYMBOL, on_resync=on_resync)
    seq.set_snapshot(snap(SYMBOL, 1_000))
    released: list[DepthDiff] = []
    skipped = 0
    with capture_logs():
        for i, d in enumerate(diffs):
            if i % 100 == 50:  # storm: lose one diff in every hundred
                skipped += 1
                continue
            released.extend(seq.add_diff(d))
            if need_resync:
                need_resync = False
                # fresh snapshot at the lost diff's u (its pu is in the gap event)
                released.extend(seq.set_snapshot(snap(SYMBOL, seq.gaps[-1].got, ts=d.ts_event)))
                assert seq.synced and seq.buffered == 0  # never stale after resync
    assert len(seq.gaps) == skipped == n // 100
    assert seq.synced and seq.buffered == 0
    assert [d.final_update_id for d in released] == [
        d.final_update_id for i, d in enumerate(diffs) if i % 100 != 50
    ]


def test_sequencer_presnapshot_buffer_bounded_and_fast_on_overflow_storm():
    n = int(120_000 * SCALE)
    max_buffer = max(1_000, n // 3)
    diffs = diff_chain(n)
    seq = DepthSequencer(SYMBOL, max_buffer=max_buffer)
    t = time.perf_counter()
    with capture_logs() as cap:
        for d in diffs:  # no snapshot yet: everything buffers
            assert seq.add_diff(d) == []
    elapsed = time.perf_counter() - t
    assert seq.buffered == max_buffer  # bounded, oldest dropped
    overflow = [e for e in cap if e["event"] == "seq_buffer_overflow"]
    assert len(overflow) == n - max_buffer
    assert elapsed < 10.0  # generous; quadratic drop-oldest would explode here

    # a snapshot at the newest dropped diff drains the whole surviving buffer
    released = seq.set_snapshot(snap(SYMBOL, diffs[n - max_buffer - 1].final_update_id))
    assert seq.synced and seq.buffered == 0 and seq.gaps == []
    assert [d.final_update_id for d in released] == [
        d.final_update_id for d in diffs[n - max_buffer :]
    ]
    print(f"\nsequencer overflow storm: {n} pre-snapshot diffs in {elapsed:.2f}s, "
          f"buffer capped at {max_buffer}")


# --------------------------------------------------------------------------- #
# 4) ws client storms
# --------------------------------------------------------------------------- #
def test_ws_reconnect_storm_backoff_capped_no_task_leak():
    n = 1_000

    async def factory(url: str):
        raise ConnectionError("refused")

    client, sleep = make_client(factory, max_reconnects=n)

    async def run() -> list:
        with capture_logs():
            msgs = [m async for m, _ts in client.messages()]
        assert len(asyncio.all_tasks()) == 1  # generator leaked no tasks
        return msgs

    assert asyncio.run(run()) == []
    assert [e.kind for e in client.events] == [DISCONNECTED] * (n + 1)  # one event per failure
    rng = random.Random(SEED)
    expected = [backoff_delay(a, 0.5, 30.0, rng) for a in range(1, n + 1)]
    assert sleep.calls == expected  # deterministic equal-jitter sequence
    assert max(sleep.calls) <= 30.0
    assert all(15.0 <= d <= 30.0 for d in sleep.calls[6:])  # capped, never past cap
    assert sleep.calls[:6] == sorted(sleep.calls[:6])  # growth up to the cap


def test_ws_100k_messages_delivered_in_order():
    n = int(100_000 * SCALE)
    payloads = [f"m{i}" for i in range(n)]

    class IterTransport:
        def __init__(self, items):
            self._it = iter(items)
            self.closed = False

        async def recv(self, timeout: float | None = None):
            try:
                return next(self._it)
            except StopIteration:
                raise TransportExhausted from None

        async def close(self) -> None:
            self.closed = True

    transport = IterTransport(payloads)

    async def factory(url: str):
        return transport

    client, sleep = make_client(factory)
    t = time.perf_counter()
    msgs = asyncio.run(drain(client))
    elapsed = time.perf_counter() - t
    assert msgs == payloads  # all delivered, in order
    assert transport.closed
    assert [e.kind for e in client.events] == [CONNECTED]
    assert sleep.calls == []
    rate = n / elapsed
    assert rate > 10_000  # generous sanity bound
    print(f"\nws pump: {n} msgs in {elapsed:.2f}s = {rate:,.0f} msg/s")


def test_ws_transport_exhausted_mid_stream_ends_cleanly():
    class Scripted:
        def __init__(self, script):
            self.script = list(script)
            self.closed = False

        async def recv(self, timeout: float | None = None):
            item = self.script.pop(0) if self.script else TransportExhausted()
            if isinstance(item, Exception):
                raise item
            return item

        async def close(self) -> None:
            self.closed = True

    transport = Scripted(["a", "b", TransportExhausted()])

    async def factory(url: str):
        return transport

    client, sleep = make_client(factory)
    assert asyncio.run(drain(client)) == ["a", "b"]
    assert transport.closed  # finally-path close even on clean end
    assert [e.kind for e in client.events] == [CONNECTED]
    assert sleep.calls == []


def test_ws_heartbeat_silence_gap_and_disconnect_once_per_cycle():
    cycles = 50
    made: list = []

    class Scripted:
        def __init__(self, script):
            self.script = list(script)
            self.closed = False

        async def recv(self, timeout: float | None = None):
            assert timeout == 5.0  # heartbeat propagated to the transport
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        async def close(self) -> None:
            self.closed = True

    async def factory(url: str):
        if len(made) == cycles:
            raise TransportExhausted
        t = Scripted([f"m{len(made)}", TimeoutError()])
        made.append(t)
        return t

    client, sleep = make_client(factory, heartbeat_s=5.0)
    with capture_logs():
        msgs = asyncio.run(drain(client))
    assert msgs == [f"m{i}" for i in range(cycles)]
    assert [e.kind for e in client.events] == [CONNECTED, GAP_SUSPECTED, DISCONNECTED] * cycles
    assert len(sleep.calls) == cycles  # one backoff per silent cycle
    assert all(t.closed for t in made)


# --------------------------------------------------------------------------- #
# 5) recorder e2e: sequencer + writer + gap log under a mixed 50k flood
# --------------------------------------------------------------------------- #
def test_recorder_e2e_mixed_flood_nothing_reaching_writer_is_lost(tmp_data):
    n_diffs = int(20_000 * SCALE)
    n_trades = int(8_000 * SCALE)
    n_liq = int(2_000 * SCALE)
    skip_every = max(n_diffs // 4, 2)
    symbols = ["BTCUSDT", "SOLUSDT"]

    chains: dict[str, list[DepthDiff]] = {}
    n_skipped: dict[str, int] = {}
    for k, s in enumerate(symbols):
        full = diff_chain(n_diffs, symbol=s, start_u=1_000 + k * 50_000_000)
        chains[s] = [d for i, d in enumerate(full) if i % skip_every != skip_every // 2]
        n_skipped[s] = n_diffs - len(chains[s])

    # per-source order preserved, sources interleaved by a seeded rng (ws-like)
    sources: list[list] = [
        *[chains[s] for s in symbols],
        [make_trade(i, T0 + i * NS_PER_MS) for i in range(n_trades)],
        [make_liq(T0 + i * NS_PER_MS) for i in range(n_liq)],
    ]
    rng = random.Random(SEED)
    heads = [0] * len(sources)
    merged: list = []
    while True:
        alive = [j for j in range(len(sources)) if heads[j] < len(sources[j])]
        if not alive:
            break
        j = rng.choices(alive, weights=[len(sources[j]) - heads[j] for j in alive])[0]
        merged.append(sources[j][heads[j]])
        heads[j] += 1

    writer = BatchWriter(tmp_data, max_rows=5_000, max_age_s=3600.0, clock=FakeClock())
    gap_log: list[GapEvent] = []
    pending: set[str] = set()
    seqs = {
        s: DepthSequencer(s, on_gap=gap_log.append, on_resync=lambda s=s: pending.add(s))
        for s in symbols
    }
    for k, s in enumerate(symbols):
        seqs[s].set_snapshot(snap(s, 1_000 + k * 50_000_000))

    to_writer = 0
    snaps_written = 0
    t = time.perf_counter()
    with capture_logs():
        for rec in merged:
            if isinstance(rec, DepthDiff):
                seq = seqs[rec.symbol]
                for ready in seq.add_diff(rec):
                    writer.add(ready)
                    to_writer += 1
                if rec.symbol in pending:  # record_live-style resync on next diff
                    pending.discard(rec.symbol)
                    fresh = snap(rec.symbol, seq.gaps[-1].got, ts=rec.ts_event)
                    for ready in seq.set_snapshot(fresh):
                        writer.add(ready)
                        to_writer += 1
                    writer.add(fresh)
                    snaps_written += 1
            else:
                writer.add(rec)
                to_writer += 1
        writer.flush_all()
        write_gap_events(tmp_data, gap_log, EXCHANGE, "depth_diff")
    elapsed = time.perf_counter() - t
    assert writer.buffered_rows == 0

    trades = read_stream(tmp_data, "trade", exchange=EXCHANGE, symbol=SYMBOL)
    assert sorted(trades["trade_id"].to_list()) == list(range(n_trades))
    assert len(read_stream(tmp_data, "liquidation", exchange=EXCHANGE, symbol=SYMBOL)) == n_liq
    depth_total = 0
    for s in symbols:
        depth = read_stream(tmp_data, "depth_diff", exchange=EXCHANGE, symbol=s)
        assert len(depth) == len(chains[s])  # every fed diff eventually released and stored
        assert depth["final_update_id"].is_sorted()
        depth_total += len(depth)
    assert len(read_stream(tmp_data, "book_snapshot", exchange=EXCHANGE)) == snaps_written
    assert to_writer == depth_total + n_trades + n_liq

    gaps = read_gaps(tmp_data, exchange=EXCHANGE)
    assert len(gaps) == len(gap_log) == sum(n_skipped.values())
    assert set(gaps["symbol"].to_list()) == set(symbols)
    assert (gaps["got"] != gaps["expected"]).all()
    rate = len(merged) / elapsed
    print(f"\nrecorder e2e: {len(merged)} msgs in {elapsed:.2f}s = {rate:,.0f} msg/s, "
          f"gaps {len(gap_log)}, resyncs {snaps_written}")


# --------------------------------------------------------------------------- #
# 6) quality report on pathological lakes
# --------------------------------------------------------------------------- #
def test_quality_report_on_empty_lake(tmp_data):
    t0, t1 = T0, T0 + HOUR_NS
    report = daily_quality_report(tmp_data, EXCHANGE, SYMBOL, t0, t1)
    for q in report.streams.values():
        assert q.n_records == 0
        assert q.silence_gaps == [(t0, t1)]  # whole window honestly reported silent
        assert q.uptime_pct == 0.0
        assert q.latency_ms.size == 0
    assert report.seq_gaps.is_empty() and not report.clean


def test_quality_report_single_file_single_row(tmp_data):
    ts = T0 + 30 * 60 * NS_PER_S
    writer = BatchWriter(tmp_data, max_rows=1, max_age_s=3600.0, clock=FakeClock())
    writer.add(make_trade(0, ts))
    report = daily_quality_report(
        tmp_data, EXCHANGE, SYMBOL, T0, T0 + HOUR_NS, max_silence_s={"trade": 10.0}
    )
    q = report.streams["trade"]
    assert q.n_records == 1 and q.latency_ms.size == 1
    assert q.silence_gaps == [(T0, ts), (ts, T0 + HOUR_NS)]
    assert q.uptime_pct == 0.0 and not report.clean


def test_quality_report_out_of_order_ts(tmp_data):
    n = int(5_000 * SCALE)
    trades = [make_trade(i, T0 + i * 100 * NS_PER_MS) for i in range(n)]
    rng = random.Random(SEED)
    rng.shuffle(trades)  # lake written wildly out of order
    writer = BatchWriter(tmp_data, max_rows=500, max_age_s=3600.0, clock=FakeClock())
    for tr in trades:
        writer.add(tr)
    writer.flush_all()
    t1 = T0 + n * 100 * NS_PER_MS
    report = daily_quality_report(
        tmp_data, EXCHANGE, SYMBOL, T0, t1, max_silence_s={"trade": 5.0}
    )
    q = report.streams["trade"]
    assert q.n_records == n
    assert q.silence_gaps == [] and q.uptime_pct == 100.0  # dense once sorted
    assert report.clean
    assert read_stream(tmp_data, "trade", exchange=EXCHANGE, symbol=SYMBOL)["ts_event"].is_sorted()


def test_quality_report_day_with_22h_hole(tmp_data):
    d0 = (T0 // DAY_NS) * DAY_NS
    recs = [make_trade(i, d0 + i * NS_PER_S) for i in range(3_600)]  # hour 00
    recs += [make_trade(3_600 + i, d0 + 23 * HOUR_NS + i * NS_PER_S) for i in range(3_600)]
    writer = BatchWriter(tmp_data, max_rows=100_000, max_age_s=3600.0, clock=FakeClock())
    for tr in recs:
        writer.add(tr)
    writer.flush_all()
    report = daily_quality_report(
        tmp_data, EXCHANGE, SYMBOL, d0, d0 + DAY_NS, max_silence_s={"trade": 10.0}
    )
    q = report.streams["trade"]
    assert q.n_records == 7_200
    assert q.silence_gaps == [(d0 + 3_599 * NS_PER_S, d0 + 23 * HOUR_NS)]
    assert 8.0 < q.uptime_pct < 9.0  # ~2 live hours of 24, honestly reported
    assert not report.clean
