"""M1: BatchWriter (rows/age flush, hourly partitions, no loss) and RestPoller."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import polars as pl

from trading_system.collectors.binance import parse_open_interest
from trading_system.collectors.recorder import BatchWriter, RestPoller
from trading_system.core.io import read_stream
from trading_system.core.schema import Side, Trade
from trading_system.core.timeutils import NS_PER_MS, NS_PER_S, ns_to_dt

FIX = Path(__file__).parent / "fixtures" / "m1"
EXCHANGE = "binance_usdm"
SYMBOL = "BTCUSDT"
HOUR_NS = 3_600 * NS_PER_S
# 10 minutes before an hour boundary so the record stream spans two hour dirs
T0 = (1_755_600_000 * NS_PER_S // HOUR_NS) * HOUR_NS + 50 * 60 * NS_PER_S


def make_trade(i: int, ts: int) -> Trade:
    return Trade(
        exchange=EXCHANGE,
        symbol=SYMBOL,
        ts_event=ts,
        ts_recv=ts + 5 * NS_PER_MS,
        price=50_000.0 + i,
        qty=0.01,
        qty_usd=(50_000.0 + i) * 0.01,
        side=Side.BUY if i % 2 == 0 else Side.SELL,
        trade_id=i,
    )


class FakeClock:
    def __init__(self, ts: int = T0):
        self.ts = ts

    def __call__(self) -> int:
        return self.ts


# --------------------------------------------------------------------------- #
def test_no_record_lost_across_flush_boundaries(tmp_data):
    clock = FakeClock()
    writer = BatchWriter(tmp_data, max_rows=100, max_age_s=3600.0, clock=clock)
    n = 250
    written: list[Path] = []
    for i in range(n):
        ts = T0 + i * 6 * NS_PER_S  # 25 minutes total, crosses the hour boundary
        written.extend(writer.add(make_trade(i, ts)))
    assert writer.buffered_rows == 50
    written.extend(writer.flush_all())
    assert writer.buffered_rows == 0
    assert len(written) >= 3  # two row-flushes + final

    frame = read_stream(tmp_data, "trade", exchange=EXCHANGE, symbol=SYMBOL)
    assert len(frame) == n
    assert sorted(frame["trade_id"].to_list()) == list(range(n))  # exactly once each


def test_partitions_land_in_correct_hour_dirs(tmp_data):
    writer = BatchWriter(tmp_data, max_rows=10_000, max_age_s=3600.0, clock=FakeClock())
    for i in range(250):
        writer.add(make_trade(i, T0 + i * 6 * NS_PER_S))
    writer.flush_all()

    files = sorted(tmp_data.glob("trade/exchange=*/symbol=*/date=*/hour=*/part-*.parquet"))
    assert len(files) == 2  # one batch split across two hour partitions
    hours = set()
    for f in files:
        hour_dir = f.parent.name  # hour=HH
        date_dir = f.parent.parent.name  # date=YYYY-MM-DD
        part = pl.read_parquet(f)
        for ts in part["ts_event"].to_list():
            dt = ns_to_dt(ts)
            assert f"hour={dt.strftime('%H')}" == hour_dir
            assert f"date={dt.strftime('%Y-%m-%d')}" == date_dir
        hours.add(hour_dir)
    assert len(hours) == 2


def test_flush_on_max_age(tmp_data):
    clock = FakeClock()
    writer = BatchWriter(tmp_data, max_rows=10_000, max_age_s=60.0, clock=clock)
    for i in range(5):
        assert writer.add(make_trade(i, T0 + i)) == []
    assert writer.poll() == []  # not aged yet
    clock.ts += 61 * NS_PER_S
    written = writer.poll()  # age flush without any new record
    assert written and writer.buffered_rows == 0
    assert len(read_stream(tmp_data, "trade", exchange=EXCHANGE, symbol=SYMBOL)) == 5

    # age flush also triggers on add()
    for i in range(5, 8):
        writer.add(make_trade(i, T0 + i))
    clock.ts += 61 * NS_PER_S
    written = writer.add(make_trade(8, T0 + 8))
    assert written
    assert len(read_stream(tmp_data, "trade", exchange=EXCHANGE, symbol=SYMBOL)) == 9


def test_append_only_parts_never_overwritten(tmp_data):
    writer = BatchWriter(tmp_data, max_rows=50, max_age_s=3600.0, clock=FakeClock())
    for i in range(100):
        writer.add(make_trade(i, T0 + i))  # all inside one hour
    files = list(tmp_data.glob("trade/**/part-*.parquet"))
    assert sorted(f.name for f in files) == ["part-00000.parquet", "part-00001.parquet"]


# --------------------------------------------------------------------------- #
def test_rest_poller_open_interest_cadence():
    payload = json.loads((FIX / "open_interest.json").read_text())
    fetches: list[int] = []

    async def fetch():
        fetches.append(len(fetches))
        return {**payload, "time": payload["time"] + len(fetches) * 7000}

    clock = FakeClock()
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)
        clock.ts += int(s * NS_PER_S)

    sunk = []
    poller = RestPoller(
        7.0,
        fetch,
        lambda p, ts: [parse_open_interest(p, ts, price=50_000.0)],
        sunk.append,
        clock=clock,
        sleep=fake_sleep,
    )
    emitted = asyncio.run(poller.run(n_polls=5))
    assert emitted == 5 and len(sunk) == 5
    assert sleeps == [7.0] * 4  # no sleep after the final poll
    assert [r.ts_recv for r in sunk] == sorted({r.ts_recv for r in sunk})  # fake clock times
    assert all(r.open_interest == 10659.509 for r in sunk)
    assert sunk[0].open_interest_usd == 10659.509 * 50_000.0


def test_rest_poller_survives_fetch_errors():
    calls = {"n": 0}

    async def fetch():
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionError("http 502")
        return {"openInterest": "1.0", "symbol": SYMBOL, "time": 1_755_600_000_000}

    errors: list[Exception] = []
    sunk = []

    async def fake_sleep(s: float) -> None:
        pass

    poller = RestPoller(
        300.0,
        fetch,
        lambda p, ts: [parse_open_interest(p, ts)],
        sunk.append,
        clock=FakeClock(),
        sleep=fake_sleep,
        on_error=errors.append,
    )
    emitted = asyncio.run(poller.run(n_polls=4))
    assert emitted == 3  # one poll lost, poller kept going
    assert len(errors) == 1 and isinstance(errors[0], ConnectionError)


def test_rest_poller_stop():
    async def fetch():
        return {"openInterest": "1.0", "symbol": SYMBOL, "time": 1_755_600_000_000}

    sunk = []
    poller = RestPoller(
        1.0,
        fetch,
        lambda p, ts: [parse_open_interest(p, ts)],
        sunk.append,
        clock=FakeClock(),
        sleep=lambda s: _stop_after_sleep(poller, s),
    )

    async def _stop_after_sleep(p: RestPoller, s: float) -> None:
        p.stop()

    assert asyncio.run(poller.run()) == 1
    assert len(sunk) == 1
