"""M1: reconnecting ws client — deterministic backoff, heartbeat, events."""

from __future__ import annotations

import asyncio
import random

from trading_system.collectors.ws_client import (
    CONNECTED,
    DISCONNECTED,
    GAP_SUSPECTED,
    ReconnectingWSClient,
    TransportExhausted,
    backoff_delay,
)
from trading_system.core.timeutils import NS_PER_S

SEED = 7


class FakeClock:
    def __init__(self, start: int = 1_755_600_000 * NS_PER_S):
        self.ts = start

    def __call__(self) -> int:
        self.ts += 1_000_000  # 1ms per observation
        return self.ts


class FakeSleep:
    """Records requested delays and advances the fake clock; never real-sleeps."""

    def __init__(self, clock: FakeClock):
        self.calls: list[float] = []
        self.clock = clock

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.ts += int(seconds * NS_PER_S)


class ScriptedTransport:
    """Yields scripted items; an Exception instance in the script is raised."""

    def __init__(self, script: list):
        self.script = list(script)
        self.closed = False

    async def recv(self, timeout: float | None = None):
        if not self.script:
            raise TransportExhausted
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self) -> None:
        self.closed = True


def make_client(factory, **kw) -> tuple[ReconnectingWSClient, FakeSleep]:
    clock = FakeClock()
    sleep = FakeSleep(clock)
    client = ReconnectingWSClient(
        "wss://test/stream",
        factory,
        rng=random.Random(SEED),
        clock=clock,
        sleep=sleep,
        **kw,
    )
    return client, sleep


async def drain(client: ReconnectingWSClient) -> list:
    return [payload async for payload, _ts in client.messages()]


# --------------------------------------------------------------------------- #
def test_backoff_delay_deterministic_growth_and_cap():
    rng1, rng2 = random.Random(SEED), random.Random(SEED)
    a = [backoff_delay(i, 0.5, 30.0, rng1) for i in range(1, 9)]
    b = [backoff_delay(i, 0.5, 30.0, rng2) for i in range(1, 9)]
    assert a == b  # same seed => identical jitter
    for i, d in enumerate(a, start=1):
        exp = min(30.0, 0.5 * 2 ** (i - 1))
        assert exp / 2 <= d <= exp
    assert a == sorted(a)  # equal-jitter windows do not overlap
    capped = backoff_delay(50, 0.5, 30.0, random.Random(SEED))
    assert 15.0 <= capped <= 30.0


def test_reconnect_after_drop_collects_all_messages():
    transports = [
        ScriptedTransport(["a", "b", ConnectionError("reset by peer")]),
        ScriptedTransport(["c"]),
    ]
    made: list[ScriptedTransport] = []

    async def factory(url: str):
        t = transports.pop(0)
        made.append(t)
        return t

    client, sleep = make_client(factory)
    msgs = asyncio.run(drain(client))
    assert msgs == ["a", "b", "c"]
    kinds = [e.kind for e in client.events]
    assert kinds == [CONNECTED, GAP_SUSPECTED, DISCONNECTED, CONNECTED]
    assert all(t.closed for t in made)
    # exactly one backoff sleep, deterministic from the seeded rng
    expected = backoff_delay(1, 0.5, 30.0, random.Random(SEED))
    assert sleep.calls == [expected]


def test_heartbeat_timeout_forces_reconnect_with_gap_suspicion():
    calls = {"n": 0}

    async def factory(url: str):
        calls["n"] += 1
        if calls["n"] == 1:
            return ScriptedTransport(["m1", TimeoutError()])
        raise TransportExhausted

    client, sleep = make_client(factory, heartbeat_s=5.0)
    msgs = asyncio.run(drain(client))
    assert msgs == ["m1"]
    kinds = [e.kind for e in client.events]
    assert kinds == [CONNECTED, GAP_SUSPECTED, DISCONNECTED]
    gap = client.events[1]
    assert "heartbeat" in gap.detail
    assert len(sleep.calls) == 1  # backed off before the (refused) reconnect


def test_connect_failures_back_off_exponentially_until_limit():
    async def factory(url: str):
        raise ConnectionError("refused")

    client, sleep = make_client(factory, max_reconnects=6)
    msgs = asyncio.run(drain(client))
    assert msgs == []
    assert [e.kind for e in client.events] == [DISCONNECTED] * 7
    rng = random.Random(SEED)
    expected = [backoff_delay(a, 0.5, 30.0, rng) for a in range(1, 7)]
    assert sleep.calls == expected  # deterministic, growing, capped sequence
    assert all(d <= 30.0 for d in sleep.calls)


def test_events_carry_fake_clock_timestamps_and_message_ts_recv():
    async def factory(url: str):
        return ScriptedTransport(["x"])

    client, _ = make_client(factory)

    async def run():
        out = []
        async for payload, ts in client.messages():
            out.append((payload, ts))
        return out

    msgs = asyncio.run(run())
    assert len(msgs) == 1
    payload, ts_recv = msgs[0]
    assert payload == "x"
    assert ts_recv > 1_755_600_000 * NS_PER_S  # from injected clock, not wall clock
    assert client.events[0].ts < ts_recv


def test_stop_ends_stream():
    async def factory(url: str):
        return ScriptedTransport(["a", "b", "c", "d"])

    client, _ = make_client(factory)

    async def run():
        got = []
        async for payload, _ts in client.messages():
            got.append(payload)
            if len(got) == 2:
                client.stop()
        return got

    assert asyncio.run(run()) == ["a", "b"]
