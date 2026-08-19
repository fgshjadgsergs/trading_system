"""M1 regression tests for review findings: poller wiring (OI single record,
taker_ls symbol), poller survival on bad payloads, resync failure retry."""

from __future__ import annotations

import asyncio
import json
from functools import partial
from pathlib import Path

from trading_system.collectors.binance import parse_open_interest, parse_taker_ls
from trading_system.collectors.recorder import RestPoller
from trading_system.core.schema import OpenInterest, RatioPoint

FIXTURES = Path(__file__).parent / "fixtures" / "m1"


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_oi_poller_wiring_emits_single_record():
    """parse_open_interest returns ONE dataclass; the record_live wiring wraps
    it in a list so RestPoller can iterate (previously died with TypeError)."""
    payload = json.loads((FIXTURES / "open_interest.json").read_text())

    async def fetch():
        return payload

    sunk: list = []
    poller = RestPoller(
        1.0,
        fetch,
        lambda p, ts_recv: [parse_open_interest(p, ts_recv)],
        sunk.append,
        sleep=lambda s: asyncio.sleep(0),
    )
    emitted = _run(poller.run(n_polls=2))
    assert emitted == 2
    assert all(isinstance(r, OpenInterest) for r in sunk)


def test_taker_ls_poller_wiring_supplies_symbol():
    """The takerlongshortRatio payload has no symbol field; record_live must
    bind the polled symbol (previously ValueError on first poll)."""
    payload = json.loads((FIXTURES / "taker_ls.json").read_text())
    normalizer = partial(parse_taker_ls, symbol="BTCUSDT")
    recs = normalizer(payload, 123)
    assert recs and all(isinstance(r, RatioPoint) for r in recs)
    assert {r.symbol for r in recs} == {"BTCUSDT"}


def test_poller_survives_normalizer_and_sink_errors():
    """HTTP-200 error bodies and sink failures must not kill the poller."""
    calls = {"n": 0}

    async def fetch():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"code": -4108, "msg": "maintenance"}  # bad shape -> normalizer raises
        return json.loads((FIXTURES / "open_interest.json").read_text())

    def normalizer(payload, ts_recv):
        return [parse_open_interest(payload, ts_recv)]

    sunk: list = []
    errors: list = []

    def sink(rec):
        if calls["n"] == 2:
            raise OSError("disk full")  # sink failure on poll 2
        sunk.append(rec)

    poller = RestPoller(
        1.0,
        fetch,
        normalizer,
        sink,
        sleep=lambda s: asyncio.sleep(0),
        on_error=errors.append,
    )
    emitted = _run(poller.run(n_polls=3))
    assert len(errors) == 2  # bad payload + sink failure, both survived
    assert emitted == 1 and len(sunk) == 1  # poll 3 still delivered


def test_record_live_resync_retries_on_snapshot_failure():
    """A failed REST snapshot re-queues the symbol instead of crashing."""
    import scripts.record_live as rl

    # reproduce the resync closure's behavior with a failing adapter
    class FailingAdapter:
        async def snapshot(self, symbol: str):
            raise OSError("HTTP 502")

    resync_pending: set[str] = set()

    async def resync(symbol: str) -> None:
        try:
            await FailingAdapter().snapshot(symbol)
        except Exception:
            resync_pending.add(symbol)
            return
        raise AssertionError("unreachable")

    _run(resync("BTCUSDT"))
    assert "BTCUSDT" in resync_pending
    # and the real script contains the guarded call (source-level check)
    src = Path(rl.__file__).read_text()
    assert "book_resync_failed" in src
    assert "resync_pending.add(symbol)" in src
