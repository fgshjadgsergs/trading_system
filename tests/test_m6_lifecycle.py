"""M6: level lifecycle events from hand-built book states + tape."""

from __future__ import annotations

from trading_system.core.schema import Side, Trade
from trading_system.core.timeutils import NS_PER_MS, NS_PER_S
from trading_system.spoof.lifecycle import BookState, LevelEventType, LevelJournal

T0 = 1_755_600_000 * NS_PER_S
STEP = 500 * NS_PER_MS

BIDS = ((99.0, 1.0), (98.0, 1.0), (97.0, 1.0))
ASKS = ((101.0, 1.0), (102.0, 1.0), (103.0, 1.0))


def _state(i: int, extra_bid: tuple[float, float] | None = None) -> BookState:
    bids = BIDS + ((extra_bid,) if extra_bid is not None else ())
    return BookState(ts=T0 + i * STEP, bids=bids, asks=ASKS)


def _trade(i: int, price: float, qty: float, side: Side, tid: int = 1) -> Trade:
    ts = T0 + i * STEP - 100 * NS_PER_MS  # shortly before state i is observed
    return Trade(
        exchange="binance_usdm",
        symbol="BTCUSDT",
        ts_event=ts,
        ts_recv=ts + NS_PER_MS,
        price=price,
        qty=qty,
        qty_usd=price * qty,
        side=side,
        trade_id=tid,
    )


def _journal(**kw) -> LevelJournal:
    kw.setdefault("large_k", 3.0)
    kw.setdefault("iceberg_refill_ms", 300)
    return LevelJournal(**kw)


def test_event_kinds_and_episode_accounting():
    states = [
        _state(0),
        _state(1, (96.0, 5.0)),  # appeared
        _state(2, (96.0, 7.0)),  # grew
        _state(3, (96.0, 4.0)),  # -3 explained by a print
        _state(4),  # removed with no prints -> canceled
    ]
    trades = [_trade(3, 96.0, 3.0, Side.SELL)]
    j = _journal().run(states, trades)

    kinds = [str(e.kind) for e in j.events if e.price == 96.0]
    assert kinds == ["appeared", "grew", "reduced_by_trade", "canceled"]
    (ep,) = [e for e in j.episodes if e.price == 96.0]
    assert ep.birth_ts == T0 + STEP
    assert ep.death_ts == T0 + 4 * STEP
    assert ep.max_qty == 7.0
    assert abs(ep.filled_qty - 3.0) < 1e-9
    assert abs(ep.canceled_qty - 4.0) < 1e-9
    assert abs(ep.fill_frac - 3.0 / 7.0) < 1e-9
    assert not ep.alive
    assert ep.lifetime_ns == 3 * STEP
    # qty 5 > large_k(3) * median(1.0) -> large
    assert ep.was_large
    # background levels never exceed the threshold
    assert all(not e.was_large for e in j.episodes if e.price != 96.0)


def test_taker_side_must_match_level_side():
    states = [_state(0), _state(1, (96.0, 5.0)), _state(2, (96.0, 2.0))]
    buy_print = [_trade(2, 96.0, 3.0, Side.BUY)]  # wrong side for a bid level

    j = _journal().run(states, buy_print)
    ev = [e for e in j.events if e.price == 96.0 and e.qty_before == 5.0]
    assert ev[0].kind is LevelEventType.CANCELED
    assert ev[0].canceled_qty == 3.0

    j2 = _journal(match_taker_side=False).run(states, buy_print)
    ev2 = [e for e in j2.events if e.price == 96.0 and e.qty_before == 5.0]
    assert ev2[0].kind is LevelEventType.REDUCED_BY_TRADE


def test_qty_tolerance_on_fill_matching():
    states = [_state(0), _state(1, (96.0, 5.0)), _state(2, (96.0, 4.0))]
    prints = [_trade(2, 96.0, 0.9, Side.SELL)]  # decrease is 1.0

    j = _journal(qty_rel_tol=0.15).run(states, prints)
    ev = [e for e in j.events if e.price == 96.0 and e.qty_before == 5.0][0]
    assert ev.kind is LevelEventType.REDUCED_BY_TRADE
    assert abs(ev.filled_qty - 0.9) < 1e-9

    j2 = _journal(qty_rel_tol=0.05).run(states, prints)
    ev2 = [e for e in j2.events if e.price == 96.0 and e.qty_before == 5.0][0]
    assert ev2.kind is LevelEventType.CANCELED
    assert abs(ev2.filled_qty - 0.9) < 1e-9
    assert abs(ev2.canceled_qty - 0.1) < 1e-9


def test_prints_are_consumed_once():
    states = [
        _state(0),
        _state(1, (96.0, 5.0)),
        _state(2, (96.0, 4.0)),
        _state(3, (96.0, 3.0)),
    ]
    prints = [_trade(2, 96.0, 1.0, Side.SELL)]  # explains only the first decrease
    j = _journal().run(states, prints)
    ev = [e for e in j.events if e.price == 96.0 and e.qty_before in (5.0, 4.0)]
    assert ev[0].kind is LevelEventType.REDUCED_BY_TRADE
    assert ev[1].kind is LevelEventType.CANCELED
    assert ev[1].filled_qty == 0.0


def test_frames_shapes_and_grid_tracking():
    states = [_state(0), _state(1, (96.0, 5.0)), _state(2)]
    j = _journal().run(states, [])
    ev = j.events_frame()
    assert set(ev.columns) >= {"ts", "side", "price", "kind", "filled_qty", "canceled_qty"}
    grid = j.grid_frame()
    # 6 background levels x 3 states + the wall once
    assert grid.height == 6 * 3 + 1
    wall_rows = grid.filter(grid["price"] == 96.0)
    assert wall_rows.height == 1
    assert bool(wall_rows["is_large"][0])
    # config-driven defaults resolve when params are omitted
    j_default = LevelJournal()
    assert j_default.large_k > 0
    assert j_default.iceberg_refill_ms > 0
