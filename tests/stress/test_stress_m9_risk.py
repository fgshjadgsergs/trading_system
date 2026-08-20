"""M9 stress: sizing under degenerate inputs, limit cascades, order-state storms,
reconcile at 1000-order scale.

Scenarios (scaled by env STRESS_SCALE, default 1):
- sizing: zero/NaN/inf realized vol and equity must size to zero, never NaN or
  infinity; the EWMA estimator survives 1e5 updates without drifting and is
  not poisoned forever by a single NaN/inf return;
- limits: a cascade of losing equity updates trips the daily stop exactly once
  with the trip pinned to the first crossing tick; the exact -pct boundary
  trips; the kill switch trips at exactly the error threshold, never flaps on
  error-success alternation, and honors the freshness boundary to the ns;
- orders: a seeded storm of 10k transition attempts (legal and illegal mixed)
  against the journaled state machine — every illegal attempt is rejected
  without touching state or journal, and replay equals memory afterwards;
- reconcile: 1000 local orders unknown to the exchange, 1000 exchange ghosts
  unknown locally, and a 50/50 mix with adopted fills and in-flight cancels —
  convergence, idempotence, and not a single order lost or duplicated.

Seeded, offline, no sleeps; only coarse elapsed ceilings on the heavy parts.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest

from trading_system.risk.limits import DailyStop, KillSwitch
from trading_system.risk.orders import (
    _FILL_STATES,
    ALLOWED_TRANSITIONS,
    InvalidTransition,
    OrderJournal,
    OrderState,
    OrderStateMachine,
    ReplayedOrder,
    replay_journal,
)
from trading_system.risk.reconcile import (
    ExchangeOrder,
    ExchangePosition,
    FakeExchange,
    reconcile_on_restart,
)
from trading_system.risk.sizing import EwmaVol, VolTargetSizer, vol_target_position_usd

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))
SEED = 42
NS_PER_S = 1_000_000_000
NS_PER_DAY = 86_400 * NS_PER_S
T0 = 100 * NS_PER_DAY + NS_PER_DAY // 2  # noon of an arbitrary UTC day
SYM = "BTCUSDT"


def n_scaled(base: int) -> int:
    return max(1, int(base * SCALE))


# --------------------------------------------------------------------------
# 1) Sizing: degenerate inputs can never produce NaN/inf sizes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_vol", [0.0, -1.0, 1e-12, math.nan, math.inf, -math.inf])
def test_sizing_zero_or_garbage_vol_sizes_to_zero(bad_vol):
    usd, capped, reason = vol_target_position_usd(100_000.0, 0.01, bad_vol, 1e12)
    assert usd == 0.0 and not capped
    assert "zero-vol guard" in reason


@pytest.mark.parametrize(
    "equity", [0.0, -1.0, -1e12, math.nan, math.inf, -math.inf]
)
def test_sizing_zero_negative_or_garbage_equity_sizes_to_zero(equity):
    """Regression: NaN equity used to sail through the <=0 guard and return a
    NaN notional; inf equity used to size straight to the cap on garbage."""
    usd, capped, _ = vol_target_position_usd(equity, 0.01, 0.02, 1e6)
    assert usd == 0.0 and not capped


def test_sizing_garbage_target_vol_sizes_to_zero():
    for bad in (math.nan, math.inf, -1.0, 0.0):
        usd, capped, _ = vol_target_position_usd(100_000.0, bad, 0.02, 1e6)
        assert usd == 0.0 and not capped


def test_sizing_random_sweep_never_emits_nonfinite():
    rng = np.random.default_rng(SEED)
    n = n_scaled(10_000)
    specials = np.array([0.0, -1.0, math.nan, math.inf, -math.inf, 1e-300, 1e300])
    max_usd = 250_000.0
    for _ in range(n):
        equity = float(rng.choice(specials)) if rng.random() < 0.3 else float(rng.uniform(-1e9, 1e9))
        target = float(rng.choice(specials)) if rng.random() < 0.3 else float(rng.uniform(0, 0.2))
        vol = float(rng.choice(specials)) if rng.random() < 0.3 else float(rng.uniform(0, 0.5))
        usd, capped, reason = vol_target_position_usd(equity, target, vol, max_usd)
        assert math.isfinite(usd), f"non-finite size for ({equity}, {target}, {vol})"
        assert 0.0 <= usd <= max_usd
        assert isinstance(capped, bool) and reason


def test_sizer_object_zero_vol_and_bad_price_are_safe():
    s = VolTargetSizer(target_daily_vol=0.01, max_position_usd=1e5, qty_step=0.001)
    for _ in range(50):
        s.update(0.0)  # realized vol identically zero
    res = s.size(equity=100_000.0, price=50_000.0)
    assert res.position_usd == 0.0 and res.qty == 0.0
    assert "zero-vol guard" in res.reason
    s.update(0.001)
    for price in (0.0, -5.0, math.nan):
        r = s.size(equity=100_000.0, price=price)
        assert r.qty == 0.0 and math.isfinite(r.position_usd)


# --------------------------------------------------------------------------
# 2) EWMA: NaN immunity and 1e5-update stability
# --------------------------------------------------------------------------


def test_ewma_single_nan_or_inf_return_does_not_poison_estimator():
    """Regression: one NaN return used to turn the variance into NaN forever,
    silently zeroing all future sizing until a manual reset."""
    est = EwmaVol(halflife_bars=10.0, bars_per_day=1440.0)
    est.update(0.001)
    before = est.daily_vol
    n_before = est.n
    for bad in (math.nan, math.inf, -math.inf):
        assert est.update(bad) == pytest.approx(before)
    assert est.n == n_before  # ignored entirely, not counted
    for _ in range(200):
        est.update(0.001)
    assert est.daily_vol == pytest.approx(0.001 * math.sqrt(1440.0), rel=1e-6)


def test_ewma_1e5_constant_updates_converge_exactly():
    est = EwmaVol(halflife_bars=60.0, bars_per_day=1440.0)
    r = 0.001
    n = max(5_000, n_scaled(100_000))  # floor keeps the 1e-9 tolerance meaningful
    t_start = time.perf_counter()
    for _ in range(n):
        est.update(r)
    elapsed = time.perf_counter() - t_start
    assert elapsed < 30.0
    assert est.n == n
    assert est.daily_vol == pytest.approx(r * math.sqrt(1440.0), rel=1e-9)


def test_ewma_1e5_random_updates_track_true_vol():
    rng = np.random.default_rng(SEED)
    true_r = 0.002
    est = EwmaVol(halflife_bars=200.0, bars_per_day=1440.0)
    for ret in rng.normal(0.0, true_r, max(20_000, n_scaled(100_000))):
        est.update(float(ret))
    got = est.daily_vol
    assert math.isfinite(got) and got > 0.0
    assert got == pytest.approx(true_r * math.sqrt(1440.0), rel=0.10)


def test_ewma_long_zero_tail_decays_to_zero_finite():
    est = EwmaVol(halflife_bars=5.0, bars_per_day=1440.0)
    est.update(0.05)
    for _ in range(max(5_000, n_scaled(100_000))):
        est.update(0.0)
    assert 0.0 <= est.daily_vol < 1e-100  # fully decayed, no denormal garbage
    assert math.isfinite(est.daily_vol)


# --------------------------------------------------------------------------
# 3) Limits: daily-stop cascade and kill-switch flapping
# --------------------------------------------------------------------------


def test_daily_stop_cascade_trips_exactly_once_with_pinned_ts():
    stop = DailyStop(daily_stop_pct=0.03)
    stop.update(T0, 100_000.0)
    n = max(600, n_scaled(10_000))  # the -3% line is crossed at tick 300
    trip_ts = None
    trips = 0
    for i in range(1, n + 1):
        ts = T0 + i * NS_PER_S
        state = stop.update(ts, 100_000.0 - i * 10.0)  # -3% reached at i == 300
        if state.halted and trip_ts is None:
            trip_ts = state.ts
            trips += 1
        elif state.halted:
            assert state.ts == trip_ts, "halt re-fired with a new ts"
    assert trips == 1
    assert trip_ts == T0 + 300 * NS_PER_S  # first tick with pnl <= -3000
    assert stop.state.halted and stop.state.ts == trip_ts


def test_daily_stop_boundary_exactly_minus_pct():
    stop = DailyStop(daily_stop_pct=0.03)
    stop.update(T0, 100_000.0)
    assert not stop.update(T0 + 1, 97_000.0000001).halted  # one hair above the line
    stop2 = DailyStop(daily_stop_pct=0.03)
    stop2.update(T0, 100_000.0)
    assert stop2.update(T0 + 1, 97_000.0).halted  # exactly -3% trips


def test_kill_switch_trips_at_exact_error_threshold_only():
    ks = KillSwitch(max_consecutive_errors=5, stale_after_s=5.0)
    for i in range(4):
        assert not ks.record_error(T0 + i).tripped
    state = ks.record_error(T0 + 4)
    assert state.tripped and state.flatten and state.ts == T0 + 4
    # further noise never moves the trip
    for i in range(100):
        ks.record_error(T0 + 10 + i)
        ks.record_success(T0 + 10 + i)
    assert ks.state.ts == T0 + 4


def test_kill_switch_error_success_flapping_never_trips():
    ks = KillSwitch(max_consecutive_errors=2, stale_after_s=5.0)
    n = n_scaled(10_000)
    for i in range(n):
        assert not ks.record_error(T0 + 2 * i).tripped
        ks.record_success(T0 + 2 * i + 1)
    assert not ks.state.tripped
    assert ks.consecutive_errors == 0
    # threshold-1 runs broken by successes never accumulate across the run
    ks2 = KillSwitch(max_consecutive_errors=3, stale_after_s=5.0)
    for i in range(n):
        ks2.record_error(T0 + 3 * i)
        ks2.record_error(T0 + 3 * i + 1)
        ks2.record_success(T0 + 3 * i + 2)
    assert not ks2.state.tripped


def test_kill_switch_freshness_boundary_to_the_nanosecond():
    ks = KillSwitch(max_consecutive_errors=3, stale_after_s=5.0)
    ks.record_market_data(T0)
    assert not ks.check(T0 + 5 * NS_PER_S).tripped  # gap == limit: still fine
    assert ks.check(T0 + 5 * NS_PER_S + 1).tripped  # one ns beyond: dead feed


def test_kill_switch_survives_1e5_out_of_order_ticks():
    ks = KillSwitch(max_consecutive_errors=3, stale_after_s=5.0)
    rng = np.random.default_rng(SEED)
    n = n_scaled(100_000)
    ticks = T0 + rng.integers(0, 4 * NS_PER_S, n)  # shuffled within 4s < limit
    t_start = time.perf_counter()
    for t in ticks:
        ks.record_market_data(int(t))
    assert not ks.check(int(ticks.max()) + NS_PER_S).tripped
    assert time.perf_counter() - t_start < 20.0


# --------------------------------------------------------------------------
# 4) Orders: 10k-transition storm with illegal attempts mixed in
# --------------------------------------------------------------------------


def _mirror_fill(shadow: dict, to: OrderState, fill_qty: float, order_qty: float) -> None:
    """Reference model of the documented fill accounting."""
    if to is OrderState.FILLED and fill_qty <= 0:
        fill_qty = max(0.0, order_qty - shadow["filled"])
    if fill_qty > 0 and to in _FILL_STATES:
        shadow["filled"] += fill_qty


def test_order_storm_10k_mixed_transitions(tmp_path):
    n_attempts = max(500, n_scaled(10_000))
    n_orders = 150
    rng = np.random.default_rng(SEED)
    states = list(OrderState)
    journal = OrderJournal(tmp_path / "storm.jsonl")
    machines = [
        OrderStateMachine(f"o{k}", journal, symbol=SYM, side="BUY", qty=1.0, ts=T0)
        for k in range(n_orders)
    ]
    shadow = {m.order_id: {"state": OrderState.IDLE, "filled": 0.0} for m in machines}

    n_valid = n_invalid = 0
    t_start = time.perf_counter()
    for _ in range(n_attempts):
        m = machines[int(rng.integers(n_orders))]
        allowed = sorted(ALLOWED_TRANSITIONS[m.state])
        if allowed and rng.random() < 0.6:
            dst = allowed[int(rng.integers(len(allowed)))]
        else:
            dst = states[int(rng.integers(len(states)))]
        fill_qty = float(rng.uniform(0.0, 0.05)) if dst in _FILL_STATES else 0.0
        sh = shadow[m.order_id]
        src = m.state
        if dst in ALLOWED_TRANSITIONS[src]:
            m.transition(dst, T0 + n_valid, fill_qty=fill_qty, fill_price=100.0)
            _mirror_fill(sh, dst, fill_qty, 1.0)
            sh["state"] = dst
            n_valid += 1
        else:
            with pytest.raises(InvalidTransition):
                m.transition(dst, T0, fill_qty=fill_qty, fill_price=100.0)
            assert m.state is src, "rejected transition mutated state"
            n_invalid += 1
    elapsed = time.perf_counter() - t_start
    assert elapsed < 60.0
    assert n_valid > 0 and n_invalid > 0
    assert n_valid + n_invalid == n_attempts

    # memory agrees with the reference model for every order
    for m in machines:
        sh = shadow[m.order_id]
        assert m.state is sh["state"]
        assert m.filled_qty == pytest.approx(sh["filled"], abs=1e-9)

    # the journal holds exactly the accepted events, and replay equals memory
    lines = (tmp_path / "storm.jsonl").read_text().splitlines()
    assert len(lines) == n_orders + n_valid  # one genesis line per order + valid events
    replayed = replay_journal(tmp_path / "storm.jsonl")
    assert len(replayed) == n_orders
    for m in machines:
        r = replayed[m.order_id]
        assert r.state is m.state
        assert r.filled_qty == pytest.approx(m.filled_qty, abs=1e-9)


@pytest.mark.parametrize(
    ("setup", "attempt"),
    [
        # fill after cancel
        ((OrderState.PENDING_NEW, OrderState.CANCELED), OrderState.PARTIALLY_FILLED),
        ((OrderState.PENDING_NEW, OrderState.CANCELED), OrderState.FILLED),
        # cancel after fill
        ((OrderState.PENDING_NEW, OrderState.FILLED), OrderState.CANCELED),
        ((OrderState.PENDING_NEW, OrderState.FILLED), OrderState.PENDING_CANCEL),
        # duplicate fill on a terminal order
        ((OrderState.PENDING_NEW, OrderState.FILLED), OrderState.FILLED),
        ((OrderState.PENDING_NEW, OrderState.REJECTED), OrderState.FILLED),
    ],
    ids=lambda v: "-".join(s.value for s in v) if isinstance(v, tuple) else v.value,
)
def test_terminal_orders_reject_1000_late_events(tmp_path, setup, attempt):
    journal = OrderJournal(tmp_path / "late.jsonl")
    m = OrderStateMachine("o1", journal, symbol=SYM, side="BUY", qty=1.0, ts=T0)
    for i, step in enumerate(setup):
        m.transition(step, T0 + i)
    frozen_state = m.state
    frozen_filled = m.filled_qty
    n_lines = len((tmp_path / "late.jsonl").read_text().splitlines())
    for i in range(n_scaled(1000)):
        with pytest.raises(InvalidTransition):
            m.transition(attempt, T0 + 100 + i, fill_qty=0.5, fill_price=100.0)
    assert m.state is frozen_state
    assert m.filled_qty == frozen_filled
    assert len((tmp_path / "late.jsonl").read_text().splitlines()) == n_lines
    assert replay_journal(tmp_path / "late.jsonl")["o1"].state is frozen_state


@pytest.mark.xfail(
    reason=(
        "design: the state machine journals legal PARTIALLY_FILLED self-loops without "
        "validating cumulative fill_qty against the order quantity, so an exchange feed "
        "bug can overfill an order (filled_qty > qty) without any error"
    ),
    strict=True,
)
def test_overfill_beyond_order_qty_is_rejected(tmp_path):
    journal = OrderJournal(tmp_path / "overfill.jsonl")
    m = OrderStateMachine("o1", journal, symbol=SYM, side="BUY", qty=1.0, ts=T0)
    m.transition(OrderState.PENDING_NEW, T0 + 1)
    m.transition(OrderState.PARTIALLY_FILLED, T0 + 2, fill_qty=0.6, fill_price=100.0)
    m.transition(OrderState.PARTIALLY_FILLED, T0 + 3, fill_qty=0.6, fill_price=100.0)
    assert m.filled_qty <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# 5) Reconcile at 1000-order scale
# --------------------------------------------------------------------------


def _local_open(order_id: str, state: OrderState = OrderState.OPEN, filled: float = 0.0):
    return ReplayedOrder(
        order_id=order_id,
        state=state,
        filled_qty=filled,
        meta={"symbol": SYM, "side": "BUY", "qty": 1.0},
    )


def test_reconcile_1000_local_orders_unknown_to_exchange():
    n = n_scaled(1000)
    local = {f"L{i}": _local_open(f"L{i}") for i in range(n)}
    exchange = FakeExchange(position=ExchangePosition(SYM, 0.0, 0.0))
    t_start = time.perf_counter()
    report = reconcile_on_restart(local, exchange, SYM, ts=T0, local_position_qty=0.0)
    assert time.perf_counter() - t_start < 30.0
    assert report.converged
    assert set(report.orders) == set(local)  # nothing lost, nothing invented
    assert all(o.state is OrderState.CANCELED for o in report.orders.values())
    assert sum(1 for a in report.actions if a.kind == "mark_canceled") == n
    assert exchange.canceled == []  # nothing to touch on the exchange side


def test_reconcile_1000_exchange_ghosts_unknown_locally():
    n = n_scaled(1000)
    ghosts = [ExchangeOrder(f"G{i}", SYM, "SELL", 1.0, 100.0, 0.0, "NEW") for i in range(n)]
    exchange = FakeExchange(position=ExchangePosition(SYM, 0.0, 0.0), open_orders=ghosts)
    report = reconcile_on_restart({}, exchange, SYM, ts=T0, local_position_qty=0.0)
    assert report.converged
    assert sorted(exchange.canceled) == sorted(f"G{i}" for i in range(n))
    assert len(set(exchange.canceled)) == n  # each ghost canceled exactly once
    assert exchange.get_open_orders(SYM) == []
    assert sum(1 for a in report.actions if a.kind == "cancel_unknown_order") == n


def test_reconcile_50_50_mix_converges_without_losing_orders():
    n = max(10, n_scaled(1000))
    half = n // 2
    local: dict[str, ReplayedOrder] = {}
    exch_orders: list[ExchangeOrder] = []
    for i in range(n):
        oid = f"L{i}"
        if i < half:
            local[oid] = _local_open(oid)  # exchange lost these
        else:
            # shared: exchange saw extra fills; every 5th had a cancel in flight
            state = OrderState.PENDING_CANCEL if i % 5 == 0 else OrderState.OPEN
            local[oid] = _local_open(oid, state=state)
            exch_orders.append(
                ExchangeOrder(oid, SYM, "BUY", 1.0, 100.0, 0.25, "PARTIALLY_FILLED")
            )
    for i in range(half):  # ghosts on top
        exch_orders.append(ExchangeOrder(f"G{i}", SYM, "SELL", 1.0, 101.0, 0.0, "NEW"))
    # plus one locally terminal order the exchange still shows open
    local["T0"] = _local_open("T0", state=OrderState.FILLED, filled=1.0)
    exch_orders.append(ExchangeOrder("T0", SYM, "BUY", 1.0, 100.0, 1.0, "NEW"))

    exchange = FakeExchange(position=ExchangePosition(SYM, 2.5, 100.0), open_orders=exch_orders)
    report = reconcile_on_restart(local, exchange, SYM, ts=T0, local_position_qty=0.0)

    assert report.converged
    assert set(report.orders) == set(local)  # no local order lost or duplicated
    # lost half canceled locally; ghosts + stale terminal canceled on the exchange
    for i in range(half):
        assert report.orders[f"L{i}"].state is OrderState.CANCELED
    assert set(exchange.canceled) == {f"G{i}" for i in range(half)} | {"T0"} | {
        f"L{i}" for i in range(half, n) if i % 5 == 0  # re-issued in-flight cancels
    }
    assert len(exchange.canceled) == len(set(exchange.canceled))
    # shared survivors adopted the exchange fills exactly once
    survivors = [f"L{i}" for i in range(half, n) if i % 5 != 0]
    for oid in survivors:
        assert report.orders[oid].state is OrderState.PARTIALLY_FILLED
        assert report.orders[oid].filled_qty == pytest.approx(0.25)
    # in-flight cancels got their fills adopted, then were re-canceled
    for i in range(half, n):
        if i % 5 == 0:
            assert report.orders[f"L{i}"].state is OrderState.CANCELED
            assert report.orders[f"L{i}"].filled_qty == pytest.approx(0.25)
    assert report.position is not None and report.position.qty == pytest.approx(2.5)
    assert any(m.kind == "position" for m in report.mismatches)

    # a second pass right after is a clean no-op: fully idempotent
    report2 = reconcile_on_restart(
        local, exchange, SYM, ts=T0 + NS_PER_S, local_position_qty=2.5
    )
    assert report2.converged
    assert report2.actions == [] and report2.mismatches == []


def test_reconcile_with_journal_replays_converged_state(tmp_path):
    """Journaled variant at 100 orders: after reconcile, a fresh replay of the
    journal reproduces the converged state exactly."""
    n = 100
    jpath = tmp_path / "reconcile.jsonl"
    journal = OrderJournal(jpath)
    for i in range(n):
        m = OrderStateMachine(f"L{i}", journal, symbol=SYM, side="BUY", qty=1.0, ts=T0)
        m.transition(OrderState.PENDING_NEW, T0 + i)
        m.transition(OrderState.OPEN, T0 + i + 1)
    exch_orders = [
        ExchangeOrder(f"L{i}", SYM, "BUY", 1.0, 100.0, 0.5, "PARTIALLY_FILLED")
        for i in range(n // 2)
    ]  # the other half vanished from the exchange
    exchange = FakeExchange(position=ExchangePosition(SYM, 25.0, 100.0), open_orders=exch_orders)

    replayed = replay_journal(jpath)
    report = reconcile_on_restart(
        replayed, exchange, SYM, ts=T0 + NS_PER_S, local_position_qty=0.0,
        journal=OrderJournal(jpath),
    )
    assert report.converged
    replayed2 = replay_journal(jpath)
    assert set(replayed2) == set(report.orders)
    for oid, o in report.orders.items():
        assert replayed2[oid].state is o.state
        assert replayed2[oid].filled_qty == pytest.approx(o.filled_qty)
    # every journaled transition, including reconcile's, is table-legal
    import json

    for line in jpath.read_text().splitlines():
        entry = json.loads(line)
        if entry["kind"] == "transition":
            src = OrderState(entry["from"])
            dst = OrderState(entry["to"])
            assert dst in ALLOWED_TRANSITIONS[src]
