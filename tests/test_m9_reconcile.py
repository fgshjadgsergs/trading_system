"""M9 crash drill: kill the process mid-order, restart from the journal,
reconcile against a divergent FakeExchange — the reconciled state must EQUAL
exchange state and the report must list every divergence.
"""

from __future__ import annotations

import json

import pytest

from trading_system.risk.orders import (
    ALLOWED_TRANSITIONS,
    OrderJournal,
    OrderState,
    OrderStateMachine,
    replay_journal,
)
from trading_system.risk.reconcile import (
    ExchangeOrder,
    ExchangePosition,
    FakeExchange,
    reconcile_on_restart,
)

TS0 = 1_755_600_000_000_000_000
SYM = "BTCUSDT"


def T(seconds: float) -> int:
    return TS0 + int(seconds * 1_000_000_000)


def assert_journal_transitions_legal(path) -> None:
    """Every transition the journal holds must obey ALLOWED_TRANSITIONS."""
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        if entry["kind"] != "transition":
            continue
        src = OrderState(entry["from"])
        dst = OrderState(entry["to"])
        assert dst in ALLOWED_TRANSITIONS[src], f"illegal journaled transition {src}->{dst}"


def assert_matches_exchange(orders, exchange, symbol) -> None:
    """Local open orders exactly mirror the exchange's open orders."""
    exch_open = {o.order_id: o for o in exchange.get_open_orders(symbol)}
    local_open = {oid: o for oid, o in orders.items() if o.is_open}
    assert set(local_open) == set(exch_open)
    for oid, eo in exch_open.items():
        assert local_open[oid].filled_qty == pytest.approx(eo.filled_qty)


def start_order_flow(journal: OrderJournal) -> None:
    """The scripted pre-crash flow: three orders in flight, then the kill."""
    a = OrderStateMachine("A", journal, symbol=SYM, side="BUY", qty=1.0, ts=T(0.0))
    a.transition(OrderState.PENDING_NEW, T(0.1))
    a.transition(OrderState.OPEN, T(0.4))
    a.transition(OrderState.PARTIALLY_FILLED, T(1.2), fill_qty=0.4, fill_price=50_000.0)

    b = OrderStateMachine("B", journal, symbol=SYM, side="SELL", qty=0.5, ts=T(1.5))
    b.transition(OrderState.PENDING_NEW, T(1.5))  # ack never arrives

    c = OrderStateMachine("C", journal, symbol=SYM, side="BUY", qty=0.6, ts=T(1.8))
    c.transition(OrderState.PENDING_NEW, T(1.8))
    c.transition(OrderState.OPEN, T(2.0))
    c.transition(OrderState.PENDING_CANCEL, T(2.4))  # cancel in flight at the kill
    # CRASH: a, b, c go out of scope — all in-memory state is dropped.


def divergent_exchange() -> FakeExchange:
    """Exchange truth after the crash, diverged on every axis."""
    return FakeExchange(
        position=ExchangePosition(SYM, 0.9, 50_010.0),
        open_orders=[
            # A got more fills while we were down
            ExchangeOrder("A", SYM, "BUY", 1.0, 50_000.0, 0.7, "PARTIALLY_FILLED"),
            # C's cancel never landed AND it got a fill meanwhile
            ExchangeOrder("C", SYM, "BUY", 0.6, 49_900.0, 0.2, "PARTIALLY_FILLED"),
            # G is unknown to the journal entirely (ghost order)
            ExchangeOrder("G", SYM, "SELL", 0.3, 50_500.0, 0.0, "NEW"),
        ],
    )


def test_crash_drill_reconciles_to_exchange_state(tmp_path):
    jpath = tmp_path / "journal.jsonl"
    start_order_flow(OrderJournal(jpath))
    exchange = divergent_exchange()

    # Restart: replay the journal, reconcile against the exchange.
    replayed = replay_journal(jpath)
    assert replayed["A"].state is OrderState.PARTIALLY_FILLED
    assert replayed["C"].state is OrderState.PENDING_CANCEL
    report = reconcile_on_restart(
        replayed, exchange, SYM, ts=T(3.5), local_position_qty=0.4, journal=OrderJournal(jpath)
    )

    # Reconciled state EQUALS exchange state.
    assert report.converged
    assert_matches_exchange(report.orders, exchange, SYM)
    assert report.orders["A"].filled_qty == pytest.approx(0.7)
    assert report.orders["B"].state is OrderState.CANCELED  # never reached the exchange
    assert report.orders["C"].state is OrderState.CANCELED  # cancel re-issued
    assert report.orders["C"].filled_qty == pytest.approx(0.2)  # fill adopted first
    assert "G" in exchange.canceled  # ghost order canceled on the exchange
    assert report.position is not None and report.position.qty == pytest.approx(0.9)

    # The report lists every divergence.
    kinds = {m.kind for m in report.mismatches}
    assert {
        "unknown_on_exchange",
        "missing_on_exchange",
        "fill_qty",
        "state",
        "position",
    } <= kinds
    by_kind_order = {(m.kind, m.order_id) for m in report.mismatches}
    assert ("unknown_on_exchange", "G") in by_kind_order
    assert ("missing_on_exchange", "B") in by_kind_order
    assert ("fill_qty", "A") in by_kind_order
    assert ("fill_qty", "C") in by_kind_order
    assert ("position", None) in by_kind_order
    action_kinds = {a.kind for a in report.actions}
    assert {
        "cancel_unknown_order",
        "mark_canceled",
        "adopt_fill",
        "reissue_cancel",
        "adopt_position",
    } <= action_kinds

    # The journal converged too: a second restart replays the reconciled state,
    # and every journaled transition (incl. reconcile's) is table-legal.
    assert_journal_transitions_legal(jpath)
    replayed2 = replay_journal(jpath)
    for oid, o in report.orders.items():
        assert replayed2[oid].state is o.state
        assert replayed2[oid].filled_qty == pytest.approx(o.filled_qty)

    # A second reconcile right after is a clean no-op.
    report2 = reconcile_on_restart(
        replayed2,
        exchange,
        SYM,
        ts=T(4.0),
        local_position_qty=report.position.qty,
        journal=OrderJournal(jpath),
    )
    assert report2.converged
    assert report2.mismatches == [] and report2.actions == []


def test_ack_lost_adopts_open(tmp_path):
    jpath = tmp_path / "j.jsonl"
    journal = OrderJournal(jpath)
    o = OrderStateMachine("A", journal, symbol=SYM, side="BUY", qty=1.0, ts=T(0))
    o.transition(OrderState.PENDING_NEW, T(1))
    del o  # crash before the ack arrived

    exchange = FakeExchange(
        open_orders=[ExchangeOrder("A", SYM, "BUY", 1.0, 50_000.0, 0.0, "NEW")]
    )
    report = reconcile_on_restart(
        replay_journal(jpath), exchange, SYM, ts=T(5), journal=OrderJournal(jpath)
    )
    assert report.converged
    assert report.orders["A"].state is OrderState.OPEN
    assert any(a.kind == "adopt_ack" for a in report.actions)
    assert_journal_transitions_legal(jpath)


def test_cancel_landed_while_down_marks_canceled(tmp_path):
    jpath = tmp_path / "j.jsonl"
    journal = OrderJournal(jpath)
    o = OrderStateMachine("A", journal, symbol=SYM, side="BUY", qty=1.0, ts=T(0))
    o.transition(OrderState.PENDING_NEW, T(1))
    o.transition(OrderState.OPEN, T(2))
    o.transition(OrderState.PENDING_CANCEL, T(3))
    del o

    exchange = FakeExchange()  # cancel DID land: order gone, flat position
    report = reconcile_on_restart(
        replay_journal(jpath), exchange, SYM, ts=T(5), journal=OrderJournal(jpath)
    )
    assert report.converged
    assert report.orders["A"].state is OrderState.CANCELED
    assert any(a.kind == "mark_canceled" for a in report.actions)
    assert_journal_transitions_legal(jpath)


def test_reconcile_without_journal_still_converges_memory(tmp_path):
    jpath = tmp_path / "j.jsonl"
    journal = OrderJournal(jpath)
    o = OrderStateMachine("A", journal, symbol=SYM, side="BUY", qty=1.0, ts=T(0))
    o.transition(OrderState.PENDING_NEW, T(1))
    del o
    exchange = FakeExchange(
        open_orders=[ExchangeOrder("A", SYM, "BUY", 1.0, 50_000.0, 0.0, "NEW")]
    )
    report = reconcile_on_restart(replay_journal(jpath), exchange, SYM, ts=T(5), journal=None)
    assert report.converged
    assert report.orders["A"].state is OrderState.OPEN


def test_clean_state_reconciles_with_no_findings(tmp_path):
    jpath = tmp_path / "j.jsonl"
    journal = OrderJournal(jpath)
    o = OrderStateMachine("A", journal, symbol=SYM, side="BUY", qty=1.0, ts=T(0))
    o.transition(OrderState.PENDING_NEW, T(1))
    o.transition(OrderState.OPEN, T(2))
    del o
    exchange = FakeExchange(
        position=ExchangePosition(SYM, 0.0, 0.0),
        open_orders=[ExchangeOrder("A", SYM, "BUY", 1.0, 50_000.0, 0.0, "NEW")],
    )
    report = reconcile_on_restart(
        replay_journal(jpath), exchange, SYM, ts=T(5), local_position_qty=0.0
    )
    assert report.converged
    assert report.mismatches == [] and report.actions == []
    assert exchange.canceled == []
