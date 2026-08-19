"""M9 order state machine: full transition matrix, write-ahead journal, replay.

The matrix test is parametrized over EVERY (state, target) pair derived from
orders.ALLOWED_TRANSITIONS — the same table the machine validates against —
so the tests can never drift from the code.
"""

from __future__ import annotations

import json

import pytest

from trading_system.risk.orders import (
    ALLOWED_TRANSITIONS,
    OPEN_STATES,
    TERMINAL_STATES,
    InvalidTransition,
    JournalCorrupted,
    OrderJournal,
    OrderState,
    OrderStateMachine,
    replay_journal,
)

TS0 = 1_755_600_000_000_000_000  # UTC ns


def T(seconds: float) -> int:
    return TS0 + int(seconds * 1_000_000_000)


# A legal path from IDLE into every state, used to set up the matrix test.
PATHS: dict[OrderState, tuple[OrderState, ...]] = {
    OrderState.IDLE: (),
    OrderState.PENDING_NEW: (OrderState.PENDING_NEW,),
    OrderState.OPEN: (OrderState.PENDING_NEW, OrderState.OPEN),
    OrderState.PARTIALLY_FILLED: (OrderState.PENDING_NEW, OrderState.PARTIALLY_FILLED),
    OrderState.FILLED: (OrderState.PENDING_NEW, OrderState.FILLED),
    OrderState.PENDING_CANCEL: (
        OrderState.PENDING_NEW,
        OrderState.OPEN,
        OrderState.PENDING_CANCEL,
    ),
    OrderState.CANCELED: (OrderState.PENDING_NEW, OrderState.CANCELED),
    OrderState.REJECTED: (OrderState.PENDING_NEW, OrderState.REJECTED),
}

# Single source of truth: the full (state, target) matrix from the table.
MATRIX = [
    (src, dst, dst in ALLOWED_TRANSITIONS[src]) for src in OrderState for dst in OrderState
]


def make_machine(tmp_path, state: OrderState, order_id: str = "o1") -> OrderStateMachine:
    journal = OrderJournal(tmp_path / "journal.jsonl")
    m = OrderStateMachine(order_id, journal, symbol="BTCUSDT", side="BUY", qty=1.0, ts=T(0))
    for i, step in enumerate(PATHS[state]):
        m.transition(step, T(i + 1))
    assert m.state is state
    return m


def journal_lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_table_covers_every_state():
    assert set(ALLOWED_TRANSITIONS) == set(OrderState)


def test_setup_paths_are_legal_and_reach_every_state():
    for state, path in PATHS.items():
        cur = OrderState.IDLE
        for step in path:
            assert step in ALLOWED_TRANSITIONS[cur], f"illegal setup path for {state}"
            cur = step
        assert cur is state


def test_terminal_and_open_states_derive_from_table():
    assert TERMINAL_STATES == {s for s, targets in ALLOWED_TRANSITIONS.items() if not targets}
    assert TERMINAL_STATES == {OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED}
    assert OPEN_STATES.isdisjoint(TERMINAL_STATES)
    assert OrderState.IDLE not in OPEN_STATES


@pytest.mark.parametrize(
    ("src", "dst", "allowed"),
    MATRIX,
    ids=[f"{src.value}->{dst.value}" for src, dst, _ in MATRIX],
)
def test_transition_matrix(tmp_path, src: OrderState, dst: OrderState, allowed: bool):
    m = make_machine(tmp_path, src)
    n_before = len(journal_lines(m.journal.path))
    if allowed:
        assert m.transition(dst, T(100)) is dst
        assert m.state is dst
        assert len(journal_lines(m.journal.path)) == n_before + 1
        # replay agrees with memory after the flip
        assert replay_journal(m.journal.path)["o1"].state is dst
    else:
        with pytest.raises(InvalidTransition) as exc:
            m.transition(dst, T(100))
        assert exc.value.src is src and exc.value.dst is dst
        assert m.state is src  # forbidden transition mutates nothing
        assert len(journal_lines(m.journal.path)) == n_before  # and journals nothing


# --------------------------------------------------------------------------
# Write-ahead property
# --------------------------------------------------------------------------


class SimulatedCrash(RuntimeError):
    pass


class CrashingJournal(OrderJournal):
    """Real journal that can crash immediately before or after the durable write."""

    def __init__(self, path, crash_point: str) -> None:
        super().__init__(path)
        self.armed = False
        self.crash_point = crash_point

    def append(self, entry: dict) -> None:
        if self.armed and self.crash_point == "before":
            raise SimulatedCrash("power lost before the journal write")
        super().append(entry)
        if self.armed and self.crash_point == "after":
            raise SimulatedCrash("power lost after the journal write, before memory flip")


def test_journal_written_before_memory_flip(tmp_path):
    """Crash between journal append and the in-memory flip: journal is AHEAD."""
    journal = CrashingJournal(tmp_path / "j.jsonl", crash_point="after")
    m = OrderStateMachine("o1", journal, symbol="BTCUSDT", side="BUY", qty=1.0, ts=T(0))
    m.transition(OrderState.PENDING_NEW, T(1))
    journal.armed = True
    with pytest.raises(SimulatedCrash):
        m.transition(OrderState.OPEN, T(2))
    # memory never flipped ...
    assert m.state is OrderState.PENDING_NEW
    # ... but the journal already holds the transition: write-ahead order proven
    replayed = replay_journal(journal.path)["o1"]
    assert replayed.state is OrderState.OPEN
    assert replayed.seq == 2


def test_crash_before_journal_write_leaves_both_unchanged(tmp_path):
    journal = CrashingJournal(tmp_path / "j.jsonl", crash_point="before")
    m = OrderStateMachine("o1", journal, symbol="BTCUSDT", side="BUY", qty=1.0, ts=T(0))
    m.transition(OrderState.PENDING_NEW, T(1))
    journal.armed = True
    with pytest.raises(SimulatedCrash):
        m.transition(OrderState.OPEN, T(2))
    assert m.state is OrderState.PENDING_NEW
    assert replay_journal(journal.path)["o1"].state is OrderState.PENDING_NEW


def test_restart_from_journal_continues_sequence(tmp_path):
    path = tmp_path / "j.jsonl"
    journal = OrderJournal(path)
    m = OrderStateMachine("o1", journal, symbol="BTCUSDT", side="BUY", qty=1.0, ts=T(0))
    m.transition(OrderState.PENDING_NEW, T(1))
    m.transition(OrderState.OPEN, T(2))
    del m  # crash: in-memory state gone

    restored = OrderStateMachine.from_replay(replay_journal(path)["o1"], OrderJournal(path))
    assert restored.state is OrderState.OPEN
    restored.transition(OrderState.FILLED, T(3))
    entries = journal_lines(path)
    seqs = [e["seq"] for e in entries]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)  # strictly monotonic
    assert replay_journal(path)["o1"].state is OrderState.FILLED


# --------------------------------------------------------------------------
# Fill accounting and replay equivalence
# --------------------------------------------------------------------------


def test_partial_fills_accumulate_and_average(tmp_path):
    m = make_machine(tmp_path, OrderState.OPEN)
    m.transition(OrderState.PARTIALLY_FILLED, T(10), fill_qty=0.4, fill_price=100.0)
    m.transition(OrderState.PARTIALLY_FILLED, T(11), fill_qty=0.4, fill_price=110.0)
    assert m.filled_qty == pytest.approx(0.8)
    assert m.avg_fill_price == pytest.approx(105.0)
    m.transition(OrderState.FILLED, T(12))  # no qty: fills the remainder
    assert m.filled_qty == pytest.approx(1.0)
    assert m.avg_fill_price == pytest.approx(105.0)  # no price on the remainder

    replayed = replay_journal(m.journal.path)["o1"]
    assert replayed.state is OrderState.FILLED
    assert replayed.filled_qty == pytest.approx(m.filled_qty)
    assert replayed.avg_fill_price == pytest.approx(m.avg_fill_price)


def test_fill_during_pending_cancel_self_loop_is_counted(tmp_path):
    """Regression: fills on the PENDING_CANCEL self-loop must not be dropped."""
    m = make_machine(tmp_path, OrderState.PENDING_CANCEL)
    m.transition(OrderState.PENDING_CANCEL, T(10), fill_qty=0.3, fill_price=100.0)
    assert m.filled_qty == pytest.approx(0.3)
    assert m.avg_fill_price == pytest.approx(100.0)
    m.transition(OrderState.CANCELED, T(11))

    replayed = replay_journal(m.journal.path)["o1"]
    assert replayed.state is OrderState.CANCELED
    assert replayed.filled_qty == pytest.approx(0.3)
    assert replayed.avg_fill_price == pytest.approx(100.0)


def test_filled_from_pending_cancel_fills_remainder(tmp_path):
    m = make_machine(tmp_path, OrderState.PENDING_CANCEL)
    m.transition(OrderState.FILLED, T(10))
    assert m.filled_qty == pytest.approx(1.0)
    assert replay_journal(m.journal.path)["o1"].filled_qty == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Journal robustness
# --------------------------------------------------------------------------


def test_torn_final_line_is_skipped(tmp_path):
    m = make_machine(tmp_path, OrderState.OPEN)
    path = m.journal.path
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"kind":"transition","order_id":"o1","seq":3,"ts":123,"fr')  # torn mid-write
    replayed = replay_journal(path)["o1"]
    assert replayed.state is OrderState.OPEN  # last full entry wins


def test_corrupt_middle_line_raises(tmp_path):
    m = make_machine(tmp_path, OrderState.OPEN)
    path = m.journal.path
    with open(path, "a", encoding="utf-8") as f:
        f.write("NOT JSON\n")
    m.transition(OrderState.FILLED, T(10))  # a valid line lands after the bad one
    with pytest.raises(JournalCorrupted):
        replay_journal(path)


def test_replay_missing_file_is_empty(tmp_path):
    assert replay_journal(tmp_path / "absent.jsonl") == {}


def test_multiple_orders_share_one_journal(tmp_path):
    journal = OrderJournal(tmp_path / "j.jsonl")
    a = OrderStateMachine("a", journal, symbol="BTCUSDT", side="BUY", qty=1.0, ts=T(0))
    b = OrderStateMachine("b", journal, symbol="SOLUSDT", side="SELL", qty=2.0, ts=T(0))
    a.transition(OrderState.PENDING_NEW, T(1))
    b.transition(OrderState.PENDING_NEW, T(1))
    a.transition(OrderState.REJECTED, T(2))
    replayed = replay_journal(journal.path)
    assert replayed["a"].state is OrderState.REJECTED
    assert replayed["b"].state is OrderState.PENDING_NEW
    assert replayed["b"].meta["symbol"] == "SOLUSDT"
