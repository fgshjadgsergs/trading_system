"""Order state machine with a write-ahead JSONL journal.

States and the allowed-transition table are the single source of truth (the
state diagram in reports.py is drawn from ALLOWED_TRANSITIONS). Every
transition is appended to a persistent JSONL journal BEFORE the in-memory
state flips, so a crash at any point leaves a journal that is at or ahead of
memory — replay_journal() reconstructs the last durable state for
reconciliation against exchange truth.
"""

from __future__ import annotations

import enum
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


class OrderState(enum.StrEnum):
    IDLE = "IDLE"
    PENDING_NEW = "PENDING_NEW"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    PENDING_CANCEL = "PENDING_CANCEL"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


# Single source of truth for what an order may do. Self-loops model repeated
# partial fills and fills arriving while a cancel is in flight.
ALLOWED_TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.IDLE: frozenset({OrderState.PENDING_NEW}),
    OrderState.PENDING_NEW: frozenset(
        {
            OrderState.OPEN,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.CANCELED,
        }
    ),
    OrderState.OPEN: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.PENDING_CANCEL,
            OrderState.CANCELED,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.PENDING_CANCEL,
            OrderState.CANCELED,
        }
    ),
    OrderState.PENDING_CANCEL: frozenset(
        {OrderState.PENDING_CANCEL, OrderState.CANCELED, OrderState.FILLED}
    ),
    OrderState.FILLED: frozenset(),
    OrderState.CANCELED: frozenset(),
    OrderState.REJECTED: frozenset(),
}

TERMINAL_STATES = frozenset(s for s, targets in ALLOWED_TRANSITIONS.items() if not targets)
OPEN_STATES = frozenset(
    {
        OrderState.PENDING_NEW,
        OrderState.OPEN,
        OrderState.PARTIALLY_FILLED,
        OrderState.PENDING_CANCEL,
    }
)


class InvalidTransition(Exception):
    """Raised on a transition not present in ALLOWED_TRANSITIONS."""

    def __init__(self, order_id: str, src: OrderState, dst: OrderState) -> None:
        self.order_id = order_id
        self.src = src
        self.dst = dst
        super().__init__(f"order {order_id}: {src} -> {dst} is not an allowed transition")


class JournalCorrupted(Exception):
    """Raised when a journal line other than a torn final line fails to parse."""


class OrderJournal:
    """Append-only JSONL journal; each append is flushed and fsynced."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: dict) -> None:
        line = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


@dataclass
class ReplayedOrder:
    """State of one order reconstructed from the journal."""

    order_id: str
    state: OrderState = OrderState.IDLE
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    seq: int = -1
    last_ts: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.state in OPEN_STATES


class OrderStateMachine:
    """One order's lifecycle; journals every event before mutating memory."""

    def __init__(
        self,
        order_id: str,
        journal: OrderJournal,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float | None = None,
        ts: int = 0,
        _restored: ReplayedOrder | None = None,
    ) -> None:
        self.order_id = order_id
        self.journal = journal
        self.meta = {"symbol": symbol, "side": side, "qty": qty, "price": price}
        if _restored is None:
            self._state = OrderState.IDLE
            self.filled_qty = 0.0
            self.avg_fill_price: float | None = None
            self._seq = 0
            # Genesis record: order registered with the journal (write-ahead).
            self.journal.append(
                {
                    "kind": "created",
                    "order_id": order_id,
                    "seq": 0,
                    "ts": ts,
                    "state": str(OrderState.IDLE),
                    "meta": self.meta,
                }
            )
        else:
            self._state = _restored.state
            self.filled_qty = _restored.filled_qty
            self.avg_fill_price = _restored.avg_fill_price
            self._seq = _restored.seq

    @classmethod
    def from_replay(cls, replayed: ReplayedOrder, journal: OrderJournal) -> OrderStateMachine:
        """Restore a machine from replay_journal output without re-journaling."""
        meta = replayed.meta
        return cls(
            replayed.order_id,
            journal,
            symbol=meta.get("symbol", ""),
            side=meta.get("side", ""),
            qty=meta.get("qty", 0.0),
            price=meta.get("price"),
            _restored=replayed,
        )

    @property
    def state(self) -> OrderState:
        return self._state

    @property
    def is_terminal(self) -> bool:
        return self._state in TERMINAL_STATES

    def transition(
        self,
        to: OrderState,
        ts: int,
        *,
        reason: str = "",
        fill_qty: float = 0.0,
        fill_price: float | None = None,
    ) -> OrderState:
        """Validate, journal (write-ahead), then flip the in-memory state.

        A crash after the journal append but before the flip leaves the
        journal one step ahead of memory — replay treats the journal as truth.
        """
        if to not in ALLOWED_TRANSITIONS[self._state]:
            raise InvalidTransition(self.order_id, self._state, to)
        seq = self._seq + 1
        self.journal.append(
            {
                "kind": "transition",
                "order_id": self.order_id,
                "seq": seq,
                "ts": ts,
                "from": str(self._state),
                "to": str(to),
                "reason": reason,
                "fill_qty": fill_qty,
                "fill_price": fill_price,
            }
        )
        self._seq = seq
        self._apply_fill(to, fill_qty, fill_price)
        self._state = to
        return self._state

    def _apply_fill(self, to: OrderState, fill_qty: float, fill_price: float | None) -> None:
        if to is OrderState.FILLED and fill_qty <= 0:
            fill_qty = max(0.0, self.meta["qty"] - self.filled_qty)  # fill the remainder
        if fill_qty > 0 and to in (OrderState.PARTIALLY_FILLED, OrderState.FILLED):
            prev = self.filled_qty
            self.filled_qty = prev + fill_qty
            if fill_price is not None:
                if self.avg_fill_price is None or prev <= 0:
                    self.avg_fill_price = fill_price
                else:
                    self.avg_fill_price = (
                        self.avg_fill_price * prev + fill_price * fill_qty
                    ) / self.filled_qty


def _apply_entry(orders: dict[str, ReplayedOrder], entry: dict) -> None:
    oid = entry["order_id"]
    if entry["kind"] == "created":
        orders[oid] = ReplayedOrder(
            order_id=oid, seq=entry["seq"], last_ts=entry["ts"], meta=entry.get("meta", {})
        )
        return
    o = orders.setdefault(oid, ReplayedOrder(order_id=oid))
    to = OrderState(entry["to"])
    fill_qty = float(entry.get("fill_qty") or 0.0)
    fill_price = entry.get("fill_price")
    if to is OrderState.FILLED and fill_qty <= 0:
        fill_qty = max(0.0, float(o.meta.get("qty", 0.0)) - o.filled_qty)
    if fill_qty > 0 and to in (OrderState.PARTIALLY_FILLED, OrderState.FILLED):
        prev = o.filled_qty
        o.filled_qty = prev + fill_qty
        if fill_price is not None:
            if o.avg_fill_price is None or prev <= 0:
                o.avg_fill_price = float(fill_price)
            else:
                o.avg_fill_price = (o.avg_fill_price * prev + float(fill_price) * fill_qty) / (
                    o.filled_qty
                )
    o.state = to
    o.seq = entry["seq"]
    o.last_ts = entry["ts"]


def replay_journal(path: str | Path) -> dict[str, ReplayedOrder]:
    """Reconstruct per-order state from the journal.

    A torn FINAL line (crash mid-write) is skipped; a malformed line anywhere
    else raises JournalCorrupted.
    """
    path = Path(path)
    orders: dict[str, ReplayedOrder] = {}
    if not path.exists():
        return orders
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            if i == len(lines) - 1:
                break  # torn last line from a crash mid-append
            raise JournalCorrupted(f"{path}: bad journal line {i + 1}") from e
        _apply_entry(orders, entry)
    return orders
