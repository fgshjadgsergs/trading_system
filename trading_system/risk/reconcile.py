"""Restart reconciliation: converge journal-replayed state to exchange truth.

The exchange client sits behind a small protocol; tests and drills use
FakeExchange. The real futures-testnet client is an ops integration item —
it only needs to implement ExchangeClient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from trading_system.risk.orders import (
    OrderJournal,
    OrderState,
    ReplayedOrder,
)

_QTY_TOL = 1e-9


@dataclass(frozen=True, slots=True)
class ExchangeOrder:
    """Open order as reported by the exchange."""

    order_id: str
    symbol: str
    side: str
    qty: float
    price: float | None
    filled_qty: float
    status: str  # "NEW" | "PARTIALLY_FILLED"


@dataclass(frozen=True, slots=True)
class ExchangePosition:
    symbol: str
    qty: float  # signed coins; 0.0 == flat
    entry_price: float


class ExchangeClient(Protocol):
    """Minimal read/cancel surface needed to reconcile after a restart."""

    def get_position(self, symbol: str) -> ExchangePosition: ...

    def get_open_orders(self, symbol: str) -> list[ExchangeOrder]: ...

    def cancel_order(self, symbol: str, order_id: str) -> bool: ...


class FakeExchange:
    """Injectable in-memory exchange: a position and a set of open orders."""

    def __init__(
        self,
        position: ExchangePosition | None = None,
        open_orders: list[ExchangeOrder] | None = None,
    ) -> None:
        self._positions: dict[str, ExchangePosition] = {}
        if position is not None:
            self._positions[position.symbol] = position
        self._orders: dict[str, ExchangeOrder] = {o.order_id: o for o in (open_orders or [])}
        self.canceled: list[str] = []

    def get_position(self, symbol: str) -> ExchangePosition:
        return self._positions.get(symbol, ExchangePosition(symbol, 0.0, 0.0))

    def get_open_orders(self, symbol: str) -> list[ExchangeOrder]:
        return [o for o in self._orders.values() if o.symbol == symbol]

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        if order_id in self._orders:
            del self._orders[order_id]
            self.canceled.append(order_id)
            return True
        return False


@dataclass(frozen=True, slots=True)
class Action:
    """One corrective step taken during reconciliation."""

    kind: str  # cancel_unknown_order | mark_canceled | adopt_ack | adopt_fill | adopt_position
    order_id: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class Mismatch:
    """One local-vs-exchange divergence found during reconciliation."""

    kind: str  # missing_on_exchange | unknown_on_exchange | fill_qty | state | position
    order_id: str | None
    local: str
    exchange: str


@dataclass
class ReconcileReport:
    """Outcome of reconcile_on_restart: exchange truth adopted locally."""

    ts: int
    symbol: str
    actions: list[Action] = field(default_factory=list)
    mismatches: list[Mismatch] = field(default_factory=list)
    position: ExchangePosition | None = None
    orders: dict[str, ReplayedOrder] = field(default_factory=dict)
    converged: bool = False


_STATUS_TO_STATE = {
    "NEW": OrderState.OPEN,
    "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
}


def _journal_transition(
    journal: OrderJournal | None,
    order: ReplayedOrder,
    to: OrderState,
    ts: int,
    reason: str,
    fill_qty: float = 0.0,
    fill_price: float | None = None,
) -> None:
    """Append a reconciliation transition so the journal converges too."""
    if journal is None:
        return
    order.seq += 1
    journal.append(
        {
            "kind": "transition",
            "order_id": order.order_id,
            "seq": order.seq,
            "ts": ts,
            "from": str(order.state),
            "to": str(to),
            "reason": reason,
            "fill_qty": fill_qty,
            "fill_price": fill_price,
        }
    )


def reconcile_on_restart(
    journal_state: dict[str, ReplayedOrder],
    client: ExchangeClient,
    symbol: str,
    *,
    ts: int,
    local_position_qty: float = 0.0,
    journal: OrderJournal | None = None,
) -> ReconcileReport:
    """Converge journal-replayed local state to exchange truth.

    - exchange open orders unknown locally (or locally terminal) are canceled;
    - local open orders missing on the exchange are marked CANCELED;
    - ack/fill divergences on shared orders adopt the exchange's view;
    - the exchange position is adopted; any delta vs local belief is flagged.

    When a journal is passed, every adopted change is appended to it, so a
    subsequent replay reproduces the converged state.
    """
    report = ReconcileReport(ts=ts, symbol=symbol)
    exch_orders = {o.order_id: o for o in client.get_open_orders(symbol)}
    local_open = {
        oid: o for oid, o in journal_state.items() if o.is_open and o.meta.get("symbol") == symbol
    }

    # 1) Orders the exchange has but we do not consider open -> cancel them.
    for oid, eo in list(exch_orders.items()):
        if oid in local_open:
            continue
        local_desc = str(journal_state[oid].state) if oid in journal_state else "absent"
        report.mismatches.append(
            Mismatch(kind="unknown_on_exchange", order_id=oid, local=local_desc, exchange=eo.status)
        )
        client.cancel_order(symbol, oid)
        report.actions.append(
            Action(
                kind="cancel_unknown_order",
                order_id=oid,
                detail=f"exchange showed {eo.status} not open locally ({local_desc}); canceled",
            )
        )
        del exch_orders[oid]

    # 2) Local open orders the exchange no longer has -> mark canceled locally.
    for oid, lo in local_open.items():
        if oid in exch_orders:
            continue
        report.mismatches.append(
            Mismatch(kind="missing_on_exchange", order_id=oid, local=str(lo.state), exchange="absent")
        )
        _journal_transition(journal, lo, OrderState.CANCELED, ts, "reconcile: not on exchange")
        lo.state = OrderState.CANCELED
        lo.last_ts = ts
        report.actions.append(
            Action(kind="mark_canceled", order_id=oid, detail="open locally, absent on exchange")
        )

    # 3) Shared orders: adopt exchange ack state and fill quantity.
    for oid, eo in exch_orders.items():
        lo = local_open[oid]
        target_state = _STATUS_TO_STATE.get(eo.status, OrderState.OPEN)
        if lo.state is OrderState.PENDING_NEW and target_state in (
            OrderState.OPEN,
            OrderState.PARTIALLY_FILLED,
        ):
            report.mismatches.append(
                Mismatch(kind="state", order_id=oid, local=str(lo.state), exchange=eo.status)
            )
            _journal_transition(journal, lo, OrderState.OPEN, ts, "reconcile: ack lost")
            lo.state = OrderState.OPEN
            lo.last_ts = ts
            report.actions.append(
                Action(kind="adopt_ack", order_id=oid, detail="exchange acked while we were down")
            )
        if eo.filled_qty - lo.filled_qty > _QTY_TOL:
            delta = eo.filled_qty - lo.filled_qty
            report.mismatches.append(
                Mismatch(
                    kind="fill_qty",
                    order_id=oid,
                    local=f"{lo.filled_qty:g}",
                    exchange=f"{eo.filled_qty:g}",
                )
            )
            _journal_transition(
                journal,
                lo,
                OrderState.PARTIALLY_FILLED,
                ts,
                "reconcile: adopt exchange fills",
                fill_qty=delta,
                fill_price=eo.price,
            )
            lo.filled_qty = eo.filled_qty
            lo.state = OrderState.PARTIALLY_FILLED
            lo.last_ts = ts
            report.actions.append(
                Action(kind="adopt_fill", order_id=oid, detail=f"adopted fill delta {delta:g}")
            )
        elif lo.state is OrderState.PARTIALLY_FILLED and target_state is OrderState.PARTIALLY_FILLED:
            pass  # already consistent

    # 4) Position: exchange is the truth.
    pos = client.get_position(symbol)
    if abs(pos.qty - local_position_qty) > _QTY_TOL:
        report.mismatches.append(
            Mismatch(
                kind="position",
                order_id=None,
                local=f"{local_position_qty:g}",
                exchange=f"{pos.qty:g}",
            )
        )
        report.actions.append(
            Action(
                kind="adopt_position",
                order_id=None,
                detail=f"local {local_position_qty:g} -> exchange {pos.qty:g} @ {pos.entry_price:g}",
            )
        )
    report.position = pos
    report.orders = dict(journal_state)
    report.converged = _verify_converged(report, client, symbol)
    return report


def _verify_converged(report: ReconcileReport, client: ExchangeClient, symbol: str) -> bool:
    """Post-condition: local open orders == exchange open orders, fills match."""
    exch = {o.order_id: o for o in client.get_open_orders(symbol)}
    local_open = {oid: o for oid, o in report.orders.items() if o.is_open}
    if set(exch) != set(local_open):
        return False
    for oid, eo in exch.items():
        lo = local_open[oid]
        if abs(eo.filled_qty - lo.filled_qty) > _QTY_TOL:
            return False
        if _STATUS_TO_STATE.get(eo.status) not in (lo.state, None) and lo.state is not OrderState.OPEN:
            return False
    return True
