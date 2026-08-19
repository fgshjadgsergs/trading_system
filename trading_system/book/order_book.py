"""L2 order book with strict Binance USDT-M futures U/u/pu sequencing.

The book is rebuilt from a REST snapshot plus the incremental depth stream.
Quantities are absolute: a diff level (price, qty) replaces the level, qty == 0
removes it. Sequencing is enforced strictly — any gap, reorder or invariant
violation raises NeedsResync and poisons the book until the next snapshot; the
book never continues silently on corrupt state.
"""

from __future__ import annotations

import math

import structlog

from trading_system.core.schema import BookSnapshot, DepthDiff, PriceLevel

log = structlog.get_logger(__name__)


class NeedsResync(Exception):
    """Book state can no longer be trusted; the caller must re-snapshot."""


class BookInvariantError(NeedsResync):
    """A book invariant failed (crossed book, non-positive qty, bad price)."""


class OrderBook:
    """In-memory L2 book for one (exchange, symbol).

    Lifecycle: ``apply_snapshot`` -> ``apply_diff`` per stream event. The first
    diff after a snapshot must either straddle the snapshot id
    (``U <= last_update_id <= u``, the Binance futures rule) or continue it
    exactly (``pu == last_update_id``); diffs entirely before the snapshot
    (``u < last_update_id``) are dropped. Every later diff must satisfy
    ``pu == last final_update_id``. Violations raise :class:`NeedsResync`.
    """

    __slots__ = (
        "exchange",
        "symbol",
        "last_update_id",
        "ts_event",
        "_bids",
        "_asks",
        "_synced",
        "_awaiting_first",
    )

    def __init__(self) -> None:
        self.exchange: str = ""
        self.symbol: str = ""
        self.last_update_id: int = -1
        self.ts_event: int = 0
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._synced: bool = False
        self._awaiting_first: bool = False

    # -- state ----------------------------------------------------------------

    @property
    def synced(self) -> bool:
        """True when a snapshot is applied and no violation has occurred."""
        return self._synced

    def _poison(self, reason: str) -> NeedsResync:
        self._synced = False
        log.warning("book_needs_resync", symbol=self.symbol, reason=reason)
        return NeedsResync(reason)

    @staticmethod
    def _check_level(price: float, qty: float, where: str) -> None:
        if not (math.isfinite(price) and price > 0.0):
            raise BookInvariantError(f"{where}: bad price {price!r}")
        if not math.isfinite(qty) or qty < 0.0:
            raise BookInvariantError(f"{where}: bad qty {qty!r} at price {price!r}")

    def apply_snapshot(self, snap: BookSnapshot) -> None:
        """Reset the book from a REST snapshot; clears any poisoned state."""
        bids: dict[float, float] = {}
        asks: dict[float, float] = {}
        for price, qty in snap.bids:
            self._check_level(price, qty, "snapshot bid")
            if qty == 0.0:
                raise BookInvariantError(f"snapshot bid with zero qty at {price!r}")
            bids[price] = qty
        for price, qty in snap.asks:
            self._check_level(price, qty, "snapshot ask")
            if qty == 0.0:
                raise BookInvariantError(f"snapshot ask with zero qty at {price!r}")
            asks[price] = qty
        if bids and asks and max(bids) >= min(asks):
            raise BookInvariantError(
                f"crossed snapshot: best bid {max(bids)} >= best ask {min(asks)}"
            )
        self.exchange = snap.exchange
        self.symbol = snap.symbol
        self._bids = bids
        self._asks = asks
        self.last_update_id = snap.last_update_id
        self.ts_event = snap.ts_event
        self._synced = True
        self._awaiting_first = True

    def apply_diff(self, diff: DepthDiff) -> bool:
        """Apply one depth diff. Returns True if applied, False if dropped as stale.

        Raises NeedsResync on any sequencing gap and BookInvariantError (a
        NeedsResync subclass) on a crossed book or invalid level; either way the
        book is poisoned until the next apply_snapshot.
        """
        if not self._synced:
            raise NeedsResync("book is not synced: apply a snapshot first")
        if self._awaiting_first:
            if diff.final_update_id < self.last_update_id:
                return False  # entirely before the snapshot: drop, keep waiting
            straddles = diff.first_update_id <= self.last_update_id <= diff.final_update_id
            continues = diff.prev_final_update_id == self.last_update_id
            if not (straddles or continues):
                raise self._poison(
                    "gap after snapshot: first diff "
                    f"U={diff.first_update_id} pu={diff.prev_final_update_id} "
                    f"does not reach snapshot last_update_id={self.last_update_id}"
                )
            self._awaiting_first = False
        elif diff.prev_final_update_id != self.last_update_id:
            raise self._poison(
                f"sequence break: pu={diff.prev_final_update_id} != "
                f"last final_update_id={self.last_update_id}"
            )
        try:
            for price, qty in diff.bids:
                self._check_level(price, qty, "diff bid")
                if qty == 0.0:
                    self._bids.pop(price, None)
                else:
                    self._bids[price] = qty
            for price, qty in diff.asks:
                self._check_level(price, qty, "diff ask")
                if qty == 0.0:
                    self._asks.pop(price, None)
                else:
                    self._asks[price] = qty
            if self._bids and self._asks and max(self._bids) >= min(self._asks):
                raise BookInvariantError(
                    f"crossed book after u={diff.final_update_id}: "
                    f"best bid {max(self._bids)} >= best ask {min(self._asks)}"
                )
        except BookInvariantError as e:
            self._poison(str(e))
            raise
        self.last_update_id = diff.final_update_id
        self.ts_event = diff.ts_event
        return True

    # -- accessors ------------------------------------------------------------

    @property
    def best_bid(self) -> float | None:
        return max(self._bids) if self._bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self._asks) if self._asks else None

    @property
    def mid(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        return (bb + ba) / 2.0 if bb is not None and ba is not None else None

    @property
    def spread(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        return ba - bb if bb is not None and ba is not None else None

    def top_n(self, n: int) -> tuple[tuple[PriceLevel, ...], tuple[PriceLevel, ...]]:
        """(bids best-first, asks best-first): up to n (price, qty) per side."""
        bids = tuple(sorted(self._bids.items(), key=lambda x: -x[0])[:n])
        asks = tuple(sorted(self._asks.items())[:n])
        return bids, asks

    def depth_within(self, pct: float) -> tuple[float, float]:
        """(bid_qty, ask_qty) summed within +/-pct of mid; pct=0.005 is 0.5%."""
        m = self.mid
        if m is None:
            return 0.0, 0.0
        lo, hi = m * (1.0 - pct), m * (1.0 + pct)
        bid_qty = sum(q for p, q in self._bids.items() if p >= lo)
        ask_qty = sum(q for p, q in self._asks.items() if p <= hi)
        return bid_qty, ask_qty

    def n_levels(self) -> tuple[int, int]:
        return len(self._bids), len(self._asks)
