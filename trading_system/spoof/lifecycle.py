"""L2 level lifecycle journal: diff book states, match changes against the tape.

Binance publishes L2 (aggregate qty per price level), not L3 (individual
orders), so intent is inferred from the *life* of price levels. Between two
consecutive book states each level change becomes an event:

- ``appeared``          level present now, absent before
- ``grew``              qty increased
- ``reduced_by_trade``  qty decreased and tape prints at that price within a
                        small time window explain the decrease (qty tolerance)
- ``canceled``          qty decreased/removed with no (sufficient) prints

Per-level episodes (birth -> death) accumulate max/filled/canceled qty. A
level is "large" when its qty exceeds ``large_k`` x rolling median of visible
level sizes. Iceberg refills (regrow/reappear at the same price shortly after
a fill) are chained so repeated instant refills are visible downstream.
"""

from __future__ import annotations

import enum
import statistics
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import polars as pl
import structlog

from trading_system.core.config import load_config
from trading_system.core.schema import PriceLevel, Side, Trade
from trading_system.core.timeutils import NS_PER_MS

log = structlog.get_logger(__name__)

_PRICE_DECIMALS = 8  # float prices quantized to a dict key


def spoof_config() -> dict[str, Any]:
    """The ``spoof`` section of the base config ({} if the file is absent)."""
    try:
        return dict(load_config().get("spoof") or {})
    except FileNotFoundError:  # pragma: no cover - config ships with the repo
        return {}


def price_key(price: float) -> float:
    return round(float(price), _PRICE_DECIMALS)


class LevelEventType(enum.StrEnum):
    APPEARED = "appeared"
    GREW = "grew"
    REDUCED_BY_TRADE = "reduced_by_trade"
    CANCELED = "canceled"


@dataclass(frozen=True, slots=True)
class BookState:
    """One observed L2 book: UTC-ns timestamp plus (price, qty) per side."""

    ts: int
    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]


@dataclass(frozen=True, slots=True)
class LevelEvent:
    ts: int
    side: str  # "bid" | "ask"
    price: float
    kind: LevelEventType
    qty_before: float
    qty_after: float
    filled_qty: float  # part of a decrease explained by tape prints
    canceled_qty: float  # unexplained part of a decrease
    episode_id: int


@dataclass(slots=True)
class LevelEpisode:
    """One life of a price level: from appearance to disappearance."""

    episode_id: int
    side: str
    price: float
    birth_ts: int
    last_ts: int
    death_ts: int | None = None
    max_qty: float = 0.0
    filled_qty: float = 0.0
    canceled_qty: float = 0.0
    grown_qty: float = 0.0
    n_grew: int = 0
    n_reductions: int = 0
    iceberg_refills: int = 0  # running refill count of this episode's chain
    refill_chain_id: int = -1  # links episodes stitched by instant refills
    was_large: bool = False

    @property
    def alive(self) -> bool:
        return self.death_ts is None

    @property
    def lifetime_ns(self) -> int:
        return (self.death_ts if self.death_ts is not None else self.last_ts) - self.birth_ts

    @property
    def fill_frac(self) -> float:
        """Filled share of removed qty; 0.5 (neutral) when nothing was removed."""
        total = self.filled_qty + self.canceled_qty
        return self.filled_qty / total if total > 0 else 0.5


class _TapeMatcher:
    """Consumable index of tape prints keyed by (side, price).

    A bid level is consumed by taker SELL prints, an ask level by taker BUY
    prints (side matching can be disabled). Each print's qty is consumed at
    most once across all decrease events. ``consume`` windows must be queried
    with non-decreasing ``t_lo`` (states are processed in time order).
    """

    def __init__(self, trades: Iterable[Trade], *, match_taker_side: bool = True) -> None:
        self._side_sensitive = match_taker_side
        self._by_key: dict[tuple[str, float], list[list[float]]] = {}
        self._start: dict[tuple[str, float], int] = {}
        for t in sorted(trades, key=lambda x: x.ts_event):
            side = ("bid" if t.side is Side.SELL else "ask") if match_taker_side else ""
            self._by_key.setdefault((side, price_key(t.price)), []).append(
                [float(t.ts_event), float(t.qty)]
            )

    def consume(self, side: str, price: float, t_lo: int, t_hi: int, amount: float) -> float:
        key = (side if self._side_sensitive else "", price_key(price))
        entries = self._by_key.get(key)
        if not entries or amount <= 0:
            return 0.0
        i = self._start.get(key, 0)
        while i < len(entries) and entries[i][0] < t_lo:
            i += 1
        self._start[key] = i
        got = 0.0
        j = i
        while j < len(entries) and entries[j][0] <= t_hi and got < amount - 1e-12:
            take = min(entries[j][1], amount - got)
            entries[j][1] -= take
            got += take
            j += 1
        return got


_EVENT_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Int64,
    "side": pl.Utf8,
    "price": pl.Float64,
    "kind": pl.Utf8,
    "qty_before": pl.Float64,
    "qty_after": pl.Float64,
    "filled_qty": pl.Float64,
    "canceled_qty": pl.Float64,
    "episode_id": pl.Int64,
}

_GRID_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Int64,
    "side": pl.Utf8,
    "price": pl.Float64,
    "qty": pl.Float64,
    "episode_id": pl.Int64,
    "is_large": pl.Boolean,
}


@dataclass
class LevelJournal:
    """Builds level events and episodes from book states plus the trade tape.

    Parameters default to the ``spoof`` config section: ``large_k`` from
    ``spoof.large_level_atr_notional`` and ``iceberg_refill_ms`` from
    ``spoof.iceberg_refill_ms``; explicit arguments override the config.
    """

    large_k: float | None = None
    median_window: int = 50  # states over which level-size medians roll
    trade_window_ns: int = 150 * NS_PER_MS  # pad around (prev_ts, ts] for prints
    qty_rel_tol: float = 0.15  # decrease counts as filled if prints cover >= 1-tol
    iceberg_refill_ms: float | None = None
    match_taker_side: bool = True
    record_grid: bool = True  # keep (ts, side, price, qty) rows for every state
    qty_eps: float = 1e-9

    events: list[LevelEvent] = field(default_factory=list, init=False)
    episodes: list[LevelEpisode] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        cfg = spoof_config() if (self.large_k is None or self.iceberg_refill_ms is None) else {}
        if self.large_k is None:
            self.large_k = float(cfg.get("large_level_atr_notional", 3.0))
        if self.iceberg_refill_ms is None:
            self.iceberg_refill_ms = float(cfg.get("iceberg_refill_ms", 300))
        self._refill_ns = int(self.iceberg_refill_ms * NS_PER_MS)
        self._active: dict[tuple[str, float], LevelEpisode] = {}
        self._last_ep: dict[tuple[str, float], LevelEpisode] = {}
        self._last_fill_ts: dict[tuple[str, float], int] = {}
        self._chain: dict[tuple[str, float], tuple[int, int]] = {}  # key -> (chain_id, refills)
        self._medians: deque[float] = deque(maxlen=self.median_window)
        self._prev: tuple[dict[float, float], dict[float, float]] | None = None
        self._prev_ts: int = 0
        self._next_episode_id = 0
        self._next_chain_id = 0
        self._matcher: _TapeMatcher | None = None
        self._grid: dict[str, list] = {k: [] for k in _GRID_SCHEMA}

    # -- public API ---------------------------------------------------------

    def run(self, states: Sequence[BookState], trades: Sequence[Trade]) -> LevelJournal:
        """Process a full session of book states against the trade tape.

        ``states`` must be ordered by non-decreasing ``ts`` (equal timestamps
        are allowed); out-of-order states would silently corrupt episode
        lifetimes and tape matching, so they raise ``ValueError`` instead.
        ``trades`` may arrive in any order (the matcher sorts them).
        """
        for prev_st, st in zip(states, states[1:], strict=False):
            if st.ts < prev_st.ts:
                raise ValueError(
                    f"book states must be ordered by non-decreasing ts: "
                    f"{st.ts} follows {prev_st.ts}"
                )
        self._matcher = _TapeMatcher(trades, match_taker_side=self.match_taker_side)
        for st in states:
            self._step(st)
        log.debug(
            "level_journal_done",
            states=len(states),
            events=len(self.events),
            episodes=len(self.episodes),
        )
        return self

    @property
    def large_threshold(self) -> float:
        """Current 'large level' qty threshold (large_k x rolling median)."""
        if not self._medians:
            return float("inf")
        return float(self.large_k) * statistics.median(self._medians)

    def events_frame(self) -> pl.DataFrame:
        rows = [
            (
                e.ts,
                e.side,
                e.price,
                str(e.kind),
                e.qty_before,
                e.qty_after,
                e.filled_qty,
                e.canceled_qty,
                e.episode_id,
            )
            for e in self.events
        ]
        return pl.DataFrame(rows, schema=_EVENT_SCHEMA, orient="row")

    def grid_frame(self) -> pl.DataFrame:
        """Per-state level occupancy: one row per (state ts, visible level)."""
        return pl.DataFrame(self._grid, schema=_GRID_SCHEMA)

    # -- internals ----------------------------------------------------------

    def _step(self, st: BookState) -> None:
        cur_b = {price_key(p): q for p, q in st.bids if q > self.qty_eps}
        cur_a = {price_key(p): q for p, q in st.asks if q > self.qty_eps}
        qtys = list(cur_b.values()) + list(cur_a.values())
        if qtys:
            self._medians.append(statistics.median(qtys))
        thresh = self.large_threshold
        prev_b, prev_a = self._prev if self._prev is not None else ({}, {})
        prev_ts = self._prev_ts if self._prev is not None else st.ts
        t_lo = prev_ts - self.trade_window_ns
        t_hi = st.ts + self.trade_window_ns
        for side, cur, prev in (("bid", cur_b, prev_b), ("ask", cur_a, prev_a)):
            self._step_side(side, cur, prev, st.ts, t_lo, t_hi, thresh)
        self._prev = (cur_b, cur_a)
        self._prev_ts = st.ts

    def _step_side(
        self,
        side: str,
        cur: dict[float, float],
        prev: dict[float, float],
        ts: int,
        t_lo: int,
        t_hi: int,
        thresh: float,
    ) -> None:
        for key, q in cur.items():
            pq = prev.get(key)
            if pq is None:
                ep = self._new_episode(side, key, ts, q)
                self._note_refill(side, key, ep, ts)
                self._emit(ts, side, key, LevelEventType.APPEARED, 0.0, q, 0.0, 0.0, ep)
            else:
                ep = self._active[(side, key)]
                if q > pq + self.qty_eps:
                    ep.n_grew += 1
                    ep.grown_qty += q - pq
                    self._note_refill(side, key, ep, ts)
                    self._emit(ts, side, key, LevelEventType.GREW, pq, q, 0.0, 0.0, ep)
                elif q < pq - self.qty_eps:
                    self._reduce(side, key, ep, ts, t_lo, t_hi, pq, q, death=False)
            ep = self._active[(side, key)]
            ep.last_ts = ts
            ep.max_qty = max(ep.max_qty, q)
            is_large = q > thresh
            ep.was_large = ep.was_large or is_large
            if self.record_grid:
                g = self._grid
                g["ts"].append(ts)
                g["side"].append(side)
                g["price"].append(key)
                g["qty"].append(q)
                g["episode_id"].append(ep.episode_id)
                g["is_large"].append(is_large)
        for key, pq in prev.items():
            if key not in cur:
                ep = self._active[(side, key)]
                self._reduce(side, key, ep, ts, t_lo, t_hi, pq, 0.0, death=True)

    def _new_episode(self, side: str, key: float, ts: int, q: float) -> LevelEpisode:
        ep = LevelEpisode(
            episode_id=self._next_episode_id,
            side=side,
            price=key,
            birth_ts=ts,
            last_ts=ts,
            max_qty=q,
        )
        self._next_episode_id += 1
        self.episodes.append(ep)
        self._active[(side, key)] = ep
        return ep

    def _note_refill(self, side: str, key: float, ep: LevelEpisode, ts: int) -> None:
        """Count a grow/appear as an iceberg refill if it closely follows a fill."""
        lf = self._last_fill_ts.get((side, key))
        if lf is not None and ts - lf <= self._refill_ns:
            cid, refills = self._chain.get((side, key), (-1, 0))
            if cid < 0:
                cid = self._next_chain_id
                self._next_chain_id += 1
            refills += 1
            self._chain[(side, key)] = (cid, refills)
            ep.refill_chain_id = cid
            ep.iceberg_refills = max(ep.iceberg_refills, refills)
            prev_ep = self._last_ep.get((side, key))
            if (
                prev_ep is not None
                and prev_ep.refill_chain_id < 0
                and prev_ep.death_ts is not None
                and ts - prev_ep.death_ts <= self._refill_ns
                and prev_ep.filled_qty > self.qty_eps
            ):
                prev_ep.refill_chain_id = cid  # the episode whose fill started the chain
        elif ep.n_grew == 0 and ep.iceberg_refills == 0 and ep.refill_chain_id < 0:
            self._chain.pop((side, key), None)  # cold (re)birth ends any old chain

    def _reduce(
        self,
        side: str,
        key: float,
        ep: LevelEpisode,
        ts: int,
        t_lo: int,
        t_hi: int,
        q_before: float,
        q_after: float,
        *,
        death: bool,
    ) -> None:
        dec = q_before - q_after
        assert self._matcher is not None
        filled = self._matcher.consume(side, key, t_lo, t_hi, dec)
        canceled = max(0.0, dec - filled)
        kind = (
            LevelEventType.REDUCED_BY_TRADE
            if filled >= dec * (1.0 - self.qty_rel_tol)
            else LevelEventType.CANCELED
        )
        ep.filled_qty += filled
        ep.canceled_qty += canceled
        ep.n_reductions += 1
        if filled > self.qty_eps:
            self._last_fill_ts[(side, key)] = ts
        if death:
            ep.death_ts = ts
            ep.last_ts = ts
            self._last_ep[(side, key)] = ep
            del self._active[(side, key)]
        self._emit(ts, side, key, kind, q_before, q_after, filled, canceled, ep)

    def _emit(
        self,
        ts: int,
        side: str,
        price: float,
        kind: LevelEventType,
        qb: float,
        qa: float,
        filled: float,
        canceled: float,
        ep: LevelEpisode,
    ) -> None:
        self.events.append(
            LevelEvent(
                ts=ts,
                side=side,
                price=price,
                kind=kind,
                qty_before=qb,
                qty_after=qa,
                filled_qty=filled,
                canceled_qty=canceled,
                episode_id=ep.episode_id,
            )
        )
