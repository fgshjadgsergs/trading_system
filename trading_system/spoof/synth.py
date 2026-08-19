"""Seeded synthetic L2 sessions with planted spoof/honest/iceberg episodes.

``labeled_day`` builds a deterministic sequence of book states plus a matching
trade tape with ~25 planted patterns and returns the ground-truth labels, so
detector precision/recall can be asserted offline. Patterns:

- ``honest``   large wall that sits for tens of seconds, then is eaten by
               prints bite by bite and dies fully filled
- ``spoof``    large level at the same price that appears and is pulled with
               no prints, >= flicker_k times inside the flicker window
- ``iceberg``  visible level that is consumed by prints and instantly refilled
               at the same price, repeatedly, then dies by a final fill

Background levels sit on odd tick offsets with small qty jitter; plants use
even offsets, so planted episodes never merge with background episodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from trading_system.core.schema import Side, Trade
from trading_system.core.synth import EXCHANGE
from trading_system.core.timeutils import NS_PER_MS, NS_PER_S
from trading_system.spoof.lifecycle import BookState

T0 = 1_755_600_000 * NS_PER_S

TRUTH_SCHEMA: dict[str, pl.DataType] = {
    "pattern_id": pl.Int64,
    "label": pl.Utf8,
    "side": pl.Utf8,
    "price": pl.Float64,
    "ts_start": pl.Int64,
    "ts_end": pl.Int64,
}


@dataclass(frozen=True, slots=True)
class _Plant:
    pattern_id: int
    label: str
    side: str
    price: float
    qty_at: dict[int, float]  # state index -> visible qty
    trades: list[tuple[int, float]]  # (ts, qty) taker prints at self.price
    ts_start: int
    ts_end: int


def _taker(side: str) -> Side:
    return Side.SELL if side == "bid" else Side.BUY


def _plant_honest(
    pid: int, side: str, price: float, i0: int, ts_of, half: int, rng: np.random.Generator
) -> _Plant:
    q = float(rng.uniform(5.0, 9.0))
    hold = int(rng.integers(200, 451))  # 40..90 s at 200 ms cadence
    n_bites = int(rng.integers(4, 9))
    qty_at: dict[int, float] = {i0 + j: q for j in range(hold)}
    trades: list[tuple[int, float]] = []
    bite = q / n_bites
    for b in range(n_bites):
        i = i0 + hold + b
        left = q - bite * (b + 1)
        if left > 1e-9:
            qty_at[i] = left
        trades.append((ts_of(i) - half, bite))
    return _Plant(pid, "honest", side, price, qty_at, trades, ts_of(i0), ts_of(i0 + hold + n_bites))


def _plant_spoof(
    pid: int, side: str, price: float, i0: int, ts_of, rng: np.random.Generator, k: int
) -> _Plant:
    q = float(rng.uniform(5.0, 9.0))
    m = k + int(rng.integers(0, 3))
    qty_at: dict[int, float] = {}
    cur = i0
    for _ in range(m):
        live = int(rng.integers(3, 13))  # 0.6..2.4 s visible
        for j in range(live):
            qty_at[cur + j] = q
        cur += live + int(rng.integers(5, 31))  # 1..6 s gap, no prints
    return _Plant(pid, "spoof", side, price, qty_at, [], ts_of(i0), ts_of(cur))


def _plant_iceberg(
    pid: int, side: str, price: float, i0: int, ts_of, half: int, rng: np.random.Generator
) -> _Plant:
    q = float(rng.uniform(5.0, 8.0))
    n_cycles = int(rng.integers(4, 8))
    warmup = int(rng.integers(10, 31))  # 2..6 s of quiet standing first
    qty_at: dict[int, float] = {i0 + j: q for j in range(warmup)}
    trades: list[tuple[int, float]] = []
    i = i0 + warmup
    for _ in range(n_cycles):
        qty_at[i] = q * 0.25  # print eats 75%, refill lands on the next state
        trades.append((ts_of(i) - half, q * 0.75))
        qty_at[i + 1] = q
        qty_at[i + 2] = q
        i += 3
    trades.append((ts_of(i) - half, q))  # final full consumption -> dies by fill
    return _Plant(pid, "iceberg", side, price, qty_at, trades, ts_of(i0), ts_of(i))


def labeled_day(
    seed: int = 42,
    n_patterns: int = 25,
    duration_s: float = 300.0,
    cadence_ms: int = 200,
    mid: float = 50_000.0,
    tick: float = 1.0,
    flicker_k: int = 3,
    symbol: str = "BTCUSDT",
) -> tuple[list[BookState], list[Trade], pl.DataFrame]:
    """Deterministic book-state sequence + tape + ground-truth pattern frame."""
    rng = np.random.default_rng(seed)
    cadence_ns = cadence_ms * NS_PER_MS
    half = cadence_ns // 2
    n_states = int(duration_s * 1000 / cadence_ms) + 1

    def ts_of(i: int) -> int:
        return T0 + i * cadence_ns

    bg_offsets = np.arange(1, 60, 2)  # odd ticks: background only
    bg = {
        side: {
            float(mid + s * off * tick): float(rng.lognormal(0.0, 0.3))
            for off in bg_offsets
        }
        for side, s in (("bid", -1), ("ask", 1))
    }

    labels = [("honest", "spoof", "iceberg")[i % 3] for i in range(n_patterns)]
    rng.shuffle(labels)
    free = {
        "bid": list(rng.permutation(np.arange(4, 61, 2))),
        "ask": list(rng.permutation(np.arange(4, 61, 2))),
    }
    max_len = 470  # longest pattern in states, with margin
    plants: list[_Plant] = []
    for pid, label in enumerate(labels):
        side = "bid" if pid % 2 == 0 else "ask"
        off = int(free[side].pop())
        price = float(mid + (-1 if side == "bid" else 1) * off * tick)
        i0 = 25 + int(pid * max(1, (n_states - max_len - 50) // max(1, n_patterns)))
        i0 += int(rng.integers(0, 10))
        if label == "honest":
            plants.append(_plant_honest(pid, side, price, i0, ts_of, half, rng))
        elif label == "spoof":
            plants.append(_plant_spoof(pid, side, price, i0, ts_of, rng, flicker_k))
        else:
            plants.append(_plant_iceberg(pid, side, price, i0, ts_of, half, rng))

    trades: list[Trade] = []
    tid = 0
    for p in plants:
        taker = _taker(p.side)
        for t, q in p.trades:
            tid += 1
            trades.append(
                Trade(
                    exchange=EXCHANGE,
                    symbol=symbol,
                    ts_event=t,
                    ts_recv=t + NS_PER_MS,
                    price=p.price,
                    qty=q,
                    qty_usd=q * p.price,
                    side=taker,
                    trade_id=tid,
                )
            )
    # background noise prints at mid (no resting level there -> never matched)
    for i in range(1, n_states):
        if rng.random() < 0.3:
            t = ts_of(i) - int(rng.integers(10, 190)) * NS_PER_MS
            q = float(rng.lognormal(-1.5, 0.5))
            tid += 1
            trades.append(
                Trade(
                    exchange=EXCHANGE,
                    symbol=symbol,
                    ts_event=t,
                    ts_recv=t + NS_PER_MS,
                    price=mid,
                    qty=q,
                    qty_usd=q * mid,
                    side=Side(int(rng.choice([1, -1]))),
                    trade_id=tid,
                )
            )
    trades.sort(key=lambda t: t.ts_event)

    states: list[BookState] = []
    bg_prices = {side: list(levels.keys()) for side, levels in bg.items()}
    for i in range(n_states):
        for _ in range(3):  # small qty jitter on a few background levels
            side = "bid" if rng.random() < 0.5 else "ask"
            p = bg_prices[side][int(rng.integers(0, len(bg_prices[side])))]
            bg[side][p] = max(0.05, bg[side][p] * float(rng.uniform(0.9, 1.1)))
        levels: dict[str, list[tuple[float, float]]] = {
            "bid": list(bg["bid"].items()),
            "ask": list(bg["ask"].items()),
        }
        for p in plants:
            q = p.qty_at.get(i)
            if q is not None:
                levels[p.side].append((p.price, q))
        states.append(
            BookState(
                ts=ts_of(i),
                bids=tuple(sorted(levels["bid"], key=lambda x: -x[0])),
                asks=tuple(sorted(levels["ask"])),
            )
        )

    truth = pl.DataFrame(
        [(p.pattern_id, p.label, p.side, p.price, p.ts_start, p.ts_end) for p in plants],
        schema=TRUTH_SCHEMA,
        orient="row",
    )
    return states, trades, truth


def evaluate_spoof_flags(
    truth: pl.DataFrame,
    annotated: pl.DataFrame,
    *,
    flag: str = "flicker",
    positive_label: str = "spoof",
    price_tol: float = 0.5,
    pad_ns: int = 2 * NS_PER_S,
) -> dict[str, float]:
    """Pattern-level precision/recall of a boolean episode flag vs ground truth.

    A pattern is predicted positive when any large episode at its (side,
    price) whose birth falls inside the pattern's padded time span carries
    ``flag``. Returns precision, recall and the tp/fp/fn/tn counts.
    """
    tp = fp = fn = tn = 0
    for pat in truth.iter_rows(named=True):
        eps = annotated.filter(
            (pl.col("side") == pat["side"])
            & ((pl.col("price") - pat["price"]).abs() <= price_tol)
            & (pl.col("birth_ts") >= pat["ts_start"] - pad_ns)
            & (pl.col("birth_ts") <= pat["ts_end"] + pad_ns)
            & pl.col("was_large")
        )
        pred = bool(eps[flag].any()) if not eps.is_empty() else False
        actual = pat["label"] == positive_label
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }
