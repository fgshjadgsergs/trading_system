"""Local M2 demo generator: mean-reverting synthetic depth stream.

core.synth.synth_book_stream only ever adds liquidity at or below the current
best, so its book erodes monotonically and the spread grows without bound —
fine for sequencing tests, unrepresentative for depth visuals. This generator
keeps a band of levels around a mean-reverting mid on an integer tick grid,
emitting removals for crossed levels in the same diff, so the stream is a
valid U/u/pu chain whose book tracks the mid like a real market.
"""

from __future__ import annotations

import numpy as np

from trading_system.core.schema import BookSnapshot, DepthDiff
from trading_system.core.synth import EXCHANGE
from trading_system.core.timeutils import NS_PER_MS, NS_PER_S


def mean_reverting_book_stream(
    n_diffs: int = 36_000,
    symbol: str = "BTCUSDT",
    start_ts: int = 1_755_600_000 * NS_PER_S,
    mid0: float = 50_000.0,
    n_levels: int = 50,
    tick: float = 0.5,
    reversion: float = 0.005,
    step_sigma: float = 2.0,
    seed: int = 42,
) -> tuple[BookSnapshot, list[DepthDiff]]:
    """(snapshot, diffs): a valid depth stream whose book hugs a wandering mid."""
    rng = np.random.default_rng(seed)
    center = int(round(mid0 / tick))  # mid in ticks
    mid_k = center

    def level_qty(dist: int) -> float:
        base = float(rng.lognormal(0.3, 0.9))
        wall = 20.0 if rng.random() < 0.04 else 1.0  # occasional large wall
        return round(base * wall * (1.0 + dist / 10.0), 4)

    bids = {mid_k - 1 - i: level_qty(i) for i in range(n_levels)}
    asks = {mid_k + 1 + i: level_qty(i) for i in range(n_levels)}
    last_id = 1_000
    snapshot = BookSnapshot(
        exchange=EXCHANGE,
        symbol=symbol,
        ts_event=start_ts,
        ts_recv=start_ts + 5 * NS_PER_MS,
        last_update_id=last_id,
        bids=tuple((k * tick, q) for k, q in sorted(bids.items(), reverse=True)),
        asks=tuple((k * tick, q) for k, q in sorted(asks.items())),
    )
    diffs: list[DepthDiff] = []
    prev_final = last_id
    ts = start_ts
    for _ in range(n_diffs):
        ts += int(rng.integers(80, 120)) * NS_PER_MS
        drift = -reversion * (mid_k - center)
        mid_k += int(round(rng.normal(drift, step_sigma)))
        dbids: list[tuple[float, float]] = []
        dasks: list[tuple[float, float]] = []
        # clear levels crossed by the mid move (same-diff removals keep it valid)
        for k in [k for k in bids if k >= mid_k]:
            del bids[k]
            dbids.append((k * tick, 0.0))
        for k in [k for k in asks if k <= mid_k]:
            del asks[k]
            dasks.append((k * tick, 0.0))
        # churn: refill near best, random add/update/remove inside the band
        for book_side, sign in ((bids, -1), (asks, +1)):
            for _ in range(int(rng.integers(1, 4))):
                dist = 0 if rng.random() < 0.5 else int(rng.integers(0, n_levels))
                k = mid_k + sign * (1 + dist)
                if rng.random() < 0.25 and k in book_side:
                    del book_side[k]
                    q = 0.0
                else:
                    q = level_qty(dist)
                    book_side[k] = q
                (dbids if sign < 0 else dasks).append((k * tick, q))
        first = prev_final + 1
        final = first + int(rng.integers(0, 20))
        diffs.append(
            DepthDiff(
                exchange=EXCHANGE,
                symbol=symbol,
                ts_event=ts,
                ts_recv=ts + int(rng.integers(1, 15)) * NS_PER_MS,
                first_update_id=first,
                final_update_id=final,
                prev_final_update_id=prev_final,
                bids=tuple(dbids),
                asks=tuple(dasks),
            )
        )
        prev_final = final
    return snapshot, diffs
