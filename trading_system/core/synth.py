"""Seeded synthetic market data for tests, demos and report generation.

Everything is deterministic given a seed. Prices follow a lognormal random
walk; trades arrive as a Poisson-ish stream; the book generator emits a
snapshot plus a U/u/pu-consistent diff stream that replays into a valid book.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trading_system.core.schema import (
    BookSnapshot,
    DepthDiff,
    Liquidation,
    MarkPrice,
    OpenInterest,
    Side,
    Trade,
)
from trading_system.core.timeutils import NS_PER_MS, NS_PER_S

EXCHANGE = "binance_usdm"


def random_walk_prices(
    n: int, s0: float = 50_000.0, vol: float = 0.0004, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, vol, n)
    return s0 * np.exp(np.cumsum(log_ret))


def synth_trades(
    n: int = 10_000,
    symbol: str = "BTCUSDT",
    start_ts: int = 1_755_600_000 * NS_PER_S,
    s0: float = 50_000.0,
    vol: float = 0.0004,
    mean_gap_ms: float = 100.0,
    seed: int = 42,
) -> list[Trade]:
    rng = np.random.default_rng(seed)
    prices = random_walk_prices(n, s0=s0, vol=vol, seed=seed)
    gaps = rng.exponential(mean_gap_ms, n).astype(np.int64) * NS_PER_MS
    ts = start_ts + np.cumsum(gaps)
    qty = np.round(rng.lognormal(mean=-3.0, sigma=1.2, size=n), 4) + 0.0001
    # buy probability drifts with local momentum so CVD correlates with price
    mom = np.concatenate([[0.0], np.diff(np.log(prices))])
    p_buy = np.clip(0.5 + 400.0 * mom, 0.05, 0.95)
    sides = np.where(rng.random(n) < p_buy, Side.BUY, Side.SELL)
    out: list[Trade] = []
    for i in range(n):
        p = float(prices[i])
        q = float(qty[i])
        out.append(
            Trade(
                exchange=EXCHANGE,
                symbol=symbol,
                ts_event=int(ts[i]),
                ts_recv=int(ts[i]) + int(rng.integers(1, 30)) * NS_PER_MS,
                price=p,
                qty=q,
                qty_usd=p * q,
                side=Side(int(sides[i])),
                trade_id=i + 1,
            )
        )
    return out


@dataclass
class SynthBookStream:
    snapshot: BookSnapshot
    diffs: list[DepthDiff]


def synth_book_stream(
    n_diffs: int = 2_000,
    symbol: str = "BTCUSDT",
    start_ts: int = 1_755_600_000 * NS_PER_S,
    mid0: float = 50_000.0,
    n_levels: int = 50,
    tick: float = 0.1,
    seed: int = 42,
) -> SynthBookStream:
    """Snapshot + valid diff stream (contiguous U/u/pu, book never crosses)."""
    rng = np.random.default_rng(seed)
    mid = mid0
    bids = {round(mid - tick * (i + 1), 1): float(rng.lognormal(0, 1)) for i in range(n_levels)}
    asks = {round(mid + tick * (i + 1), 1): float(rng.lognormal(0, 1)) for i in range(n_levels)}
    last_id = 1_000
    snap = BookSnapshot(
        exchange=EXCHANGE,
        symbol=symbol,
        ts_event=start_ts,
        ts_recv=start_ts + 5 * NS_PER_MS,
        last_update_id=last_id,
        bids=tuple(sorted(bids.items(), key=lambda x: -x[0])),
        asks=tuple(sorted(asks.items())),
    )
    diffs: list[DepthDiff] = []
    prev_final = last_id
    ts = start_ts
    for _ in range(n_diffs):
        ts += int(rng.integers(80, 120)) * NS_PER_MS
        first = prev_final + 1
        final = first + int(rng.integers(0, 20))
        n_bid = int(rng.integers(0, 4))
        n_ask = int(rng.integers(0, 4))
        best_bid = max(bids) if bids else mid - tick
        best_ask = min(asks) if asks else mid + tick
        dbids: list[tuple[float, float]] = []
        dasks: list[tuple[float, float]] = []
        for _ in range(n_bid):
            p = round(best_bid - tick * int(rng.integers(0, n_levels)), 1)
            q = 0.0 if rng.random() < 0.3 else float(rng.lognormal(0, 1))
            if q == 0.0:
                bids.pop(p, None)
            elif p < best_ask:  # never cross
                bids[p] = q
            else:
                continue
            dbids.append((p, q))
        for _ in range(n_ask):
            p = round(best_ask + tick * int(rng.integers(0, n_levels)), 1)
            q = 0.0 if rng.random() < 0.3 else float(rng.lognormal(0, 1))
            if q == 0.0:
                asks.pop(p, None)
            elif p > best_bid:
                asks[p] = q
            else:
                continue
            dasks.append((p, q))
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
    return SynthBookStream(snapshot=snap, diffs=diffs)


def synth_liquidations(
    trades: list[Trade], rate: float = 0.002, seed: int = 42
) -> list[Liquidation]:
    rng = np.random.default_rng(seed)
    out: list[Liquidation] = []
    for t in trades:
        if rng.random() < rate:
            q = float(rng.lognormal(-1.0, 1.0))
            out.append(
                Liquidation(
                    exchange=t.exchange,
                    symbol=t.symbol,
                    ts_event=t.ts_event,
                    ts_recv=t.ts_recv,
                    price=t.price,
                    qty=q,
                    qty_usd=q * t.price,
                    side=Side(-int(t.side)),
                )
            )
    return out


def synth_open_interest(
    symbol: str = "BTCUSDT",
    start_ts: int = 1_755_600_000 * NS_PER_S,
    n: int = 500,
    step_s: int = 7,
    oi0: float = 80_000.0,
    price: float = 50_000.0,
    seed: int = 42,
) -> list[OpenInterest]:
    rng = np.random.default_rng(seed)
    oi = oi0 + np.cumsum(rng.normal(0, 30.0, n))
    return [
        OpenInterest(
            exchange=EXCHANGE,
            symbol=symbol,
            ts_event=start_ts + i * step_s * NS_PER_S,
            ts_recv=start_ts + i * step_s * NS_PER_S + 20 * NS_PER_MS,
            open_interest=float(oi[i]),
            open_interest_usd=float(oi[i]) * price,
        )
        for i in range(n)
    ]


def synth_mark_prices(
    trades: list[Trade], every_s: int = 1, funding: float = 1e-4
) -> list[MarkPrice]:
    out: list[MarkPrice] = []
    next_ts = trades[0].ts_event if trades else 0
    for t in trades:
        if t.ts_event >= next_ts:
            out.append(
                MarkPrice(
                    exchange=t.exchange,
                    symbol=t.symbol,
                    ts_event=t.ts_event,
                    ts_recv=t.ts_recv,
                    mark_price=t.price,
                    index_price=t.price,
                    funding_rate=funding,
                    next_funding_ts=t.ts_event + 8 * 3_600 * NS_PER_S,
                )
            )
            next_ts = t.ts_event + every_s * NS_PER_S
    return out
