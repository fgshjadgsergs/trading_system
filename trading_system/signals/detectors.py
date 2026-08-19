"""M7: rule-based signal detectors — pure functions, zero internal state.

Inputs are plain frames; outputs are event frames. Determinism and the
"fires exactly once" property come from edge-triggered conditions computed
from the data itself, never from hidden state.

Event frame columns: ts (bar close), signal (s1|s2), side (+1 long / -1 short),
price (bar close at fire), target (price objective), meta (level/pool price).
"""

from __future__ import annotations

import polars as pl

EVENT_SCHEMA = {
    "ts": pl.Int64,
    "signal": pl.Utf8,
    "side": pl.Int8,
    "price": pl.Float64,
    "target": pl.Float64,
    "meta": pl.Float64,
}


def _events(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=EVENT_SCHEMA).sort("ts")


def s1_magnet(
    bars: pl.DataFrame,
    pools: pl.DataFrame,
    k_atr: float = 3.0,
    min_heat_share: float = 0.25,
) -> pl.DataFrame:
    """S1 "magnet": price gravitates to the largest untouched liquidation pool
    within k*ATR.

    pools: (price, heat_usd, touched_ts) — touched_ts is null while intact; a
    pool consumed at touched_ts stops being a target from that moment on.
    Edge-triggered per pool: fires on the first bar where the pool becomes the
    in-range top pool, once per pool.
    """
    if "atr" not in bars.columns:
        raise ValueError("bars must carry an 'atr' column")
    total_heat = float(pools["heat_usd"].sum()) if pools.height else 0.0
    rows: list[dict] = []
    fired: set[float] = set()
    for bar in bars.iter_rows(named=True):
        atr = bar["atr"]
        if atr is None or total_heat <= 0:
            continue
        close, ts = bar["close"], bar["ts_close"]
        best: dict | None = None
        for p in pools.iter_rows(named=True):
            if p["touched_ts"] is not None and p["touched_ts"] <= ts:
                continue
            if abs(p["price"] - close) <= k_atr * atr and (
                best is None or p["heat_usd"] > best["heat_usd"]
            ):
                best = p
        if best is None or best["heat_usd"] < min_heat_share * total_heat:
            continue
        if best["price"] in fired:
            continue
        fired.add(best["price"])
        rows.append(
            {
                "ts": ts,
                "signal": "s1",
                "side": 1 if best["price"] > close else -1,
                "price": close,
                "target": best["price"],
                "meta": best["heat_usd"],
            }
        )
    return _events(rows)


def s2_sweep_reversal(
    bars: pl.DataFrame,
    levels: pl.DataFrame,
    return_bars: int = 3,
) -> pl.DataFrame:
    """S2 "sweep-reversal": pierce a clustered level, return, structure shift.

    For an equal-highs level L: bar i is the FIRST pierce (high > L, previous
    bar high <= L); the signal fires at the first bar j in (i, i+return_bars]
    that closes back below L AND below the sweep bar's low (structure shift).
    One event per pierce episode; mirrored for equal-lows.
    """
    highs = bars["high"].to_list()
    lows = bars["low"].to_list()
    closes = bars["close"].to_list()
    ts_close = bars["ts_close"].to_list()
    rows: list[dict] = []
    for lvl in levels.iter_rows(named=True):
        price, kind = lvl["price"], lvl["kind"]
        i = 1
        while i < len(highs):
            if kind == "high":
                pierced = highs[i] > price and highs[i - 1] <= price
            else:
                pierced = lows[i] < price and lows[i - 1] >= price
            if not pierced:
                i += 1
                continue
            fired_at = None
            for j in range(i + 1, min(i + return_bars, len(highs) - 1) + 1):
                if kind == "high" and closes[j] < price and closes[j] < lows[i]:
                    fired_at = j
                    break
                if kind == "low" and closes[j] > price and closes[j] > highs[i]:
                    fired_at = j
                    break
            if fired_at is not None:
                rows.append(
                    {
                        "ts": ts_close[fired_at],
                        "signal": "s2",
                        "side": -1 if kind == "high" else 1,
                        "price": closes[fired_at],
                        "target": lows[i] if kind == "high" else highs[i],
                        "meta": price,
                    }
                )
                i = fired_at + 1  # episode closed; scan for the next pierce
            else:
                i += 1
    return _events(rows)


def s3_filter(
    events: pl.DataFrame,
    zones: pl.DataFrame,
    dense_quantile: float = 0.9,
) -> pl.DataFrame:
    """S3: veto entries whose path to target crosses a dense opposing zone.

    zones: (lo, hi, heat_usd). A zone is dense if heat >= the quantile of all
    zone heats. Returns events with blocked + block_reason columns.
    """
    if events.is_empty():
        return events.with_columns(
            pl.lit(False).alias("blocked"), pl.lit(None, dtype=pl.Float64).alias("block_zone")
        )
    threshold = float(zones["heat_usd"].quantile(dense_quantile)) if zones.height else float("inf")
    blocked: list[bool] = []
    zone_hit: list[float | None] = []
    for ev in events.iter_rows(named=True):
        lo, hi = sorted((ev["price"], ev["target"]))
        hit = None
        for z in zones.iter_rows(named=True):
            if z["heat_usd"] >= threshold and z["hi"] >= lo and z["lo"] <= hi:
                hit = (z["lo"] + z["hi"]) / 2
                break
        blocked.append(hit is not None)
        zone_hit.append(hit)
    return events.with_columns(
        pl.Series("blocked", blocked, dtype=pl.Boolean),
        pl.Series("block_zone", zone_hit, dtype=pl.Float64),
    )
