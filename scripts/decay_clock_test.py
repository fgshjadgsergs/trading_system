"""Каким «временем» мерить затухание: стенные часы, свечи или активность.

Идея «затухать не по времени, а по количеству свечей» проверяется тем же
тестом, что и всё остальное в карте: ОДИН ряд, нарезанный по-разному, обязан
давать одну и ту же карту на один и тот же момент. Позиция на бирже не знает,
какой у нас на экране таймфрейм.

  none     контроль без затухания — показывает, сколько расхождения даёт сама
           дискретизация баров (снятие ценой и размещение раз в бар);
  wall     dt = длительность бара (T½ в часах) — инвариант по построению;
  bars     dt = 1 бар (T½ в барах) — на 1m это часы, на 1h те же «48 баров»
           превращаются в двое суток: карта начинает зависеть от экрана;
  volume   dt = объём бара, пересчитанный в секунды среднего темпа торговли
           (T½ в часах СРЕДНЕЙ активности). Объём аддитивен, поэтому сумма
           эффективного времени по бару не зависит от нарезки — инвариант
           сохраняется, но часы идут быстрее на активном рынке.

    python scripts/decay_clock_test.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from scripts.map_multitf import GRID, SEED_W, synth_series
from trading_system.core.schema import Side
from trading_system.liqmap.buckets import PriceBuckets
from trading_system.liqmap.map import LiqMap, StaticWeights
from trading_system.viz.style import PALETTE, apply_style, save_fig

HOUR = 3_600.0


def aggregate(bars: pl.DataFrame, factor: int) -> pl.DataFrame:
    """Схлопывает `factor` минутных баров в один — тот же ряд, другая нарезка."""
    g = bars.with_row_index("i").with_columns((pl.col("i") // factor).alias("g"))
    return (
        g.group_by("g", maintain_order=True)
        .agg(
            ts_open=pl.col("ts_open").first(),
            ts_close=pl.col("ts_close").last(),
            open=pl.col("open").first(),
            high=pl.col("high").max(),
            low=pl.col("low").min(),
            close=pl.col("close").last(),
            volume=pl.col("volume").sum(),
            quote_volume=pl.col("quote_volume").sum(),
            d_oi_usd=pl.col("d_oi_usd").sum(),
            long_share=pl.col("long_share").last(),
        )
        .drop("g")
    )


def effective_dt(bars: pl.DataFrame, clock: str, bar_s: float, total_s: float) -> np.ndarray:
    n = bars.height
    if clock == "none":
        return np.zeros(n)  # контроль: затухания нет вовсе
    if clock == "wall":
        return np.full(n, bar_s)
    if clock == "bars":
        return np.full(n, 1.0)  # единица времени = один бар
    if clock == "volume":
        v = bars["quote_volume"].to_numpy().astype(float)
        rate = v.sum() / total_s  # средний объём в секунду за весь период
        return v / rate
    raise ValueError(f"неизвестные часы {clock}")


def build(bars: pl.DataFrame, dt: np.ndarray, half_life: float, bucket: float) -> dict:
    lm = LiqMap(leverage_grid=GRID, buckets=PriceBuckets(bucket),
                weight_fn=StaticWeights(SEED_W), decay_half_life_s=half_life)
    for r, d in zip(bars.iter_rows(named=True), dt.tolist(), strict=True):
        lm.step(r["low"], r["high"], r["close"], r["d_oi_usd"] or 0.0, dt_s=d,
                long_share=r["long_share"])
    out: dict[int, float] = {}
    for side in (Side.BUY, Side.SELL):
        for idx, h in lm.heat[side].items():
            out[idx] = out.get(idx, 0.0) + h
    return out


def cosine(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    va = np.array([a.get(k, 0.0) for k in keys])
    vb = np.array([b.get(k, 0.0) for k in keys])
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return float(va @ vb / (na * nb)) if na > 0 and nb > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="SUIUSDT")
    ap.add_argument("--price", type=float, default=3.42)
    ap.add_argument("--daily-vol", type=float, default=0.062)
    ap.add_argument("--oi-daily", type=float, default=30e6)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--half-life-h", type=float, default=24.0)
    ap.add_argument("--bucket-bps", type=float, default=25.0)
    ap.add_argument("--out", default="reports/decay_clock")
    args = ap.parse_args()

    minute = synth_series(1440 * args.days, 60.0, price0=args.price, symbol=args.symbol,
                          daily_vol=args.daily_vol, oi_daily_usd=args.oi_daily)
    total_s = 60.0 * minute.height
    bucket = float(minute["close"][-1]) * args.bucket_bps * 1e-4
    factors = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
    half_life_wall = args.half_life_h * HOUR
    # «T½ в барах» задаём так, чтобы на 1m он совпадал со стенным: 1440 баров
    half_life_bars = half_life_wall / 60.0

    results: dict[str, dict[str, float]] = {}
    for clock in ("none", "wall", "bars", "volume"):
        ref = None
        row = {}
        for tf, f in factors.items():
            bars = minute if f == 1 else aggregate(minute, f)
            dt = effective_dt(bars, clock, 60.0 * f, total_s)
            hl = half_life_bars if clock == "bars" else half_life_wall
            heat = build(bars, dt, hl, bucket)
            if ref is None:
                ref, ref_mass = heat, sum(heat.values())
            mass = sum(heat.values())
            row[tf] = (cosine(ref, heat), mass / max(ref_mass, 1e-9))
            print(f"{clock:>6} | {tf:>3} | косинус с 1m {row[tf][0]:6.3f} | "
                  f"масса {mass/1e6:7.3f} млн | отн. к 1m {mass/max(ref_mass,1e-9):6.2f}x | "
                  f"бакетов {len(heat):>4}")
        results[clock] = row
        print("-" * 96)

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    tfs = list(factors)
    x = np.arange(len(tfs))
    series = (
        ("none", "контроль: без затухания", PALETTE["grid"]),
        ("wall", "стенные часы (T½ в часах)", PALETTE["long"]),
        ("bars", "по свечам (T½ в барах)", PALETTE["short"]),
        ("volume", "по активности (T½ в часах среднего объёма)", PALETTE["accent"]),
    )
    for j, (ax, key, ylabel, title) in enumerate((
            (axes[0], 0, "совпадение с минутной картой (косинус)",
             "Форма карты"),
            (axes[1], 1, "масса карты относительно минутной",
             "Масса карты"))):
        for i, (clock, label, color) in enumerate(series):
            ax.bar(x + (i - 1.5) * 0.21, [results[clock][t][key] for t in tfs],
                   width=0.20, color=color, label=label if j == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(tfs)
        ax.axhline(1.0, color=PALETTE["neutral"], lw=0.8, ls="--")
        ax.set_xlabel("нарезка ОДНОГО И ТОГО ЖЕ ряда")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Тест инвариантности: та же серия, другая нарезка — та же карта?\n"
                 f"{args.symbol}, демо-ряд, {args.days} дней · T½={args.half_life_h:g} ч "
                 f"(для «по свечам» — {half_life_bars:g} баров, то же самое на 1m)", y=1.06)
    print(save_fig(fig, "decay_clock_invariance", Path(args.out)))


if __name__ == "__main__":
    main()
