"""Рендер карты ликвидности тем же кодом, что работает на реальных данных.

Данные Binance из песочницы недоступны (прокси), поэтому ряд — реалистичная
сидированная синтетика в масштабе ETHUSDT (5m, 3 суток): GBM с кластеризацией
волатильности, всплески открытого интереса на импульсах, доли сторон из
имитации ратио-потоков.
"""
from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from trading_system.liqmap.buckets import PriceBuckets
from trading_system.liqmap.history import HeatHistory
from trading_system.liqmap.map import LiqMap, StaticWeights
from trading_system.liqmap.terminal import terminal_heat_overlay

OUT = "reports/showcase"
MIN5 = 300_000_000_000


def eth_like_bars(n: int = 288, seed: int = 20) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    vol = 0.0016 * np.exp(np.cumsum(rng.normal(0, 0.05, n)) * 0.3)  # кластеризация волы
    ret = rng.normal(0, 1, n) * vol
    ret[110:120] += 0.0035  # импульс вверх
    ret[210:218] -= 0.0045  # сброс
    price = 2320.0 * np.exp(np.cumsum(ret))
    hi = price * (1 + np.abs(rng.normal(0, 0.4, n)) * vol)
    lo = price * (1 - np.abs(rng.normal(0, 0.4, n)) * vol)
    op = np.concatenate([[price[0]], price[:-1]])
    hi = np.maximum(hi, np.maximum(op, price))
    lo = np.minimum(lo, np.minimum(op, price))
    # приток OI растёт на импульсах и падает на откатах
    impulse = np.abs(ret) / (vol + 1e-12)
    d_oi = (impulse - 0.75) * 2.2e5 + rng.normal(0, 6e4, n)
    ts = np.arange(1, n + 1, dtype=np.int64) * MIN5
    tr = np.maximum(hi - lo, np.abs(hi - op))
    atr = pl.Series(tr).rolling_mean(14, min_samples=1).to_numpy()
    long_share = np.clip(0.5 + 0.35 * np.tanh(pl.Series(ret).rolling_mean(24, min_samples=1).to_numpy() / (vol + 1e-12)), 0.15, 0.85)
    return pl.DataFrame({
        "symbol": ["ETHUSDT"] * n, "ts_open": ts - MIN5, "ts_close": ts,
        "open": op, "high": hi, "low": lo, "close": price,
        "volume": np.abs(rng.normal(4_000, 900, n)),
        "quote_volume": np.abs(rng.normal(4_000, 900, n)) * price,
        "d_oi_usd": d_oi, "atr": atr, "long_share": long_share,
    })


def build(bars: pl.DataFrame, **map_kw) -> tuple[LiqMap, HeatHistory]:
    grid = [3, 5, 10, 20, 25, 30, 40, 50, 60, 75, 100, 125]
    seed_w = np.array([1, 2, 4, 6, 5, 5, 4, 4, 3, 2, 2, 1], dtype=float)
    lm = LiqMap(
        leverage_grid=grid,
        buckets=PriceBuckets.from_atr(float(np.median(bars["atr"])), 0.1),
        weight_fn=StaticWeights(seed_w),
        decay_half_life_s=86_400.0,
        **map_kw,
    )
    hist = HeatHistory(lm)
    for r in bars.iter_rows(named=True):
        lm.step(r["low"], r["high"], r["close"], r["d_oi_usd"], dt_s=300.0,
                long_share=r["long_share"])
        hist.record(r["ts_close"])
    return lm, hist


if __name__ == "__main__":
    bars = eth_like_bars()
    lm1, h1 = build(bars)
    p1 = terminal_heat_overlay(
        bars, h1, name="eth_map_base", out_dir=OUT,
        title="ETHUSDT · карта ликвидаций · сутки 5m · базовая конфигурация",
    )
    lm2, h2 = build(bars, blur_sigma0_bps=3.5, blur_sigma1=0.03,
                    fractional_edge_consume=True, typical_account_usd=20_000.0)
    p2 = terminal_heat_overlay(
        bars, h2, name="eth_map_improved", out_dir=OUT,
        title="ETHUSDT · та же карта: ядро размытия + частичное съедание краёв + тир по типичному счёту",
    )
    import matplotlib.pyplot as plt

    from trading_system.viz.style import PALETTE, apply_style, save_fig
    snap = lm2.snapshot()
    last = float(bars["close"][-1])
    # агрегируем тонкие бакеты в читаемые ценовые полосы (0.25% цены)
    band = last * 0.0025
    lo_p, hi_p = last * 0.92, last * 1.08
    edges = np.arange(lo_p, hi_p + band, band)
    idx = np.digitize(snap["prices"], edges) - 1
    keep = (idx >= 0) & (idx < len(edges) - 1)
    longs = np.bincount(idx[keep], weights=snap["long"][keep], minlength=len(edges) - 1)
    shorts = np.bincount(idx[keep], weights=snap["short"][keep], minlength=len(edges) - 1)
    centers = edges[:-1] + band / 2
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(centers, longs / 1e3, color=PALETTE["long"], height=band * 0.85,
            label="лонг-пулы — ликвидируются при падении")
    ax.barh(centers, -shorts / 1e3, color=PALETTE["short"], height=band * 0.85,
            label="шорт-пулы — ликвидируются при росте")
    ax.axhline(last, color=PALETTE["accent"], lw=1.6)
    ax.text(ax.get_xlim()[1] * 0.98, last + band, f"цена {last:,.0f}", ha="right",
            color=PALETTE["accent"], fontsize=10, fontweight="bold")
    ax.axvline(0, color=PALETTE["neutral"], lw=1)
    ax.set_xlabel("тепло в полосе, тыс. USD")
    ax.set_ylabel("цена")
    ax.set_ylim(lo_p, hi_p)
    ax.set_title("Срез карты на конец суток: сколько денег ждёт ликвидации на каждом уровне")
    ax.legend(loc="upper left", fontsize=9)
    p3 = save_fig(fig, "eth_map_profile", OUT)
    tot_l, tot_s = float(longs.sum()), float(shorts.sum())
    print(f"в окне ±8%: лонг-тепло {tot_l:,.0f} USD, шорт-тепло {tot_s:,.0f} USD")
    for path in (p1, p2, p3):
        print(path)
    print(f"баров: {bars.height}, занятых бакетов: base={sum(len(x) for x in lm1.heat.values())}, "
          f"improved={sum(len(x) for x in lm2.heat.values())}")
    print(f"тепло на конец: base={lm1.total_heat():,.0f} USD, improved={lm2.total_heat():,.0f} USD")
