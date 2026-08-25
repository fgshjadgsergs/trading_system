"""Карта не должна меняться от того, какой таймфрейм на экране.

Эталон — карта, построенная на самом мелком доступном разрешении (1m).
Сравниваются три способа получить картинку на старшем ТФ:

  пересборка   карта строится заново из баров этого ТФ (как было);
  путь o→h→l→c бар разбивается на ноги, ΔOI и затухание делятся по длине
               ног — попытка вернуть грубому бару «мелкие шаги»;
  одна карта   карта строится один раз на 1m, старший ТФ только выбирает
               моменты показа (`HeatHistory.resample`).

Метрика — косинус между итоговым вектором тепла и эталонным минутным на тот
же момент, плюс отношение массы. Затухание тут ни при чём: контроль без
затухания в scripts/decay_clock_test.py даёт ту же деградацию.

    python scripts/tf_consistency.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from scripts.map_multitf import GRID, SEED_W, aggregate_bars, synth_series
from trading_system.core.schema import Side
from trading_system.liqmap.buckets import PriceBuckets
from trading_system.liqmap.history import HeatHistory
from trading_system.liqmap.map import LiqMap, StaticWeights
from trading_system.viz.style import PALETTE, apply_style, save_fig

HOUR = 3_600.0
FACTORS = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}


def frame(lm) -> dict[int, float]:
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


def build(bars: pl.DataFrame, bar_s: float, bucket: float, half_life: float,
          path_legs: bool = False) -> tuple[LiqMap, HeatHistory]:
    lm = LiqMap(leverage_grid=GRID, buckets=PriceBuckets(bucket),
                weight_fn=StaticWeights(SEED_W), decay_half_life_s=half_life)
    hist = HeatHistory(lm)
    for r in bars.iter_rows(named=True):
        if not path_legs:
            lm.step(r["low"], r["high"], r["close"], r["d_oi_usd"] or 0.0, dt_s=bar_s,
                    long_share=r["long_share"])
        else:
            o, hi, lo, c = r["open"], r["high"], r["low"], r["close"]
            path = [o, lo, hi, c] if c >= o else [o, hi, lo, c]
            legs = [abs(path[i + 1] - path[i]) for i in range(3)]
            total = sum(legs) or 1.0
            for i, ln in enumerate(legs):
                share = ln / total
                a, b = path[i], path[i + 1]
                lm.consume(min(a, b), max(a, b))
                lm.allocate((r["d_oi_usd"] or 0.0) * share, b, long_share=r["long_share"])
                lm.decay(bar_s * share)
        hist.record(r["ts_close"])
    return lm, hist


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="SUIUSDT")
    ap.add_argument("--price", type=float, default=3.42)
    ap.add_argument("--daily-vol", type=float, default=0.062)
    ap.add_argument("--oi-daily", type=float, default=30e6)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--half-life-h", type=float, default=24.0)
    ap.add_argument("--bucket-bps", type=float, default=25.0)
    ap.add_argument("--out", default="reports/tf_consistency")
    args = ap.parse_args()

    minute = synth_series(1440 * args.days, 60.0, price0=args.price, symbol=args.symbol,
                          daily_vol=args.daily_vol, oi_daily_usd=args.oi_daily)
    bucket = float(minute["close"][-1]) * args.bucket_bps * 1e-4
    half_life = args.half_life_h * HOUR
    ref_map, ref_hist = build(minute, 60.0, bucket, half_life)
    ref, ref_mass = frame(ref_map), ref_map.total_heat()
    print(f"эталон 1m: масса {ref_mass/1e6:.3f} млн USD, бакетов {len(ref)}")

    results: dict[str, dict[str, tuple[float, float]]] = {k: {} for k in
                                                          ("пересборка", "путь", "одна карта")}
    for tf, f in FACTORS.items():
        bars = aggregate_bars(minute, f)
        for name, kwargs in (("пересборка", {}), ("путь", {"path_legs": True})):
            lm, _ = build(bars, 60.0 * f, bucket, half_life, **kwargs)
            results[name][tf] = (cosine(ref, frame(lm)), lm.total_heat() / ref_mass)
        # одна карта: та же модель, только моменты показа
        view = ref_hist.resample(bars["ts_close"])
        shown = dict(zip(view.zones_at(len(view) - 1)[0], view.zones_at(len(view) - 1)[2],
                         strict=True))
        shown = {int(round(lo / bucket)): h for lo, h in shown.items()}
        results["одна карта"][tf] = (cosine(ref, shown),
                                     sum(shown.values()) / ref_mass)
        for name in results:
            c, m = results[name][tf]
            print(f"  {tf:>3} {name:>10} | косинус с 1m {c:6.3f} | масса {m:5.2f}x")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    tfs = list(FACTORS)
    x = np.arange(len(tfs))
    series = (("пересборка", "пересборка карты на каждом ТФ", PALETTE["short"]),
              ("путь", "пересборка с разбиением бара на путь", PALETTE["accent"]),
              ("одна карта", "одна карта, ТФ — только окно показа", PALETTE["long"]))
    for j, (ax, key, ylabel) in enumerate(((axes[0], 0, "косинус с минутной картой"),
                                           (axes[1], 1, "масса относительно минутной"))):
        for i, (name, label, color) in enumerate(series):
            ax.bar(x + (i - 1) * 0.27, [results[name][t][key] for t in tfs], width=0.26,
                   color=color, label=label if j == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(tfs)
        ax.axhline(1.0, color=PALETTE["neutral"], lw=0.8, ls="--")
        ax.set_xlabel("таймфрейм показа")
        ax.set_ylabel(ylabel)
    axes[0].set_ylim(0, 1.08)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Одна и та же история, разные таймфреймы: та же ли карта?\n"
                 f"{args.symbol}, демо-ряд, {args.days} дней · T½={args.half_life_h:g} ч · "
                 f"бакет {args.bucket_bps:g} bps", y=1.04)
    print(save_fig(fig, "tf_consistency", Path(args.out)))


if __name__ == "__main__":
    main()
