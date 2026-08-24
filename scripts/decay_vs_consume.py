"""Кто на самом деле съедает уровни карты: проход цены или затухание.

Наблюдение с графика — «цена сняла ликвидность снизу одной свечой, но не
полностью, а затухают почему-то сразу все» — разбирается по двум механизмам:

  consume  локален: снимает ТОЛЬКО бакеты, пересечённые диапазоном свечи;
  decay    глобален: умножает КАЖДЫЙ бакет на 0.5^(dt/T½), независимо от цены.

На дневном баре при T½=24ч один бар — это ровно один полураспад, поэтому вся
карта разом теряет 50% массы, и визуально это неотличимо от «снятия».

    python scripts/decay_vs_consume.py --half-lives 24 72 168
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from scripts.map_multitf import GRID, SEED_W, synth_series  # noqa: E402
from trading_system.core.schema import Side
from trading_system.viz.style import PALETTE, apply_style, save_fig

BAR_S = 86_400.0


def _make_map(bars, half_life_s: float, close_out_fraction: float):
    from trading_system.liqmap.buckets import PriceBuckets
    from trading_system.liqmap.map import LiqMap, StaticWeights

    return LiqMap(
        leverage_grid=GRID,
        buckets=PriceBuckets.from_atr(float(np.nanmedian(bars["atr"])), 0.1),
        weight_fn=StaticWeights(SEED_W),
        decay_half_life_s=half_life_s,
        close_out_fraction=close_out_fraction,
    )


def anatomy(bars, bar_i: int, half_life_s: float, close_out_fraction: float) -> dict:
    """Состояние карты вокруг одного бара: до / после consume / после decay."""
    lm = _make_map(bars, half_life_s, close_out_fraction)
    rows = bars.to_dicts()
    for r in rows[:bar_i]:
        lm.step(r["low"], r["high"], r["close"], r["d_oi_usd"] or 0.0, dt_s=BAR_S,
                long_share=r.get("long_share"))
    r = rows[bar_i]
    total = {}
    for side in (Side.BUY, Side.SELL):
        for idx, h in lm.heat[side].items():
            total[idx] = total.get(idx, 0.0) + h
    before = dict(total)
    lo_i, hi_i = lm.buckets.index(r["low"]), lm.buckets.index(r["high"])
    in_range = {i for i in before if lo_i <= i <= hi_i}
    lm.consume(r["low"], r["high"])
    after_consume = {}
    for side in (Side.BUY, Side.SELL):
        for idx, h in lm.heat[side].items():
            after_consume[idx] = after_consume.get(idx, 0.0) + h
    lm.decay(BAR_S)  # без allocate: три состояния сравнимы по одним и тем же уровням
    after_decay = {}
    for side in (Side.BUY, Side.SELL):
        for idx, h in lm.heat[side].items():
            after_decay[idx] = after_decay.get(idx, 0.0) + h
    return {
        "before": before, "after_consume": after_consume, "after_decay": after_decay,
        "in_range": in_range, "bar": r, "buckets": lm.buckets,
        "survived_in": sum(1 for i in in_range if i in after_consume),
        "survived_out": sum(1 for i in before if i not in in_range and i in after_consume),
        "n_in": len(in_range), "n_out": len(before) - len(in_range),
        "mass_in": sum(before[i] for i in in_range),
        "mass_out": sum(h for i, h in before.items() if i not in in_range),
    }


def budget(bars, half_life_s: float, close_out_fraction: float) -> dict:
    """Побарный расход массы по механизмам за весь прогон."""
    lm = _make_map(bars, half_life_s, close_out_fraction)
    eaten, faded, closed, heat = [], [], [], []
    for r in bars.iter_rows(named=True):
        c0, d0, x0 = lm.consumed, lm.decayed, lm.removed
        lm.step(r["low"], r["high"], r["close"], r["d_oi_usd"] or 0.0, dt_s=BAR_S,
                long_share=r.get("long_share"))
        eaten.append(lm.consumed - c0)
        closed.append(lm.removed - x0)
        faded.append(lm.decayed - d0)
        heat.append(lm.total_heat())
    return {"eaten": np.array(eaten), "faded": np.array(faded),
            "closed": np.array(closed), "heat": np.array(heat)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="HYPEUSDT")
    ap.add_argument("--price", type=float, default=35.0)
    ap.add_argument("--daily-vol", type=float, default=0.055)
    ap.add_argument("--oi-daily", type=float, default=40e6)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--bar", type=int, default=20, help="номер бара для разбора анатомии")
    ap.add_argument("--half-lives", nargs="+", type=float, default=[24.0, 72.0, 168.0])
    ap.add_argument("--close-out-fraction", type=float, default=1.0)
    ap.add_argument("--out", default="reports/decay_vs_consume")
    args = ap.parse_args()

    bars = synth_series(args.days, BAR_S, price0=args.price, symbol=args.symbol,
                        daily_vol=args.daily_vol, oi_daily_usd=args.oi_daily)
    base_hl = args.half_lives[0] * 3600.0
    a = anatomy(bars, args.bar, base_hl, args.close_out_fraction)
    r = a["bar"]
    print(f"бар {args.bar}: диапазон {r['low']:.2f}–{r['high']:.2f}, "
          f"бакет {a['buckets'].bucket_size:.3f} ({a['buckets'].bucket_size / r['close'] * 100:.2f}% цены)")
    print(f"  до бара: {len(a['before'])} занятых бакетов "
          f"(в диапазоне свечи {a['n_in']}, вне {a['n_out']})")
    print(f"  масса: в диапазоне {a['mass_in']:,.0f}, вне {a['mass_out']:,.0f}")
    print(f"  после CONSUME выжило: внутри {a['survived_in']}/{a['n_in']}, "
          f"ВНЕ {a['survived_out']}/{a['n_out']}  <- consume строго локален")
    keep = 0.5 ** (BAR_S / base_hl)
    print(f"  после DECAY (dt=1 бар, T½={args.half_lives[0]:.0f}ч): "
          f"у КАЖДОГО уровня осталось {keep * 100:.1f}% массы")

    budgets = {hl: budget(bars, hl * 3600.0, args.close_out_fraction)
               for hl in args.half_lives}
    for hl, b in budgets.items():
        tot = b["eaten"].sum() + b["faded"].sum() + b["closed"].sum()
        print(f"T½={hl:>5.0f}ч | съедено ценой {b['eaten'].sum() / tot * 100:5.1f}% | "
              f"закрыто по ΔOI⁻ {b['closed'].sum() / tot * 100:5.1f}% | "
              f"затухло {b['faded'].sum() / tot * 100:5.1f}% | "
              f"тепло на конец {b['heat'][-1] / 1e6:6.2f} млн USD")

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    ax = axes[0]
    idxs = sorted(a["before"])
    y = np.arange(len(idxs))
    ax.barh(y + 0.26, [a["before"][i] / 1e3 for i in idxs], height=0.26,
            color=PALETTE["neutral"], label="до бара")
    ax.barh(y, [a["after_consume"].get(i, 0.0) / 1e3 for i in idxs], height=0.26,
            color=PALETTE["accent"], label="после прохода цены")
    ax.barh(y - 0.26, [a["after_decay"].get(i, 0.0) / 1e3 for i in idxs], height=0.26,
            color=PALETTE["short"], label="после затухания")
    for k, i in enumerate(idxs):
        if i in a["in_range"]:
            ax.axhspan(k - 0.45, k + 0.45, color=PALETTE["long"], alpha=0.13, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a['buckets'].center(i):.2f}" for i in idxs], fontsize=7)
    ax.set_xlabel("тепло, тыс. USD")
    ax.set_ylabel("цена уровня")
    ax.set_title(f"Бар {args.bar} (диапазон {r['low']:.2f}–{r['high']:.2f}, зелёная зона):\n"
                 f"цена снесла {a['n_in'] - a['survived_in']} из {a['n_in']} уровней внутри, "
                 f"вне — 0 из {a['n_out']};\nзатухание срезало ВСЕ на "
                 f"{(1 - 0.5 ** (BAR_S / base_hl)) * 100:.0f}%")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    b = budgets[args.half_lives[0]]
    n = min(60, len(b["eaten"]))
    x = np.arange(n)
    ax.bar(x, b["eaten"][:n] / 1e6, color=PALETTE["accent"], label="съела цена (локально)")
    ax.bar(x, b["closed"][:n] / 1e6, bottom=b["eaten"][:n] / 1e6,
           color=PALETTE["neutral"], label="закрытие позиций ΔOI⁻")
    ax.bar(x, b["faded"][:n] / 1e6, bottom=(b["eaten"][:n] + b["closed"][:n]) / 1e6,
           color=PALETTE["short"], label="затухло (глобально)")
    ax.set_xlabel("бар (дневной)")
    ax.set_ylabel("списано за бар, млн USD")
    ax.set_title(f"Расход массы по механизмам, T½={args.half_lives[0]:.0f} ч")
    ax.legend(fontsize=7)

    ax = axes[2]
    days = np.arange(0, 15)
    for hl, color in zip(args.half_lives,
                         [PALETTE["short"], PALETTE["accent"], PALETTE["long"]], strict=False):
        ax.plot(days, 0.5 ** (days * 24.0 / hl) * 100, marker="o", ms=3, color=color,
                label=f"T½={hl:.0f} ч")
    ax.set_xlabel("дней после появления уровня")
    ax.set_ylabel("остаток массы уровня, %")
    ax.set_title("Сколько живёт уровень при разном полураспаде")
    ax.legend(fontsize=8)

    fig.suptitle(f"{args.symbol} · 1d · демо-ряд: проход цены локален, затухание — нет", y=1.12)
    print(save_fig(fig, f"decay_vs_consume_{args.symbol.lower()}", Path(args.out)))


if __name__ == "__main__":
    main()
