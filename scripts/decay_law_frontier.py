"""Может ли ОДИН закон затухания обслуживать и минутный, и дневной график.

Наблюдение, с которого всё началось: на минутках карта читается, а на
старших ТФ уровни гаснут слишком быстро. Соблазн — сделать полураспад
функцией таймфрейма; но экспонента безпамятна, и одна и та же серия,
нарезанная в 1h и 4h, при равном T½ даёт совпадающие карты (проверено).
Значит, дело не в нарезке баров, а в ФОРМЕ закона.

Здесь измеряются два свойства карты, между которыми и идёт торг:

  контраст (минутный ряд) — доля массы в верхнем дециле бакетов на конец:
      мало ярких уровней = картинка читается, «всё поровну» = каша;
  память (дневной ряд) — средний косинус между вектором тепла и им же
      7 дней назад: та же ли это карта неделю спустя.

Одна экспонента при любом T½ идёт по компромиссной кривой: быстрый T½ даёт
контраст без памяти, медленный — память без контраста. Смесь экспонент
(убывающая интенсивность: молодое тепло гаснет быстро, дожившее живёт долго)
проверяется на том же графике — лежит ли она ВЫШЕ этой кривой.

    python scripts/decay_law_frontier.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from scripts.map_multitf import GRID, SEED_W, synth_series
from trading_system.core.schema import Side
from trading_system.liqmap.buckets import PriceBuckets
from trading_system.liqmap.map import LiqMap, StaticWeights
from trading_system.liqmap.mixture import MixtureLiqMap
from trading_system.viz.style import PALETTE, apply_style, save_fig

HOUR = 3_600.0
EXP_GRID_H = [2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 96.0, 168.0, 336.0, 720.0]
MIXTURES = {
    "смесь 0.75·4ч + 0.25·14д": [(0.75, 4 * HOUR), (0.25, 336 * HOUR)],
    "смесь 0.6·3ч + 0.4·30д": [(0.6, 3 * HOUR), (0.4, 720 * HOUR)],
    "смесь 0.5·8ч + 0.5·60д": [(0.5, 8 * HOUR), (0.5, 1440 * HOUR)],
    "смесь 0.4·2ч + 0.6·90д": [(0.4, 2 * HOUR), (0.6, 2160 * HOUR)],
}
TIERS = {
    "тиры плеч: ≤10 → 30д, ≤50 → 2д, >50 → 4ч":
        [(10.0, 720 * HOUR), (50.0, 48 * HOUR), (1e9, 4 * HOUR)],
}


def run(bars, bar_s: float, law, bucket_bps: float):
    """Прогон ряда под заданным законом; возвращает кадры (агрегат по сторонам)."""
    common = dict(leverage_grid=GRID,
                  buckets=PriceBuckets(float(bars["close"][-1]) * bucket_bps * 1e-4),
                  weight_fn=StaticWeights(SEED_W))
    if isinstance(law, float):
        lm = LiqMap(decay_half_life_s=law, **common)
    elif isinstance(law, tuple):  # ("tiers", [...])
        lm = MixtureLiqMap.by_leverage(law[1], **common)
    else:
        lm = MixtureLiqMap(law, **common)
    frames = []
    for r in bars.iter_rows(named=True):
        lm.step(r["low"], r["high"], r["close"], r["d_oi_usd"] or 0.0, dt_s=bar_s,
                long_share=r["long_share"])
        frame = {}
        for side in (Side.BUY, Side.SELL):
            for idx, h in lm.heat[side].items():
                frame[idx] = frame.get(idx, 0.0) + h
        frames.append(frame)
    return frames


def top_decile(frame: dict[int, float]) -> set[int]:
    if not frame:
        return set()
    k = max(1, round(len(frame) * 0.1))
    return {i for i, _ in sorted(frame.items(), key=lambda kv: -kv[1])[:k]}


def contrast(frames) -> float:
    """Доля массы в верхнем дециле бакетов (среднее по последней трети ряда)."""
    vals = []
    for frame in frames[len(frames) * 2 // 3:]:
        total = sum(frame.values())
        if total <= 0:
            continue
        vals.append(sum(frame[i] for i in top_decile(frame)) / total)
    return float(np.mean(vals)) if vals else 0.0


def persistence(frames, lag_bars: int) -> float:
    """Насколько карта через lag_bars — это ТА ЖЕ карта: косинус между
    векторами тепла (по объединению бакетов), усреднённый по ряду.

    Жаккар верхнего дециля здесь не годится: на дневной сетке занято ~14
    бакетов, верхний дециль — это один бакет, и метрика вырождается в
    ноль/единицу независимо от закона.
    """
    vals = []
    for i in range(lag_bars, len(frames)):
        a, b = frames[i - lag_bars], frames[i]
        keys = set(a) | set(b)
        if not keys:
            continue
        va = np.array([a.get(k, 0.0) for k in keys])
        vb = np.array([b.get(k, 0.0) for k in keys])
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na > 0 and nb > 0:
            vals.append(float(va @ vb / (na * nb)))
    return float(np.mean(vals)) if vals else 0.0


def carryover(frames, lag_bars: int) -> float:
    """Доля массы, стоящей в бакетах, которые были заняты уже lag_bars назад."""
    vals = []
    for i in range(lag_bars, len(frames)):
        old, new = set(frames[i - lag_bars]), frames[i]
        total = sum(new.values())
        if total > 0:
            vals.append(sum(h for k, h in new.items() if k in old) / total)
    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="SUIUSDT")
    ap.add_argument("--price", type=float, default=3.42)
    ap.add_argument("--daily-vol", type=float, default=0.062)
    ap.add_argument("--oi-daily", type=float, default=30e6)
    ap.add_argument("--minute-days", type=int, default=2)
    ap.add_argument("--daily-days", type=int, default=180)
    ap.add_argument("--bucket-bps", type=float, default=25.0)
    ap.add_argument("--out", default="reports/decay_law")
    args = ap.parse_args()

    kw = dict(price0=args.price, symbol=args.symbol,
              daily_vol=args.daily_vol, oi_daily_usd=args.oi_daily)
    m_bars = synth_series(1440 * args.minute_days, 60.0, **kw)
    d_bars = synth_series(args.daily_days, 86_400.0, **kw)

    laws = ([(f"экспонента {h:g} ч", float(h) * HOUR) for h in EXP_GRID_H]
            + [(name, comp) for name, comp in MIXTURES.items()]
            + [(name, ("tiers", t)) for name, t in TIERS.items()])
    rows = []
    for name, law in laws:
        mf = run(m_bars, 60.0, law, args.bucket_bps)
        df = run(d_bars, 86_400.0, law, args.bucket_bps)
        c = contrast(mf)                 # минутный ряд: читается ли картинка
        p = persistence(df, lag_bars=7)  # дневной ряд: та же ли карта через неделю
        co = carryover(df, lag_bars=7)
        heat_d = sum(df[-1].values())
        rows.append((name, c, p, heat_d, len(df[-1]), co))
        print(f"{name:<40} контраст(1m) {c:5.3f} | память(1d, 7д) {p:5.3f} | "
              f"масса в старых бакетах {co:5.3f} | тепло {heat_d/1e6:6.2f} млн | "
              f"бакетов {len(df[-1]):>4}")

    apply_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    exps = [r for r in rows if r[0].startswith("экспонента")]
    ax.plot([r[2] for r in exps], [r[1] for r in exps], "-o", color=PALETTE["neutral"],
            ms=5, lw=1.4, label="одна экспонента (свип T½)", zorder=2)
    for name, c, p, *_ in exps:
        ax.annotate(name.replace("экспонента ", ""), (p, c), fontsize=7,
                    xytext=(4, -8), textcoords="offset points", color=PALETTE["neutral"])
    for marker, color, group in (("s", PALETTE["accent"], MIXTURES),
                                 ("D", PALETTE["long"], TIERS)):
        pts = [r for r in rows if r[0] in group]
        ax.scatter([r[2] for r in pts], [r[1] for r in pts], marker=marker, s=70,
                   color=color, zorder=3,
                   label="смесь экспонент" if group is MIXTURES else "полураспад по плечам")
        for name, c, p, *_ in pts:
            ax.annotate(name, (p, c), fontsize=7, xytext=(6, 4),
                        textcoords="offset points", color=color)
    ax.set_xlabel("память: косинус между картой и ею же 7 дней назад (дневной ряд)")
    ax.set_ylabel("контраст: доля массы в верхнем дециле (минутный ряд)")
    ax.set_title("Один закон затухания на оба таймфрейма: где проходит компромисс\n"
                 f"{args.symbol}, демо-ряд · бакет {args.bucket_bps:g} bps · "
                 "правее и выше — лучше")
    ax.legend(fontsize=8, loc="upper right")
    print(save_fig(fig, "decay_law_frontier", Path(args.out)))


if __name__ == "__main__":
    main()
