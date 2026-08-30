"""Диагностика живого лейка: чем наполнена карта и почему.

Печатает состав лейка, распределение ΔOI из фида платформы и итог карты в
той же конфигурации, что у платформы. Запускать при странных картинках:

    python scripts/diag_lake.py --lake data/live_lake --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from trading_system.core.io import read_stream
from trading_system.platform.feed import LakeBarFeed
from trading_system.platform.state import LiveMapState


def show(lake: Path, sym: str) -> None:
    print(f"=== {sym} @ {lake} ===")
    for stream, ts_col in (("kline", "ts_close"), ("open_interest", "ts_event"),
                           ("ratio", "ts_event")):
        try:
            df = read_stream(lake, stream, symbol=sym)
        except FileNotFoundError:
            print(f"{stream:>14}: НЕТ")
            continue
        span_h = (df[ts_col].max() - df[ts_col].min()) / 3.6e12 if df.height else 0
        line = f"{stream:>14}: {df.height} строк, {span_h:.1f} ч"
        if stream == "open_interest":
            fin = df.filter(pl.col("open_interest_usd").is_finite())
            line += (f"; USD конечен {fin.height}/{df.height}"
                     + (f", диапазон {fin['open_interest_usd'].min()/1e9:.2f}–"
                        f"{fin['open_interest_usd'].max()/1e9:.2f} млрд"
                        if fin.height else ""))
        if stream == "kline":
            closed = int(df["closed"].sum())
            line += f"; закрытых {closed}, дублей ts {df.height - df['ts_close'].n_unique()}"
        print(line)

    bars = LakeBarFeed(lake, sym).poll()
    if not bars:
        print("фид не отдал ни одного бара — смотри, чего не хватает выше")
        return
    d = [b.d_oi_usd for b in bars]
    pos = sum(x for x in d if x > 0)
    neg = sum(x for x in d if x < 0)
    nz = sum(1 for x in d if x != 0)
    top = sorted(d, key=abs, reverse=True)[:5]
    print(f"баров из фида: {len(bars)} ({bars[0].ts_close} … {bars[-1].ts_close})")
    print(f"ΔOI: ненулевых {nz}; сумма+ {pos/1e6:.2f} млн USD; "
          f"сумма- {neg/1e6:.2f} млн; топ |ΔOI|: "
          + ", ".join(f"{x/1e6:.1f}M" for x in top))

    st = LiveMapState(sym, bucket_size=bars[0].close * 30e-4)  # как платформа
    for b in bars:
        st.apply_bar(b)
    m = st.map
    print(f"карта: тепло {m.total_heat():,.0f} USD | contributed {m.contributed:,.0f} "
          f"| consumed {m.consumed:,.0f} | removed {m.removed:,.0f} "
          f"| dropped {m.dropped:,.0f}")
    pools = sorted(
        ((idx, h) for side in m.heat.values() for idx, h in side.items()),
        key=lambda kv: -kv[1])[:8]
    for idx, h in pools:
        print(f"  пул {m.buckets.center(idx):>12,.1f} -> {h:,.0f} USD")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lake", default="data/live_lake")
    ap.add_argument("--symbol", default="BTCUSDT")
    args = ap.parse_args()
    show(Path(args.lake), args.symbol)


if __name__ == "__main__":
    main()
