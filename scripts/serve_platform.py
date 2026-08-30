"""Живая платформа карты ликвидаций.

Демо без сети и лейка (минутный бар каждую секунду):
    python scripts/serve_platform.py --demo --symbols BTCUSDT ETHUSDT

На машине записи, поверх живого лейка:
    python scripts/serve_platform.py --lake data/live_lake --symbols BTCUSDT ETHUSDT

Открыть http://127.0.0.1:8080 — карта дорисовывается по мере закрытия
баров; при рестарте сервера клиент замечает смену эпохи и один раз
перечитывает снапшот.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import structlog

from trading_system.platform.feed import DemoBarFeed, LakeBarFeed
from trading_system.platform.server import Platform, serve
from trading_system.platform.state import LiveMapState

log = structlog.get_logger()

DEMO_PRICE = {"BTCUSDT": 65_000.0, "ETHUSDT": 3_200.0, "SOLUSDT": 150.0,
              "SUIUSDT": 3.4, "DOGEUSDT": 0.2, "XRPUSDT": 2.2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lake", default=None, help="живой лейк рекордера")
    ap.add_argument("--demo", action="store_true", help="демо-фид без сети")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bucket-bps", type=float, default=10.0,
                    help="шаг ценового бакета в bps от цены (10 = 0.1%%)")
    ap.add_argument("--half-life-h", type=float, default=float("inf"),
                    help="полураспад; по умолчанию inf — уровни живут до снятия ценой")
    ap.add_argument("--close-out-fraction", type=float, default=0.0)
    ap.add_argument("--poll-s", type=float, default=2.0)
    ap.add_argument("--demo-speed", type=float, default=60.0,
                    help="демо: во сколько раз ускорить время (60 = бар в секунду)")
    args = ap.parse_args()
    if not args.demo and not args.lake:
        raise SystemExit("нужен --lake или --demo")

    platform = Platform(poll_s=args.poll_s)
    half_life_s = (args.half_life_h * 3600.0
                   if args.half_life_h != float("inf") else float("inf"))

    def make_state(sym: str, price0: float) -> LiveMapState:
        return LiveMapState(
            sym,
            bucket_size=price0 * args.bucket_bps * 1e-4,
            decay_half_life_s=half_life_s,
            close_out_fraction=args.close_out_fraction,
            # свеча, задевшая КРАЙ бакета, снимает только пройденную долю —
            # иначе фитиль в треть полосы стирает её целиком
            fractional_edge_consume=True,
        )

    for i, sym in enumerate(args.symbols):
        if args.demo:
            price0 = DEMO_PRICE.get(sym, 100.0)
            feed = DemoBarFeed(sym, price0=price0, seed=7 + i, speed=args.demo_speed)
            platform.add_symbol(make_state(sym, price0), feed)
        else:
            # масштаб сетки — от цены ПЕРВОГО реального бара; до первых
            # баров символ отвечает «прогрев», а не карту не в том масштабе
            platform.add_symbol_lazy(
                sym, LakeBarFeed(args.lake, sym),
                lambda first, s=sym: make_state(s, first.close),
            )
    platform.start()
    httpd = serve(platform, args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        platform.stop()
        httpd.shutdown()


if __name__ == "__main__":
    main()
