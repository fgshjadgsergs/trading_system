"""Бэкфилл лейка историей по REST: свечи + открытый интерес + ратио.

Binance отдаёт задним числом: 1m-свечи (любая глубина), историю открытого
интереса и ратио-метрик — последние ~30 дней с шагом 5 минут. Этого
достаточно, чтобы карта стартовала с многодневной историей, а лайв её
продолжил. НЕ восстановимы задним числом: ликвидации (forceOrder), стакан
и лента сделок — их пишет только живой рекордер.

    python scripts/backfill_lake.py --lake data/live_lake --days 3 \
        --symbols BTCUSDT ETHUSDT SOLUSDT SUIUSDT

Скрипт идемпотентен: пишет только записи новее уже лежащих в лейке
(по ts), поэтому его можно гонять повторно и ПОВЕРХ идущей записи.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import structlog

from scripts.record_live import http_get_json as _http_get_json
from trading_system.collectors.binance import (
    parse_global_ls_account,
    parse_rest_klines,
    parse_taker_ls,
    parse_top_ls_position,
)
from trading_system.core.io import read_stream, write_batch
from trading_system.core.schema import OpenInterest, records_to_frame
from trading_system.core.timeutils import ms_to_ns, now_ns

log = structlog.get_logger()

async def http_get_json(url: str, params: dict | None = None) -> dict | list:
    """Обёртка с видимым прогрессом и ретраями: бэкфилл — интерактивный
    скрипт, молчаливое зависание на одном запросе выглядит как смерть."""
    for attempt in range(4):
        try:
            return await asyncio.wait_for(_http_get_json(url, params), timeout=15)
        except Exception as exc:  # noqa: BLE001 - ретраим любой сбой запроса
            wait = 2 ** attempt
            print(f"  запрос не прошёл ({exc!r}), повтор через {wait} с "
                  f"[{url.rsplit('/', 1)[-1]}]", flush=True)
            await asyncio.sleep(wait)
    raise SystemExit(f"эндпоинт не отвечает после 4 попыток: {url}")


MS_MIN = 60_000
RATIO_PARSERS = {
    "globalLongShortAccountRatio": parse_global_ls_account,
    "topLongShortPositionRatio": parse_top_ls_position,
    "takerlongshortRatio": parse_taker_ls,
}


def last_ts_in_lake(lake: Path, stream: str, symbol: str, col: str) -> int:
    try:
        df = read_stream(lake, stream, symbol=symbol)
    except FileNotFoundError:
        return 0
    return 0 if df.is_empty() else int(df[col].max())


def parse_oi_hist(payload: list[dict], ts_recv: int) -> list[OpenInterest]:
    """/futures/data/openInterestHist -> OpenInterest (USD прямо в ответе)."""
    return [
        OpenInterest(
            exchange="binance_usdm",
            symbol=it["symbol"],
            ts_event=ms_to_ns(int(it["timestamp"])),
            ts_recv=ts_recv,
            open_interest=float(it["sumOpenInterest"]),
            open_interest_usd=float(it["sumOpenInterestValue"]),
        )
        for it in payload
    ]


async def backfill_symbol(lake: Path, rest_base: str, sym: str, days: int) -> dict[str, int]:
    added = {"kline": 0, "open_interest": 0, "ratio": 0}
    print(f"{sym}: тяну свечи…", flush=True)
    now_ms = now_ns() // 1_000_000
    start_ms = now_ms - days * 86_400_000

    # -- свечи 1m: пагинация по startTime, максимум 1500 за запрос ------------
    have = last_ts_in_lake(lake, "kline", sym, "ts_close")
    cursor = max(start_ms, have // 1_000_000 + 1)
    while cursor < now_ms:
        payload = await http_get_json(
            f"{rest_base}/fapi/v1/klines",
            {"symbol": sym, "interval": "1m", "startTime": cursor, "limit": 1500},
        )
        if not payload:
            break
        klines = [k for k in parse_rest_klines(payload, sym)
                  if k.closed and k.ts_close > have]
        if klines:
            write_batch(lake, "kline", records_to_frame(klines, "kline"))
            added["kline"] += len(klines)
            have = klines[-1].ts_close
        next_cursor = int(payload[-1][0]) + MS_MIN
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        await asyncio.sleep(0.15)  # вежливый темп: лимиты REST общие с рекордером

    # -- открытый интерес: /futures/data/openInterestHist, шаг 5m -------------
    print(f"{sym}: свечей +{added['kline']}, тяну открытый интерес…", flush=True)
    have = last_ts_in_lake(lake, "open_interest", sym, "ts_event")
    cursor = max(start_ms, have // 1_000_000 + 1)
    while cursor < now_ms:
        payload = await http_get_json(
            f"{rest_base}/futures/data/openInterestHist",
            {"symbol": sym, "period": "5m", "startTime": cursor,
             "endTime": min(cursor + 499 * 300_000, now_ms), "limit": 500},
        )
        if not isinstance(payload, list) or not payload:
            break
        recs = [r for r in parse_oi_hist(payload, now_ns()) if r.ts_event > have]
        if recs:
            write_batch(lake, "open_interest",
                        records_to_frame(recs, "open_interest"))
            added["open_interest"] += len(recs)
            have = recs[-1].ts_event
        cursor = int(payload[-1]["timestamp"]) + 300_000
        await asyncio.sleep(0.15)

    # -- ратио: три метрики, тот же формат окна -------------------------------
    print(f"{sym}: OI +{added['open_interest']}, тяну ратио…", flush=True)
    have = last_ts_in_lake(lake, "ratio", sym, "ts_event")
    for path, parser in RATIO_PARSERS.items():
        cursor = max(start_ms, have // 1_000_000 + 1)
        while cursor < now_ms:
            payload = await http_get_json(
                f"{rest_base}/futures/data/{path}",
                {"symbol": sym, "period": "5m", "startTime": cursor,
                 "endTime": min(cursor + 499 * 300_000, now_ms), "limit": 500},
            )
            if not isinstance(payload, list) or not payload:
                break
            recs = [r for r in parser(payload, now_ns(), symbol=sym)
                    if r.ts_event > have]
            if recs:
                write_batch(lake, "ratio", records_to_frame(recs, "ratio"))
                added["ratio"] += len(recs)
            cursor = int(payload[-1]["timestamp"]) + 300_000
            await asyncio.sleep(0.15)

    return added


async def main_async(args: argparse.Namespace) -> None:
    lake = Path(args.lake)
    lake.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for sym in args.symbols:
        added = await backfill_symbol(lake, args.rest_base, sym, args.days)
        log.info("backfill_done", symbol=sym, **added)
        print(f"{sym:>9}: свечей +{added['kline']}, OI +{added['open_interest']}, "
              f"ратио +{added['ratio']}")
    print(f"готово за {time.time() - t0:.0f} с; лейк: {lake}")
    # контрольный вывод: сколько теперь истории по первому символу
    df = read_stream(lake, "kline", symbol=args.symbols[0])
    if not df.is_empty():
        span_h = (df["ts_close"].max() - df["ts_close"].min()) / 3.6e12
        print(f"{args.symbols[0]}: {df.height} свечей, {span_h:.1f} ч истории")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lake", default="data/live_lake")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    ap.add_argument("--days", type=int, default=3,
                    help="глубина истории (свечи — любая, OI/ратио — максимум ~30)")
    ap.add_argument("--rest-base", default="https://fapi.binance.com")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
