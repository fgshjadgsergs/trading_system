"""Live recorder (этап 1.1): Binance USDT-M websocket + REST -> parquet lake.

Runs day-1 collection: depth@100ms with REST snapshot sync, aggTrade,
forceOrder, markPrice@1s, kline_1m (checksum stream), plus REST pollers for
openInterest (5-10s) and the three long/short ratios (5 min). Raw messages
are normalized to the unified schema and batch-written append-only with
hourly rotation and exchange/symbol/date partitions.

Requires network access to fstream.binance.com / fapi.binance.com.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.parse
import urllib.request
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import structlog

from trading_system.collectors.binance import (
    BinanceUsdmAdapter,
    parse_global_ls_account,
    parse_open_interest,
    parse_taker_ls,
    parse_top_ls_position,
)
from trading_system.collectors.quality import write_gap_events
from trading_system.collectors.recorder import BatchWriter, RestPoller
from trading_system.collectors.sequencer import DepthSequencer, GapEvent
from trading_system.collectors.ws_client import websockets_transport_factory
from trading_system.core.config import load_config
from trading_system.core.schema import BookSnapshot, DepthDiff
from trading_system.core.timeutils import now_ns

log = structlog.get_logger()

WS_STREAMS = ["depth", "agg_trade", "force_order", "mark_price", "kline_1m"]


async def http_get_json(url: str, params: dict | None = None) -> dict:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    def _fetch() -> dict:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())

    return await asyncio.to_thread(_fetch)


async def run(symbols: list[str], lake: Path, cfg: dict) -> None:
    ex_cfg = cfg["exchanges"]["binance_usdm"]
    col_cfg = cfg["collect"]
    adapter = BinanceUsdmAdapter(
        http_get=http_get_json,
        ws_connect=websockets_transport_factory,
        rest_base=ex_cfg["rest_base"],
        ws_base=ex_cfg["ws_base"],
    )
    writer = BatchWriter(
        lake,
        max_rows=int(col_cfg["batch_max_rows"]),
        max_age_s=float(col_cfg["batch_max_age_s"]),
    )
    gap_log: list[GapEvent] = []
    sequencers: dict[str, DepthSequencer] = {}
    resync_pending: set[str] = set()

    def on_gap(gap: GapEvent) -> None:
        gap_log.append(gap)
        resync_pending.add(gap.symbol)
        log.warning("depth_gap", symbol=gap.symbol, expected=gap.expected, got=gap.got)

    async def resync(symbol: str) -> None:
        snap = await adapter.snapshot(symbol)
        for ready in sequencers[symbol].set_snapshot(snap):
            writer.add(ready)
        writer.add(snap)
        log.info("book_resynced", symbol=symbol, last_update_id=snap.last_update_id)

    for sym in symbols:
        sequencers[sym] = DepthSequencer(
            sym, on_gap=on_gap, on_resync=lambda s=sym: resync_pending.add(s)
        )
        resync_pending.add(sym)

    rest_base = ex_cfg["rest_base"]
    pollers = []
    for sym in symbols:
        pollers.append(
            RestPoller(
                float(col_cfg["open_interest_poll_s"]),
                partial(http_get_json, f"{rest_base}/fapi/v1/openInterest", {"symbol": sym}),
                parse_open_interest,
                writer.add,
            )
        )
        for path, parser in (
            ("/futures/data/globalLongShortAccountRatio", parse_global_ls_account),
            ("/futures/data/topLongShortPositionRatio", parse_top_ls_position),
            ("/futures/data/takerlongshortRatio", parse_taker_ls),
        ):
            pollers.append(
                RestPoller(
                    float(col_cfg["ratios_poll_s"]),
                    partial(http_get_json, f"{rest_base}{path}", {"symbol": sym, "period": "5m", "limit": 1}),
                    parser,
                    writer.add,
                )
            )
    poller_tasks = [asyncio.create_task(p.run()) for p in pollers]

    async def flush_gaps_periodically() -> None:
        while True:
            await asyncio.sleep(60)
            writer.poll()
            if gap_log:
                write_gap_events(lake, list(gap_log), "binance_usdm", "depth")
                gap_log.clear()

    flush_task = asyncio.create_task(flush_gaps_periodically())

    try:
        async for raw in adapter.subscribe(symbols, WS_STREAMS):
            for rec in adapter.normalize(raw):
                if isinstance(rec, DepthDiff):
                    seq = sequencers[rec.symbol]
                    for ready in seq.add_diff(rec):
                        writer.add(ready)
                    if rec.symbol in resync_pending:
                        resync_pending.discard(rec.symbol)
                        await resync(rec.symbol)
                elif isinstance(rec, BookSnapshot):
                    writer.add(rec)
                else:
                    writer.add(rec)
    finally:
        flush_task.cancel()
        for t in poller_tasks:
            t.cancel()
        writer.flush_all()
        if gap_log:
            write_gap_events(lake, list(gap_log), "binance_usdm", "depth")
        log.info("recorder_stopped", ts=now_ns())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--lake", default="data/lake")
    args = parser.parse_args()
    cfg = load_config(args.config)
    symbols = args.symbols or cfg["symbols"]
    lake = Path(args.lake)
    lake.mkdir(parents=True, exist_ok=True)
    log.info("recorder_start", symbols=symbols, lake=str(lake))
    asyncio.run(run(symbols, lake, cfg))


if __name__ == "__main__":
    main()
