"""15-second websocket probe: counts messages per stream on the SAME combined
URL the recorder subscribes to. Diagnoses "which streams actually arrive"
independently of the recording pipeline.

    python scripts/ws_probe.py --symbols ETHUSDT BTCUSDT SOLUSDT --seconds 15
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import websockets

from trading_system.collectors.binance import BinanceUsdmAdapter

WS_STREAMS = ["depth", "agg_trade", "force_order", "mark_price", "kline_1m"]


async def probe(symbols: list[str], seconds: float) -> None:
    adapter = BinanceUsdmAdapter()
    from trading_system.collectors.binance import combined_ws_url

    url = combined_ws_url(adapter.stream_names(symbols, WS_STREAMS))
    print(f"URL: {url}\nслушаю {seconds:.0f} секунд...")
    counts: Counter[str] = Counter()
    async with websockets.connect(url, max_size=None) as ws:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + seconds
        while loop.time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - loop.time()))
            except TimeoutError:
                break
            try:
                tag = json.loads(msg).get("stream", "<no stream tag>")
            except (json.JSONDecodeError, AttributeError):
                tag = "<unparsable>"
            # collapse per-symbol tags to the stream kind
            kind = tag.split("@", 1)[1] if "@" in tag else tag
            counts[kind] += 1
    print("\nсообщений по видам потоков:")
    for kind, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:>14}: {n}")
    expected = {"depth@100ms", "aggTrade", "markPrice@1s", "kline_1m"}
    missing = expected - {k for k in counts}
    if missing:
        print(f"\nНЕ ПРИШЛИ: {sorted(missing)} (forceOrder редкий — его отсутствие нормально)")
    else:
        print("\nвсе ожидаемые потоки живы (forceOrder редкий — придёт при ликвидациях)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["ETHUSDT", "BTCUSDT", "SOLUSDT"])
    parser.add_argument("--seconds", type=float, default=15.0)
    args = parser.parse_args()
    asyncio.run(probe(args.symbols, args.seconds))


if __name__ == "__main__":
    main()
