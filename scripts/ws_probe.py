"""Websocket-диагностика: считает сообщения по видам потоков тремя способами.

Режимы (--mode):
    url        combined-URL подписка, как в рекордере: /stream?streams=a/b/c
    subscribe  подключение к /ws БЕЗ потоков в URL, подписка JSON-сообщением
               SUBSCRIBE — сервер отвечает ack'ом (виден отказ/ошибка), затем
               LIST_SUBSCRIPTIONS показывает, что сервер думает о подписке
    raw        по одному соединению /ws/<stream> на каждый вид потока
               (первый символ), последовательно

Примеры:
    python scripts/ws_probe.py --seconds 15
    python scripts/ws_probe.py --kinds mark_price --seconds 20
    python scripts/ws_probe.py --mode subscribe --seconds 20
    python scripts/ws_probe.py --mode raw --seconds 10
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

from trading_system.collectors.binance import WS_BASE, BinanceUsdmAdapter, combined_ws_url

WS_STREAMS = ["depth", "agg_trade", "force_order", "mark_price", "kline_1m"]


def _tag(msg: str) -> str:
    """Human tag for a message: combined wrapper stream, bare event, or ack."""
    try:
        obj = json.loads(msg)
    except json.JSONDecodeError:
        return "<unparsable>"
    if not isinstance(obj, dict):
        return "<non-dict>"
    if "stream" in obj:
        tag = obj["stream"]
        return tag.split("@", 1)[1] if "@" in tag else tag
    if "id" in obj:  # SUBSCRIBE / LIST_SUBSCRIPTIONS reply
        return f"ack:{obj['id']}"
    if "e" in obj:
        return obj["e"]
    return "<no tag>"


async def _drain(ws, seconds: float, counts: Counter, samples: dict[str, str]) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + seconds
    while loop.time() < deadline:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - loop.time()))
        except TimeoutError:
            break
        except Exception as exc:  # noqa: BLE001 - диагностике важен сам факт обрыва
            print(f"  !! соединение закрылось: {exc}")
            break
        if isinstance(msg, bytes):
            msg = msg.decode(errors="replace")
        tag = _tag(msg)
        counts[tag] += 1
        if tag not in samples:
            samples[tag] = msg[:160]
            if tag.startswith("ack:"):  # ответы сервера показываем сразу
                print(f"  ← {msg[:300]}")


def _report(counts: Counter, samples: dict[str, str]) -> None:
    print("\nсообщений по видам потоков:")
    if not counts:
        print("  (ни одного сообщения)")
    for kind, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:>16}: {n}")
    print("\nпервые сообщения:")
    for kind, s in samples.items():
        print(f"  [{kind}] {s}")


async def probe_url(symbols: list[str], kinds: list[str], seconds: float) -> None:
    adapter = BinanceUsdmAdapter()
    url = combined_ws_url(adapter.stream_names(symbols, kinds))
    print(f"URL: {url}\nслушаю {seconds:.0f} секунд...")
    counts: Counter[str] = Counter()
    samples: dict[str, str] = {}
    async with websockets.connect(url, max_size=None) as ws:
        await _drain(ws, seconds, counts, samples)
    _report(counts, samples)


async def probe_subscribe(symbols: list[str], kinds: list[str], seconds: float) -> None:
    adapter = BinanceUsdmAdapter()
    params = adapter.stream_names(symbols, kinds)
    url = f"{WS_BASE}/ws"
    print(f"URL: {url} (потоки НЕ в URL — подписка сообщением)\nслушаю {seconds:.0f} секунд...")
    counts: Counter[str] = Counter()
    samples: dict[str, str] = {}
    async with websockets.connect(url, max_size=None) as ws:
        sub = {"method": "SUBSCRIBE", "params": params, "id": 1}
        print(f"  → {json.dumps(sub)}")
        await ws.send(json.dumps(sub))
        await _drain(ws, min(3.0, seconds / 3), counts, samples)
        lst = {"method": "LIST_SUBSCRIPTIONS", "id": 2}
        print(f"  → {json.dumps(lst)}")
        await ws.send(json.dumps(lst))
        await _drain(ws, seconds, counts, samples)
    _report(counts, samples)


async def probe_raw(symbols: list[str], kinds: list[str], seconds: float) -> None:
    adapter = BinanceUsdmAdapter()
    sym = symbols[0]
    counts: Counter[str] = Counter()
    samples: dict[str, str] = {}
    for kind in kinds:
        stream = adapter.stream_names([sym], [kind])[0]
        url = f"{WS_BASE}/ws/{stream}"
        print(f"\nURL: {url}\nслушаю {seconds:.0f} секунд...")
        try:
            async with websockets.connect(url, max_size=None) as ws:
                await _drain(ws, seconds, counts, samples)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! подключение не удалось: {exc}")
    _report(counts, samples)


async def probe(symbols: list[str], seconds: float, kinds: list[str], mode: str) -> None:
    if mode == "url":
        await probe_url(symbols, kinds, seconds)
    elif mode == "subscribe":
        await probe_subscribe(symbols, kinds, seconds)
    else:
        await probe_raw(symbols, kinds, seconds)
    print("\n(forceOrder редкий — его отсутствие в счётчиках нормально; "
          "остальные потоки при живом рынке должны идти постоянно)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["ETHUSDT", "BTCUSDT", "SOLUSDT"])
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=WS_STREAMS,
        choices=WS_STREAMS,
        help="какие виды потоков подписать",
    )
    parser.add_argument(
        "--mode",
        default="url",
        choices=["url", "subscribe", "raw"],
        help="способ подписки (см. докстринг)",
    )
    args = parser.parse_args()
    asyncio.run(probe(args.symbols, args.seconds, args.kinds, args.mode))


if __name__ == "__main__":
    main()
