"""HTTP-сервер платформы: снапшот/дельта по символам + статичная страница.

Только стандартная библиотека: ThreadingHTTPServer, фоновый поток двигает
фиды и состояния. Дельта-протокол описан в state.py; фронтенд в static/.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

import structlog

from trading_system.platform.state import Bar, LiveMapState

log = structlog.get_logger(__name__)


class BarFeed(Protocol):
    def poll(self) -> list: ...


class Platform:
    """Состояния и фиды по символам + фоновая прокачка."""

    def __init__(self, poll_s: float = 2.0) -> None:
        self._states: dict[str, LiveMapState | None] = {}
        self._factories: dict[str, Callable[[Bar], LiveMapState]] = {}
        self._feeds: dict[str, BarFeed] = {}
        self._poll_s = poll_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add_symbol(self, state: LiveMapState, feed: BarFeed) -> None:
        self._states[state.symbol] = state
        self._feeds[state.symbol] = feed

    def add_symbol_lazy(
        self, symbol: str, feed: BarFeed, factory: Callable[[Bar], LiveMapState]
    ) -> None:
        """Состояние создаётся из ПЕРВОГО реального бара (масштаб сетки — от
        его цены), а не из угаданной заранее цены: до первых баров символ
        отвечает «прогрев» вместо карты в неправильном масштабе."""
        self._states[symbol] = None
        self._factories[symbol] = factory
        self._feeds[symbol] = feed

    @property
    def symbols(self) -> list[str]:
        return sorted(self._states)

    def state(self, symbol: str) -> LiveMapState | None:
        return self._states.get(symbol)

    def pump_once(self) -> int:
        """Один проход по фидам; возвращает число применённых баров."""
        applied = 0
        for sym, feed in self._feeds.items():
            try:
                bars = feed.poll()
            except Exception:
                log.exception("feed_poll_failed", symbol=sym)
                continue
            st = self._states[sym]
            if st is None:
                if not bars:
                    continue
                st = self._factories[sym](bars[0])
                self._states[sym] = st
                log.info("state_created", symbol=sym,
                         bucket_size=st.map.buckets.bucket_size)
            for bar in bars:
                try:
                    applied += bool(st.apply_bar(bar))
                except ValueError:
                    log.exception("bad_bar_skipped", symbol=sym)
        return applied

    def _run(self) -> None:
        while not self._stop.is_set():
            n = self.pump_once()
            if n:
                log.info("bars_applied", n=n)
            self._stop.wait(self._poll_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="platform-pump")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def _page() -> bytes:
    return (resources.files("trading_system.platform") / "static" / "index.html").read_bytes()


def make_handler(platform: Platform) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload: dict[str, Any], code: int = 200) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - контракт BaseHTTPRequestHandler
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    body = _page()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == "/api/meta":
                    self._json({
                        "symbols": platform.symbols,
                        "states": {
                            s: (st.meta() if (st := platform.state(s)) is not None
                                else {"symbol": s, "warming": True})
                            for s in platform.symbols
                        },
                        "server_time_ns": time.time_ns(),
                    })
                elif url.path == "/api/snapshot":
                    sym = q.get("symbol", "")
                    if sym not in platform.symbols:
                        self._json({"error": "unknown symbol"}, 404)
                    elif (st := platform.state(sym)) is None:
                        self._json({"type": "warming", "symbol": sym})
                    else:
                        self._json(st.snapshot())
                elif url.path == "/api/delta":
                    sym = q.get("symbol", "")
                    if sym not in platform.symbols:
                        self._json({"error": "unknown symbol"}, 404)
                    elif (st := platform.state(sym)) is None:
                        self._json({"type": "warming", "symbol": sym,
                                    "gap": False, "frames": [], "bars": [],
                                    "epoch": None, "last_ts": None})
                    else:
                        self._json(st.delta(int(q.get("since", "0")), q.get("epoch")))
                else:
                    self._json({"error": "not found"}, 404)
            except BrokenPipeError:
                pass
            except Exception:
                log.exception("request_failed", path=self.path)
                try:
                    self._json({"error": "internal"}, 500)
                except Exception:
                    pass

        def log_message(self, fmt: str, *args: Any) -> None:
            pass  # доступ-лог не нужен, структурные логи пишет платформа

    return Handler


def serve(platform: Platform, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(platform))
    log.info("platform_listening", host=host, port=port, symbols=platform.symbols)
    return httpd
