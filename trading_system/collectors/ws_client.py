"""Reconnecting websocket client: pluggable transport, deterministic backoff.

The transport is injected as an async factory so tests drive the client with
fakes and a fake clock/rng/sleep — no sockets, no wall clock, no real jitter.

Transport protocol:
    await transport.recv(timeout) -> str | bytes   # raises TimeoutError when
        no message arrives within `timeout` seconds (heartbeat watchdog),
        ConnectionError when the connection drops, TransportExhausted when a
        fake wants to end the stream cleanly.
    await transport.close()
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

import structlog

from trading_system.core.timeutils import now_ns

log = structlog.get_logger(__name__)

CONNECTED = "connected"
DISCONNECTED = "disconnected"
GAP_SUSPECTED = "gap_suspected"


class TransportExhausted(Exception):
    """Raised by a transport to end the message stream cleanly (tests, shutdown)."""


class Transport(Protocol):
    async def recv(self, timeout: float | None = None) -> str | bytes: ...

    async def close(self) -> None: ...


TransportFactory = Callable[[str], Awaitable[Transport]]


@dataclass(frozen=True, slots=True)
class ConnectionEvent:
    """Connection lifecycle event; ts is UTC ns from the injected clock."""

    kind: str  # connected | disconnected | gap_suspected
    ts: int
    attempt: int
    detail: str = ""


def backoff_delay(attempt: int, base_s: float, cap_s: float, rng: random.Random) -> float:
    """Equal-jitter exponential backoff: delay in [exp/2, exp], exp capped."""
    if attempt < 1:
        raise ValueError("attempt starts at 1")
    exp = min(cap_s, base_s * 2 ** (attempt - 1))
    return exp / 2 + rng.random() * exp / 2


@dataclass
class ReconnectingWSClient:
    """Auto-reconnecting message pump with heartbeat watchdog and gap flagging.

    messages() yields (payload, ts_recv_ns). Every unclean drop after data was
    received on a connection also emits a gap-suspicion event: downstream book
    sequencing must verify contiguity (U/u/pu) and resync if needed.
    """

    url: str
    transport_factory: TransportFactory
    heartbeat_s: float = 30.0
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 30.0
    max_reconnects: int | None = None
    rng: random.Random | None = None
    clock: Callable[[], int] = now_ns
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    on_event: Callable[[ConnectionEvent], None] | None = None
    events: list[ConnectionEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random(0)
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def _emit(self, kind: str, attempt: int, detail: str = "") -> None:
        ev = ConnectionEvent(kind=kind, ts=self.clock(), attempt=attempt, detail=detail)
        self.events.append(ev)
        if self.on_event is not None:
            self.on_event(ev)
        log.info("ws_event", url=self.url, kind=kind, attempt=attempt, detail=detail)

    async def messages(self) -> AsyncIterator[tuple[str | bytes, int]]:
        attempt = 0  # consecutive failures; reset on every successful recv
        reconnects = 0
        while not self._stopped:
            try:
                transport = await self.transport_factory(self.url)
            except TransportExhausted:
                return
            except ConnectionError as exc:
                attempt += 1
                self._emit(DISCONNECTED, attempt, f"connect failed: {exc}")
                if not await self._backoff(attempt, reconnects):
                    return
                reconnects += 1
                continue
            self._emit(CONNECTED, attempt)
            got_data = False
            try:
                while not self._stopped:
                    try:
                        payload = await transport.recv(timeout=self.heartbeat_s)
                    except TransportExhausted:
                        return
                    except TimeoutError:
                        attempt += 1
                        self._emit(GAP_SUSPECTED, attempt, "heartbeat timeout")
                        self._emit(DISCONNECTED, attempt, "heartbeat timeout")
                        break
                    except ConnectionError as exc:
                        attempt += 1
                        if got_data:
                            self._emit(GAP_SUSPECTED, attempt, f"connection dropped: {exc}")
                        self._emit(DISCONNECTED, attempt, f"connection dropped: {exc}")
                        break
                    attempt = 0
                    got_data = True
                    yield payload, self.clock()
            finally:
                await transport.close()
            if self._stopped:
                return
            if not await self._backoff(attempt, reconnects):
                return
            reconnects += 1

    async def _backoff(self, attempt: int, reconnects: int) -> bool:
        """Sleep before the next reconnect; False when max_reconnects exhausted."""
        if self.max_reconnects is not None and reconnects >= self.max_reconnects:
            return False
        assert self.rng is not None
        await self.sleep(backoff_delay(max(attempt, 1), self.backoff_base_s, self.backoff_cap_s, self.rng))
        return True


class _WebsocketsTransport:
    """Thin adapter of the `websockets` library to the Transport protocol."""

    def __init__(self, conn: object) -> None:
        self._conn = conn

    async def recv(self, timeout: float | None = None) -> str | bytes:
        try:
            return await asyncio.wait_for(self._conn.recv(), timeout)  # type: ignore[attr-defined]
        except TimeoutError:
            raise
        except Exception as exc:  # websockets.ConnectionClosed et al.
            raise ConnectionError(str(exc)) from exc

    async def close(self) -> None:
        await self._conn.close()  # type: ignore[attr-defined]


async def websockets_transport_factory(url: str) -> Transport:
    """Production transport factory (network); never used in offline tests."""
    import websockets

    try:
        conn = await websockets.connect(url, ping_interval=15, ping_timeout=10, max_queue=2**12)
    except OSError as exc:
        raise ConnectionError(str(exc)) from exc
    return _WebsocketsTransport(conn)
