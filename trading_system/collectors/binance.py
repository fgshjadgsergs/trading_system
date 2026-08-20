"""Binance USDT-M futures adapter: stream names, websocket/REST normalizers.

All parsers are pure: they take parsed JSON (or a RawMessage) plus a local
receive timestamp and return unified-schema records. Exchange millisecond
timestamps become UTC nanoseconds via core.timeutils.ms_to_ns.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

import structlog

from trading_system.collectors.ws_client import ReconnectingWSClient
from trading_system.core.adapter import ExchangeAdapter, LiquidationFormula, RawMessage
from trading_system.core.liquidation import BinanceUsdmLiquidation
from trading_system.core.schema import (
    BookSnapshot,
    DepthDiff,
    Kline,
    Liquidation,
    MarkPrice,
    OpenInterest,
    RatioMetric,
    RatioPoint,
    Record,
    Side,
    Trade,
)
from trading_system.core.timeutils import ms_to_ns, now_ns

log = structlog.get_logger(__name__)

EXCHANGE = "binance_usdm"
WS_BASE = "wss://fstream.binance.com"
REST_BASE = "https://fapi.binance.com"

HttpGet = Callable[[str, dict[str, Any]], Awaitable[Any]]


# --------------------------------------------------------------------------- #
# stream names
# --------------------------------------------------------------------------- #
def depth_stream(symbol: str) -> str:
    return f"{symbol.lower()}@depth@100ms"


def agg_trade_stream(symbol: str) -> str:
    return f"{symbol.lower()}@aggTrade"


def force_order_stream(symbol: str) -> str:
    return f"{symbol.lower()}@forceOrder"


def mark_price_stream(symbol: str) -> str:
    return f"{symbol.lower()}@markPrice@1s"


def kline_1m_stream(symbol: str) -> str:
    """1m klines are consumed only as a checksum of our own tick-built bars."""
    return f"{symbol.lower()}@kline_1m"


STREAM_BUILDERS: dict[str, Callable[[str], str]] = {
    "depth": depth_stream,
    "agg_trade": agg_trade_stream,
    "force_order": force_order_stream,
    "mark_price": mark_price_stream,
    "kline_1m": kline_1m_stream,
}


def combined_ws_url(streams: Sequence[str], ws_base: str = WS_BASE) -> str:
    """Combined-stream websocket URL: wss://.../stream?streams=a/b/c."""
    if not streams:
        raise ValueError("at least one stream required")
    return f"{ws_base}/stream?streams={'/'.join(streams)}"


# --------------------------------------------------------------------------- #
# websocket message normalizers (pure)
# --------------------------------------------------------------------------- #
def parse_agg_trade(msg: dict, ts_recv: int) -> Trade:
    """aggTrade: field m=true means the buyer is the maker => taker side SELL."""
    price = float(msg["p"])
    qty = float(msg["q"])
    return Trade(
        exchange=EXCHANGE,
        symbol=msg["s"],
        ts_event=ms_to_ns(msg["T"]),
        ts_recv=ts_recv,
        price=price,
        qty=qty,
        qty_usd=price * qty,
        side=Side.SELL if msg["m"] else Side.BUY,
        trade_id=int(msg["a"]),
    )


def _levels(raw: Sequence[Sequence[str]]) -> tuple[tuple[float, float], ...]:
    return tuple((float(p), float(q)) for p, q in raw)


def parse_depth_update(msg: dict, ts_recv: int) -> DepthDiff:
    """depthUpdate with futures sequence fields U/u/pu; T is transaction time."""
    return DepthDiff(
        exchange=EXCHANGE,
        symbol=msg["s"],
        ts_event=ms_to_ns(msg.get("T", msg["E"])),
        ts_recv=ts_recv,
        first_update_id=int(msg["U"]),
        final_update_id=int(msg["u"]),
        prev_final_update_id=int(msg["pu"]),
        bids=_levels(msg["b"]),
        asks=_levels(msg["a"]),
    )


def parse_force_order(msg: dict, ts_recv: int) -> Liquidation:
    """forceOrder: o.S == SELL means a long position was liquidated.

    Note: the exchange emits at most one forceOrder per symbol per second, so
    this stream is a sample of liquidations, not the full set.
    """
    o = msg["o"]
    price = float(o.get("ap") or o["p"])
    qty = float(o["q"])
    return Liquidation(
        exchange=EXCHANGE,
        symbol=o["s"],
        ts_event=ms_to_ns(o.get("T", msg["E"])),
        ts_recv=ts_recv,
        price=price,
        qty=qty,
        qty_usd=price * qty,
        side=Side.SELL if o["S"] == "SELL" else Side.BUY,
    )


def parse_mark_price(msg: dict, ts_recv: int) -> MarkPrice:
    return MarkPrice(
        exchange=EXCHANGE,
        symbol=msg["s"],
        ts_event=ms_to_ns(msg["E"]),
        ts_recv=ts_recv,
        mark_price=float(msg["p"]),
        index_price=float(msg["i"]),
        funding_rate=float(msg["r"]),
        next_funding_ts=ms_to_ns(msg["T"]),
    )


def parse_kline(msg: dict) -> Kline:
    """kline event; closed flag from k.x. Used only to checksum our own bars."""
    k = msg["k"]
    return Kline(
        exchange=EXCHANGE,
        symbol=k["s"],
        ts_open=ms_to_ns(k["t"]),
        ts_close=ms_to_ns(k["T"]),
        open=float(k["o"]),
        high=float(k["h"]),
        low=float(k["l"]),
        close=float(k["c"]),
        volume=float(k["v"]),
        quote_volume=float(k["q"]),
        taker_buy_volume=float(k["V"]),
        taker_buy_quote_volume=float(k["Q"]),
        n_trades=int(k["n"]),
        closed=bool(k["x"]),
    )


# --------------------------------------------------------------------------- #
# REST payload normalizers (pure)
# --------------------------------------------------------------------------- #
def parse_depth_snapshot(payload: dict, symbol: str, ts_recv: int) -> BookSnapshot:
    """/fapi/v1/depth -> BookSnapshot (the response carries no symbol field)."""
    ts_ms = payload.get("T") or payload.get("E")
    return BookSnapshot(
        exchange=EXCHANGE,
        symbol=symbol,
        ts_event=ms_to_ns(ts_ms) if ts_ms is not None else ts_recv,
        ts_recv=ts_recv,
        last_update_id=int(payload["lastUpdateId"]),
        bids=tuple(sorted(_levels(payload["bids"]), key=lambda x: -x[0])),
        asks=tuple(sorted(_levels(payload["asks"]), key=lambda x: x[0])),
    )


def parse_open_interest(payload: dict, ts_recv: int, price: float | None = None) -> OpenInterest:
    """/fapi/v1/openInterest. USD notional needs a price the endpoint lacks;
    pass the latest mark price, else open_interest_usd is NaN (unknown)."""
    oi = float(payload["openInterest"])
    return OpenInterest(
        exchange=EXCHANGE,
        symbol=payload["symbol"],
        ts_event=ms_to_ns(payload["time"]),
        ts_recv=ts_recv,
        open_interest=oi,
        open_interest_usd=oi * price if price is not None else float("nan"),
    )


def _parse_ratio_points(
    payload: list[dict] | dict, metric: RatioMetric, ts_recv: int, symbol: str | None
) -> list[RatioPoint]:
    items = payload if isinstance(payload, list) else [payload]
    out: list[RatioPoint] = []
    for it in items:
        if "buySellRatio" in it:  # takerlongshortRatio shape (no symbol field)
            buy, sell = float(it["buyVol"]), float(it["sellVol"])
            total = buy + sell
            long_share = buy / total if total > 0 else 0.5
            short_share = sell / total if total > 0 else 0.5
            ratio = float(it["buySellRatio"])
        else:
            long_share = float(it["longAccount"])
            short_share = float(it["shortAccount"])
            ratio = float(it["longShortRatio"])
        sym = it.get("symbol", symbol)
        if sym is None:
            raise ValueError(f"{metric}: payload has no symbol and none was given")
        out.append(
            RatioPoint(
                exchange=EXCHANGE,
                symbol=sym,
                ts_event=ms_to_ns(it["timestamp"]),
                ts_recv=ts_recv,
                metric=str(metric.value),
                long_share=long_share,
                short_share=short_share,
                ratio=ratio,
            )
        )
    return out


def parse_global_ls_account(
    payload: list[dict] | dict, ts_recv: int, symbol: str | None = None
) -> list[RatioPoint]:
    """/futures/data/globalLongShortAccountRatio -> RatioPoint list."""
    return _parse_ratio_points(payload, RatioMetric.GLOBAL_LS_ACCOUNT, ts_recv, symbol)


def parse_top_ls_position(
    payload: list[dict] | dict, ts_recv: int, symbol: str | None = None
) -> list[RatioPoint]:
    """/futures/data/topLongShortPositionRatio -> RatioPoint list."""
    return _parse_ratio_points(payload, RatioMetric.TOP_LS_POSITION, ts_recv, symbol)


def parse_taker_ls(
    payload: list[dict] | dict, ts_recv: int, symbol: str | None = None
) -> list[RatioPoint]:
    """/futures/data/takerlongshortRatio -> RatioPoint list (shares from volumes)."""
    return _parse_ratio_points(payload, RatioMetric.TAKER_LS, ts_recv, symbol)


# --------------------------------------------------------------------------- #
# adapter
# --------------------------------------------------------------------------- #
class BinanceUsdmAdapter(ExchangeAdapter):
    """Binance USDT-M perpetuals adapter over the unified schema.

    Transports are injected: `http_get(url, params) -> parsed JSON` for REST,
    `ws_connect(url) -> Transport` for websockets. Tests inject fakes; no code
    path here opens a network connection on its own.
    """

    name = EXCHANGE

    def __init__(
        self,
        *,
        http_get: HttpGet | None = None,
        ws_connect: Any | None = None,
        clock: Callable[[], int] = now_ns,
        rest_base: str = REST_BASE,
        ws_base: str = WS_BASE,
        liq: LiquidationFormula | None = None,
        ws_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._http_get = http_get
        self._ws_connect = ws_connect
        self._clock = clock
        self._rest_base = rest_base
        self._ws_base = ws_base
        self._liq = liq or BinanceUsdmLiquidation()
        self._ws_client_kwargs = ws_client_kwargs or {}

    def stream_names(self, symbols: Sequence[str], streams: Sequence[str]) -> list[str]:
        """Exchange-native stream tags for logical stream keys (STREAM_BUILDERS)."""
        names: list[str] = []
        for s in streams:
            builder = STREAM_BUILDERS.get(s)
            if builder is None:
                raise KeyError(f"unknown stream key {s!r}; known: {sorted(STREAM_BUILDERS)}")
            names.extend(builder(sym) for sym in symbols)
        return names

    async def subscribe(
        self,
        symbols: Sequence[str],
        streams: Sequence[str],
        *,
        ws_client_kwargs: dict[str, Any] | None = None,
    ) -> AsyncIterator[RawMessage]:
        """Yield raw combined-stream messages; reconnects handled inside.

        ws_client_kwargs override the adapter-level defaults per call — e.g. a
        long heartbeat for legitimately quiet streams like forceOrder.
        """
        if self._ws_connect is None:
            raise ValueError("ws_connect transport factory required for subscribe()")
        url = combined_ws_url(self.stream_names(symbols, streams), self._ws_base)
        client = ReconnectingWSClient(
            url,
            self._ws_connect,
            clock=self._clock,
            **{**self._ws_client_kwargs, **(ws_client_kwargs or {})},
        )
        async for payload, ts_recv in client.messages():
            text = payload.decode() if isinstance(payload, bytes) else payload
            tag = ""
            try:
                tag = json.loads(text).get("stream", "")
            except (json.JSONDecodeError, AttributeError):
                pass
            yield RawMessage(stream=tag, payload=payload, ts_recv=ts_recv)

    def normalize(self, raw: RawMessage) -> list[Record]:
        """One raw ws message -> zero or more unified records."""
        text = raw.payload.decode() if isinstance(raw.payload, bytes) else raw.payload
        msg = json.loads(text)
        if isinstance(msg, dict) and "stream" in msg and "data" in msg:
            msg = msg["data"]
        if not isinstance(msg, dict):
            return []
        event = msg.get("e")
        ts = raw.ts_recv
        if event == "aggTrade":
            return [parse_agg_trade(msg, ts)]
        if event == "depthUpdate":
            return [parse_depth_update(msg, ts)]
        if event == "forceOrder":
            return [parse_force_order(msg, ts)]
        if event == "markPriceUpdate":
            return [parse_mark_price(msg, ts)]
        if event == "kline":
            return [parse_kline(msg)]
        if event is not None:
            log.warning("unknown_binance_event", event_type=event, stream=raw.stream)
        return []

    async def snapshot(self, symbol: str, depth: int = 1000) -> BookSnapshot:
        """REST order book snapshot via the injected http_get transport."""
        if self._http_get is None:
            raise ValueError("http_get transport required for snapshot()")
        payload = await self._http_get(
            f"{self._rest_base}/fapi/v1/depth", {"symbol": symbol, "limit": depth}
        )
        return parse_depth_snapshot(payload, symbol=symbol, ts_recv=self._clock())

    def liq_formula(self) -> LiquidationFormula:
        return self._liq
