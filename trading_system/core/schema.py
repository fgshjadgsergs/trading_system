"""Unified multi-exchange data schema.

Every record is keyed by (exchange, symbol, ts_event, ts_recv); timestamps are
UTC nanoseconds; volumes are carried both in coins (qty) and USD (qty_usd) so
contracts of different exchanges normalize to the same units.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import polars as pl

PriceLevel = tuple[float, float]  # (price, qty in coins)


class Side(enum.IntEnum):
    """Taker side for trades; order side for liquidations."""

    BUY = 1
    SELL = -1


class RatioMetric(enum.StrEnum):
    GLOBAL_LS_ACCOUNT = "global_ls_account"
    TOP_LS_POSITION = "top_ls_position"
    TAKER_LS = "taker_ls"


@dataclass(frozen=True, slots=True)
class Trade:
    exchange: str
    symbol: str
    ts_event: int
    ts_recv: int
    price: float
    qty: float
    qty_usd: float
    side: Side  # taker side
    trade_id: int


@dataclass(frozen=True, slots=True)
class DepthDiff:
    """Incremental L2 update. Binance: U/u/pu sequence fields."""

    exchange: str
    symbol: str
    ts_event: int
    ts_recv: int
    first_update_id: int  # U
    final_update_id: int  # u
    prev_final_update_id: int  # pu
    bids: tuple[PriceLevel, ...]  # absolute quantities; qty == 0 removes level
    asks: tuple[PriceLevel, ...]


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    exchange: str
    symbol: str
    ts_event: int
    ts_recv: int
    last_update_id: int
    bids: tuple[PriceLevel, ...]  # sorted best (highest) first
    asks: tuple[PriceLevel, ...]  # sorted best (lowest) first


@dataclass(frozen=True, slots=True)
class Liquidation:
    exchange: str
    symbol: str
    ts_event: int
    ts_recv: int
    price: float
    qty: float
    qty_usd: float
    side: Side  # order side: SELL == long position liquidated

    @property
    def liquidated_long(self) -> bool:
        return self.side is Side.SELL


@dataclass(frozen=True, slots=True)
class MarkPrice:
    exchange: str
    symbol: str
    ts_event: int
    ts_recv: int
    mark_price: float
    index_price: float
    funding_rate: float
    next_funding_ts: int


@dataclass(frozen=True, slots=True)
class OpenInterest:
    exchange: str
    symbol: str
    ts_event: int
    ts_recv: int
    open_interest: float  # coins
    open_interest_usd: float


@dataclass(frozen=True, slots=True)
class RatioPoint:
    exchange: str
    symbol: str
    ts_event: int
    ts_recv: int
    metric: str  # RatioMetric value
    long_share: float
    short_share: float
    ratio: float


@dataclass(frozen=True, slots=True)
class Kline:
    exchange: str
    symbol: str
    ts_open: int
    ts_close: int
    open: float
    high: float
    low: float
    close: float
    volume: float  # coins
    quote_volume: float  # USD
    taker_buy_volume: float
    taker_buy_quote_volume: float
    n_trades: int
    closed: bool = field(default=True)


Record = Trade | DepthDiff | BookSnapshot | Liquidation | MarkPrice | OpenInterest | RatioPoint | Kline

_KEY = {
    "exchange": pl.Utf8,
    "symbol": pl.Utf8,
    "ts_event": pl.Int64,
    "ts_recv": pl.Int64,
}
_LEVELS = pl.List(pl.Struct({"price": pl.Float64, "qty": pl.Float64}))

POLARS_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "trade": {
        **_KEY,
        "price": pl.Float64,
        "qty": pl.Float64,
        "qty_usd": pl.Float64,
        "side": pl.Int8,
        "trade_id": pl.Int64,
    },
    "depth_diff": {
        **_KEY,
        "first_update_id": pl.Int64,
        "final_update_id": pl.Int64,
        "prev_final_update_id": pl.Int64,
        "bids": _LEVELS,
        "asks": _LEVELS,
    },
    "book_snapshot": {
        **_KEY,
        "last_update_id": pl.Int64,
        "bids": _LEVELS,
        "asks": _LEVELS,
    },
    "liquidation": {
        **_KEY,
        "price": pl.Float64,
        "qty": pl.Float64,
        "qty_usd": pl.Float64,
        "side": pl.Int8,
    },
    "mark_price": {
        **_KEY,
        "mark_price": pl.Float64,
        "index_price": pl.Float64,
        "funding_rate": pl.Float64,
        "next_funding_ts": pl.Int64,
    },
    "open_interest": {
        **_KEY,
        "open_interest": pl.Float64,
        "open_interest_usd": pl.Float64,
    },
    "ratio": {
        **_KEY,
        "metric": pl.Utf8,
        "long_share": pl.Float64,
        "short_share": pl.Float64,
        "ratio": pl.Float64,
    },
    "kline": {
        "exchange": pl.Utf8,
        "symbol": pl.Utf8,
        "ts_open": pl.Int64,
        "ts_close": pl.Int64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "quote_volume": pl.Float64,
        "taker_buy_volume": pl.Float64,
        "taker_buy_quote_volume": pl.Float64,
        "n_trades": pl.Int64,
        "closed": pl.Boolean,
    },
}

STREAM_OF_TYPE: dict[type, str] = {
    Trade: "trade",
    DepthDiff: "depth_diff",
    BookSnapshot: "book_snapshot",
    Liquidation: "liquidation",
    MarkPrice: "mark_price",
    OpenInterest: "open_interest",
    RatioPoint: "ratio",
    Kline: "kline",
}


def _levels_to_dicts(levels: tuple[PriceLevel, ...]) -> list[dict[str, float]]:
    return [{"price": float(p), "qty": float(q)} for p, q in levels]


def record_to_row(rec: Record) -> dict:
    """Dataclass record -> row dict matching POLARS_SCHEMAS of its stream."""
    stream = STREAM_OF_TYPE[type(rec)]
    row: dict = {}
    for name in POLARS_SCHEMAS[stream]:
        v = getattr(rec, name)
        if name in ("bids", "asks"):
            v = _levels_to_dicts(v)
        elif isinstance(v, Side):
            v = int(v)
        row[name] = v
    return row


def records_to_frame(records: list[Record], stream: str) -> pl.DataFrame:
    schema = POLARS_SCHEMAS[stream]
    rows = [record_to_row(r) for r in records]
    return pl.DataFrame(rows, schema=schema, orient="row") if rows else pl.DataFrame(schema=schema)
