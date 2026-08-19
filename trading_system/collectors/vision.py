"""data.binance.vision historical archives for USDT-M perpetuals (checklist 1.2).

Covers: URL/plan builders for the daily/monthly zip layout, a checksum-verified
downloader with an injectable ``fetch``, normalizers from every archive CSV kind
into the unified polars schemas of :mod:`trading_system.core.schema`, an
own-recording vs Vision reconciliation, a dataset catalog over the parquet lake
and seaborn demo reports. Everything is offline-testable: the only network
touchpoint is the ``fetch(url) -> bytes`` callable supplied by the caller.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import zipfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import structlog

from trading_system.core.io import write_batch
from trading_system.core.schema import POLARS_SCHEMAS
from trading_system.core.timeutils import NS_PER_MIN

log = structlog.get_logger(__name__)

VISION_BASE = "https://data.binance.vision"
EXCHANGE = "binance_usdm"

FetchFn = Callable[[str], bytes]

# ---------------------------------------------------------------------------
# URL / path builders
# ---------------------------------------------------------------------------

KINDS: tuple[str, ...] = (
    "aggTrades",
    "trades",
    "klines",
    "bookTicker",
    "bookDepth",
    "metrics",
    "liquidationSnapshot",
    "fundingRate",
    "premiumIndexKlines",
)
# Kinds whose path/filename carry a kline interval subdirectory.
INTERVAL_KINDS: frozenset[str] = frozenset({"klines", "premiumIndexKlines"})
# Periods each kind is actually published under on data.binance.vision.
KIND_PERIODS: dict[str, tuple[str, ...]] = {
    "aggTrades": ("daily", "monthly"),
    "trades": ("daily", "monthly"),
    "klines": ("daily", "monthly"),
    "bookTicker": ("daily", "monthly"),
    "bookDepth": ("daily",),
    "metrics": ("daily",),
    "liquidationSnapshot": ("daily",),
    "fundingRate": ("monthly",),
    "premiumIndexKlines": ("daily", "monthly"),
}

# Streams produced by normalize_csv per kind (unified + local schemas).
KIND_STREAMS: dict[str, tuple[str, ...]] = {
    "aggTrades": ("trade",),
    "trades": ("trade",),
    "klines": ("kline",),
    "premiumIndexKlines": ("premium_index_kline",),
    "liquidationSnapshot": ("liquidation",),
    "metrics": ("open_interest", "ratio"),
    "fundingRate": ("mark_price",),
    "bookTicker": ("book_ticker",),
    "bookDepth": ("book_depth",),
}

# Local schemas for streams the shared core schema does not define.
BOOK_TICKER_SCHEMA: dict[str, pl.DataType] = {
    "exchange": pl.Utf8,
    "symbol": pl.Utf8,
    "ts_event": pl.Int64,
    "ts_recv": pl.Int64,
    "update_id": pl.Int64,
    "bid_price": pl.Float64,
    "bid_qty": pl.Float64,
    "ask_price": pl.Float64,
    "ask_qty": pl.Float64,
}
BOOK_DEPTH_SCHEMA: dict[str, pl.DataType] = {
    "exchange": pl.Utf8,
    "symbol": pl.Utf8,
    "ts_event": pl.Int64,
    "ts_recv": pl.Int64,
    "percentage": pl.Float64,
    "depth": pl.Float64,
    "notional": pl.Float64,
}
LOCAL_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "book_ticker": BOOK_TICKER_SCHEMA,
    "book_depth": BOOK_DEPTH_SCHEMA,
    "premium_index_kline": POLARS_SCHEMAS["kline"],
}


def stream_schema(stream: str) -> dict[str, pl.DataType]:
    """Schema for a stream: unified core schema first, local extensions second."""
    if stream in POLARS_SCHEMAS:
        return POLARS_SCHEMAS[stream]
    return LOCAL_SCHEMAS[stream]


def _validate(kind: str, period: str, interval: str | None) -> None:
    if kind not in KIND_PERIODS:
        raise ValueError(f"unknown Vision kind: {kind!r}")
    if period not in KIND_PERIODS[kind]:
        raise ValueError(f"{kind} is not published {period}; available: {KIND_PERIODS[kind]}")
    if kind in INTERVAL_KINDS and not interval:
        raise ValueError(f"{kind} requires an interval (e.g. '1m')")


def archive_name(
    kind: str, symbol: str, date: dt.date, period: str, interval: str | None = None
) -> str:
    """Zip file name, e.g. BTCUSDT-aggTrades-2024-01-15.zip / BTCUSDT-1m-2024-01.zip."""
    _validate(kind, period, interval)
    tag = date.strftime("%Y-%m-%d") if period == "daily" else date.strftime("%Y-%m")
    mid = interval if kind in INTERVAL_KINDS else kind
    return f"{symbol}-{mid}-{tag}.zip"


def vision_path(
    kind: str, symbol: str, date: dt.date, period: str, interval: str | None = None
) -> str:
    """Server-relative path under data.binance.vision for one archive."""
    _validate(kind, period, interval)
    parts = ["data", "futures", "um", period, kind, symbol]
    if kind in INTERVAL_KINDS:
        parts.append(str(interval))
    parts.append(archive_name(kind, symbol, date, period, interval))
    return "/".join(parts)


def vision_url(
    kind: str,
    symbol: str,
    date: dt.date,
    period: str,
    interval: str | None = None,
    base: str = VISION_BASE,
) -> str:
    return f"{base}/{vision_path(kind, symbol, date, period, interval)}"


def checksum_url(
    kind: str,
    symbol: str,
    date: dt.date,
    period: str,
    interval: str | None = None,
    base: str = VISION_BASE,
) -> str:
    return vision_url(kind, symbol, date, period, interval, base) + ".CHECKSUM"


@dataclass(frozen=True, slots=True)
class DownloadItem:
    """One archive to fetch: URLs plus the mirror-relative local path."""

    symbol: str
    kind: str
    period: str
    date: dt.date  # first day of month for monthly items
    interval: str | None
    url: str
    checksum_url: str
    rel_path: str


def _iter_days(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def _iter_months(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    d = start.replace(day=1)
    while d <= end:
        yield d
        d = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)


def plan_downloads(
    symbols: Sequence[str],
    kinds: Sequence[str],
    start: dt.date,
    end: dt.date,
    period: str = "daily",
    intervals: Sequence[str] = ("1m",),
    base: str = VISION_BASE,
) -> list[DownloadItem]:
    """Full download list for symbols x kinds x date range.

    Kinds not published under the requested period fall back to their only
    available period (fundingRate is monthly-only; metrics, bookDepth and
    liquidationSnapshot are daily-only).
    """
    if end < start:
        raise ValueError("end date before start date")
    items: list[DownloadItem] = []
    for symbol in symbols:
        for kind in kinds:
            if kind not in KIND_PERIODS:
                raise ValueError(f"unknown Vision kind: {kind!r}")
            eff_period = period if period in KIND_PERIODS[kind] else KIND_PERIODS[kind][0]
            dates = _iter_days(start, end) if eff_period == "daily" else _iter_months(start, end)
            kind_intervals: Sequence[str | None] = (
                list(intervals) if kind in INTERVAL_KINDS else [None]
            )
            for date in dates:
                for interval in kind_intervals:
                    rel = vision_path(kind, symbol, date, eff_period, interval)
                    items.append(
                        DownloadItem(
                            symbol=symbol,
                            kind=kind,
                            period=eff_period,
                            date=date,
                            interval=interval,
                            url=f"{base}/{rel}",
                            checksum_url=f"{base}/{rel}.CHECKSUM",
                            rel_path=rel,
                        )
                    )
    return items


# ---------------------------------------------------------------------------
# Downloader with sha256 verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DownloadResult:
    item: DownloadItem
    path: Path | None
    status: str  # downloaded | skipped_existing | checksum_mismatch | fetch_error
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in ("downloaded", "skipped_existing")


def parse_checksum(data: bytes | str) -> str | None:
    """Extract the sha256 hex digest from a .CHECKSUM file (sha256sum format)."""
    text = data.decode("utf-8", "replace") if isinstance(data, bytes) else data
    for line in text.splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if len(token) == 64 and all(c in "0123456789abcdefABCDEF" for c in token):
            return token.lower()
    return None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(
    items: Sequence[DownloadItem],
    dest_root: Path,
    fetch: FetchFn,
    *,
    verify: bool = True,
    skip_existing: bool = True,
) -> list[DownloadResult]:
    """Fetch archives into ``dest_root`` mirroring the Vision path layout.

    A corrupt or unparsable checksum yields a ``checksum_mismatch`` error
    record and the archive is not written; existing files are skipped.
    """
    results: list[DownloadResult] = []
    for item in items:
        dest = dest_root / item.rel_path
        if skip_existing and dest.exists():
            results.append(DownloadResult(item, dest, "skipped_existing"))
            continue
        try:
            blob = fetch(item.url)
        except Exception as exc:  # noqa: BLE001 - injectable transport, any failure is data
            log.warning("vision_fetch_failed", url=item.url, error=str(exc))
            results.append(DownloadResult(item, None, "fetch_error", str(exc)))
            continue
        if verify:
            try:
                checksum_blob = fetch(item.checksum_url)
            except Exception as exc:  # noqa: BLE001
                log.warning("vision_checksum_fetch_failed", url=item.checksum_url, error=str(exc))
                results.append(DownloadResult(item, None, "fetch_error", f"checksum: {exc}"))
                continue
            expected = parse_checksum(checksum_blob)
            actual = sha256_hex(blob)
            if expected is None or actual != expected:
                log.error(
                    "vision_checksum_mismatch",
                    url=item.url,
                    expected=expected,
                    actual=actual,
                )
                results.append(
                    DownloadResult(
                        item, None, "checksum_mismatch", f"expected={expected} actual={actual}"
                    )
                )
                continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        results.append(DownloadResult(item, dest, "downloaded"))
    return results


# ---------------------------------------------------------------------------
# CSV normalizers -> unified schema frames
# ---------------------------------------------------------------------------

_CSV_COLUMNS: dict[str, tuple[str, ...]] = {
    "aggTrades": (
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "transact_time",
        "is_buyer_maker",
    ),
    "trades": ("id", "price", "qty", "quote_qty", "time", "is_buyer_maker"),
    "klines": (
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ),
    "bookTicker": (
        "update_id",
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
        "transaction_time",
        "event_time",
    ),
    "bookDepth": ("timestamp", "percentage", "depth", "notional"),
    "metrics": (
        "create_time",
        "symbol",
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ),
    "liquidationSnapshot": (
        "time",
        "symbol",
        "side",
        "order_type",
        "time_in_force",
        "original_quantity",
        "price",
        "average_price",
        "order_status",
        "last_fill_quantity",
        "accumulated_fill_quantity",
    ),
    "fundingRate": ("calc_time", "funding_interval_hours", "last_funding_rate"),
}
_CSV_COLUMNS["premiumIndexKlines"] = _CSV_COLUMNS["klines"]

# Column index probed to decide whether row 0 is a header (must be numeric in data).
_NUMERIC_PROBE: dict[str, int] = {
    "aggTrades": 1,
    "trades": 1,
    "klines": 0,
    "premiumIndexKlines": 0,
    "bookTicker": 0,
    "bookDepth": 2,
    "metrics": 2,
    "liquidationSnapshot": 0,
    "fundingRate": 0,
}


def _has_header(data: bytes, probe_idx: int) -> bool:
    first_line = data.split(b"\n", 1)[0].decode("utf-8", "replace").strip()
    tokens = [t.strip() for t in first_line.split(",")]
    if probe_idx >= len(tokens):
        return False
    try:
        float(tokens[probe_idx])
    except ValueError:
        return True
    return False


def _read_csv(data: bytes, kind: str) -> pl.DataFrame:
    """Archive CSV bytes -> frame with canonical column names (header optional)."""
    columns = _CSV_COLUMNS[kind]
    if not data.strip():
        return pl.DataFrame(schema=dict.fromkeys(columns, pl.Utf8))
    has_header = _has_header(data, _NUMERIC_PROBE[kind])
    df = pl.read_csv(io.BytesIO(data), has_header=has_header, infer_schema_length=10_000)
    if df.width < len([c for c in columns if c != "ignore"]):
        raise ValueError(f"{kind} CSV has {df.width} columns, expected >= {len(columns) - 1}")
    renames = dict(zip(df.columns, columns, strict=False))
    return df.rename(renames)


def _epoch_to_ns(col: str) -> pl.Expr:
    """Epoch of unknown unit (s/ms/us/ns) -> UTC nanoseconds, integer math."""
    v = pl.col(col).cast(pl.Int64, strict=False)
    return (
        pl.when(v < 100_000_000_000)
        .then(v * 1_000_000_000)
        .when(v < 100_000_000_000_000)
        .then(v * 1_000_000)
        .when(v < 100_000_000_000_000_000)
        .then(v * 1_000)
        .otherwise(v)
    )


def _time_str_or_epoch_to_ns(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Timestamp column that is either an epoch or 'YYYY-MM-DD HH:MM:SS' -> ns."""
    if df.schema[col] in (pl.Utf8, pl.String):
        sample = df.get_column(col).drop_nulls()
        is_epoch = False
        if len(sample):
            try:
                float(sample[0])
                is_epoch = True
            except ValueError:
                is_epoch = False
        if not is_epoch:
            return df.with_columns(
                pl.col(col)
                .str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC")
                .dt.epoch("ns")
                .alias(col)
            )
    return df.with_columns(_epoch_to_ns(col).alias(col))


def _bool_expr(col: str, dtype: pl.DataType) -> pl.Expr:
    if dtype == pl.Boolean:
        return pl.col(col)
    return pl.col(col).cast(pl.Utf8).str.to_lowercase().is_in(["true", "1"])


def _conform(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return df.select([pl.col(n).cast(t) for n, t in schema.items()])


def _normalize_agg_trades(df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    df = df.with_columns(_epoch_to_ns("transact_time").alias("ts_event"))
    out = df.select(
        exchange=pl.lit(EXCHANGE),
        symbol=pl.lit(symbol),
        ts_event=pl.col("ts_event"),
        ts_recv=pl.col("ts_event"),
        price=pl.col("price").cast(pl.Float64),
        qty=pl.col("quantity").cast(pl.Float64),
        qty_usd=pl.col("price").cast(pl.Float64) * pl.col("quantity").cast(pl.Float64),
        # buyer is maker => taker sold
        side=pl.when(_bool_expr("is_buyer_maker", df.schema["is_buyer_maker"]))
        .then(pl.lit(-1))
        .otherwise(pl.lit(1)),
        trade_id=pl.col("agg_trade_id").cast(pl.Int64),
    )
    return _conform(out.sort("ts_event"), POLARS_SCHEMAS["trade"])


def _normalize_trades(df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    df = df.with_columns(_epoch_to_ns("time").alias("ts_event"))
    out = df.select(
        exchange=pl.lit(EXCHANGE),
        symbol=pl.lit(symbol),
        ts_event=pl.col("ts_event"),
        ts_recv=pl.col("ts_event"),
        price=pl.col("price").cast(pl.Float64),
        qty=pl.col("qty").cast(pl.Float64),
        qty_usd=pl.col("price").cast(pl.Float64) * pl.col("qty").cast(pl.Float64),
        side=pl.when(_bool_expr("is_buyer_maker", df.schema["is_buyer_maker"]))
        .then(pl.lit(-1))
        .otherwise(pl.lit(1)),
        trade_id=pl.col("id").cast(pl.Int64),
    )
    return _conform(out.sort("ts_event"), POLARS_SCHEMAS["trade"])


def _normalize_klines(df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    out = df.select(
        exchange=pl.lit(EXCHANGE),
        symbol=pl.lit(symbol),
        ts_open=_epoch_to_ns("open_time"),
        ts_close=_epoch_to_ns("close_time"),
        open=pl.col("open").cast(pl.Float64),
        high=pl.col("high").cast(pl.Float64),
        low=pl.col("low").cast(pl.Float64),
        close=pl.col("close").cast(pl.Float64),
        volume=pl.col("volume").cast(pl.Float64),
        quote_volume=pl.col("quote_volume").cast(pl.Float64),
        taker_buy_volume=pl.col("taker_buy_volume").cast(pl.Float64),
        taker_buy_quote_volume=pl.col("taker_buy_quote_volume").cast(pl.Float64),
        n_trades=pl.col("count").cast(pl.Int64),
        closed=pl.lit(True),
    )
    return _conform(out.sort("ts_open"), POLARS_SCHEMAS["kline"])


def _normalize_liquidations(df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    df = df.with_columns(_epoch_to_ns("time").alias("ts_event"))
    price = (
        pl.when(pl.col("average_price").cast(pl.Float64) > 0)
        .then(pl.col("average_price").cast(pl.Float64))
        .otherwise(pl.col("price").cast(pl.Float64))
    )
    qty = (
        pl.when(pl.col("accumulated_fill_quantity").cast(pl.Float64) > 0)
        .then(pl.col("accumulated_fill_quantity").cast(pl.Float64))
        .otherwise(pl.col("original_quantity").cast(pl.Float64))
    )
    out = df.select(
        exchange=pl.lit(EXCHANGE),
        symbol=pl.lit(symbol),
        ts_event=pl.col("ts_event"),
        ts_recv=pl.col("ts_event"),
        price=price,
        qty=qty,
        qty_usd=price * qty,
        side=pl.when(pl.col("side").cast(pl.Utf8).str.to_uppercase() == "BUY")
        .then(pl.lit(1))
        .otherwise(pl.lit(-1)),
    )
    return _conform(out.sort("ts_event"), POLARS_SCHEMAS["liquidation"])


_RATIO_SOURCES: tuple[tuple[str, str], ...] = (
    ("global_ls_account", "count_long_short_ratio"),
    ("top_ls_position", "sum_toptrader_long_short_ratio"),
    ("taker_ls", "sum_taker_long_short_vol_ratio"),
)


def _normalize_metrics(df: pl.DataFrame, symbol: str) -> dict[str, pl.DataFrame]:
    df = _time_str_or_epoch_to_ns(df, "create_time")
    oi = df.select(
        exchange=pl.lit(EXCHANGE),
        symbol=pl.lit(symbol),
        ts_event=pl.col("create_time"),
        ts_recv=pl.col("create_time"),
        open_interest=pl.col("sum_open_interest").cast(pl.Float64, strict=False),
        open_interest_usd=pl.col("sum_open_interest_value").cast(pl.Float64, strict=False),
    )
    ratio_parts: list[pl.DataFrame] = []
    for metric, source in _RATIO_SOURCES:
        r = pl.col(source).cast(pl.Float64, strict=False)
        part = (
            df.filter(r.is_not_null())
            .select(
                exchange=pl.lit(EXCHANGE),
                symbol=pl.lit(symbol),
                ts_event=pl.col("create_time"),
                ts_recv=pl.col("create_time"),
                metric=pl.lit(metric),
                long_share=r / (1.0 + r),
                short_share=1.0 / (1.0 + r),
                ratio=r,
            )
        )
        ratio_parts.append(_conform(part, POLARS_SCHEMAS["ratio"]))
    ratios = pl.concat(ratio_parts).sort("ts_event", "metric")
    return {
        "open_interest": _conform(oi.sort("ts_event"), POLARS_SCHEMAS["open_interest"]),
        "ratio": ratios,
    }


def _normalize_funding(df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """fundingRate -> mark_price rows: only funding_rate is known, mark/index are NaN.

    next_funding_ts = ts_event + funding_interval_hours (default 8h).
    """
    df = df.with_columns(_epoch_to_ns("calc_time").alias("ts_event"))
    interval_h = (
        pl.col("funding_interval_hours").cast(pl.Int64, strict=False).fill_null(8)
        if "funding_interval_hours" in df.columns
        else pl.lit(8, dtype=pl.Int64)
    )
    out = df.select(
        exchange=pl.lit(EXCHANGE),
        symbol=pl.lit(symbol),
        ts_event=pl.col("ts_event"),
        ts_recv=pl.col("ts_event"),
        mark_price=pl.lit(float("nan")),
        index_price=pl.lit(float("nan")),
        funding_rate=pl.col("last_funding_rate").cast(pl.Float64),
        next_funding_ts=pl.col("ts_event") + interval_h * 3_600 * 1_000_000_000,
    )
    return _conform(out.sort("ts_event"), POLARS_SCHEMAS["mark_price"])


def _normalize_book_ticker(df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    df = df.with_columns(
        _epoch_to_ns("transaction_time").alias("_tx"), _epoch_to_ns("event_time").alias("_ev")
    )
    ts_event = pl.when(pl.col("_tx") > 0).then(pl.col("_tx")).otherwise(pl.col("_ev"))
    out = df.select(
        exchange=pl.lit(EXCHANGE),
        symbol=pl.lit(symbol),
        ts_event=ts_event,
        ts_recv=pl.when(pl.col("_ev") > 0).then(pl.col("_ev")).otherwise(ts_event),
        update_id=pl.col("update_id").cast(pl.Int64),
        bid_price=pl.col("best_bid_price").cast(pl.Float64),
        bid_qty=pl.col("best_bid_qty").cast(pl.Float64),
        ask_price=pl.col("best_ask_price").cast(pl.Float64),
        ask_qty=pl.col("best_ask_qty").cast(pl.Float64),
    )
    return _conform(out.sort("ts_event"), BOOK_TICKER_SCHEMA)


def _normalize_book_depth(df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    df = _time_str_or_epoch_to_ns(df, "timestamp")
    out = df.select(
        exchange=pl.lit(EXCHANGE),
        symbol=pl.lit(symbol),
        ts_event=pl.col("timestamp"),
        ts_recv=pl.col("timestamp"),
        percentage=pl.col("percentage").cast(pl.Float64),
        depth=pl.col("depth").cast(pl.Float64),
        notional=pl.col("notional").cast(pl.Float64),
    )
    return _conform(out.sort("ts_event"), BOOK_DEPTH_SCHEMA)


def normalize_csv(kind: str, symbol: str, data: bytes) -> dict[str, pl.DataFrame]:
    """One archive CSV -> {stream: unified-schema frame}.

    Archives carry no local receive time, so ts_recv = ts_event.
    """
    if kind not in _CSV_COLUMNS:
        raise ValueError(f"no normalizer for Vision kind: {kind!r}")
    df = _read_csv(data, kind)
    if df.is_empty():
        return {s: pl.DataFrame(schema=stream_schema(s)) for s in KIND_STREAMS[kind]}
    if kind == "aggTrades":
        return {"trade": _normalize_agg_trades(df, symbol)}
    if kind == "trades":
        return {"trade": _normalize_trades(df, symbol)}
    if kind == "klines":
        return {"kline": _normalize_klines(df, symbol)}
    if kind == "premiumIndexKlines":
        return {"premium_index_kline": _normalize_klines(df, symbol)}
    if kind == "liquidationSnapshot":
        return {"liquidation": _normalize_liquidations(df, symbol)}
    if kind == "metrics":
        return _normalize_metrics(df, symbol)
    if kind == "fundingRate":
        return {"mark_price": _normalize_funding(df, symbol)}
    if kind == "bookTicker":
        return {"book_ticker": _normalize_book_ticker(df, symbol)}
    return {"book_depth": _normalize_book_depth(df, symbol)}


def extract_csv(zip_data: bytes) -> bytes:
    """First CSV member of a Vision zip archive."""
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")] or zf.namelist()
        if not names:
            raise ValueError("empty zip archive")
        return zf.read(names[0])


def ingest_zip(
    zip_data: bytes, kind: str, symbol: str, lake_root: Path
) -> dict[str, list[Path]]:
    """Unzip + normalize one archive and append it to the parquet lake."""
    frames = normalize_csv(kind, symbol, extract_csv(zip_data))
    return {stream: write_batch(lake_root, stream, frame) for stream, frame in frames.items()}


def read_local_stream(
    root: Path,
    stream: str,
    exchange: str | None = None,
    symbol: str | None = None,
) -> pl.DataFrame:
    """Reader for locally-defined streams (book_ticker/book_depth/premium_index_kline).

    core.io.read_stream only knows POLARS_SCHEMAS; this mirrors it for
    LOCAL_SCHEMAS over the same partition layout.
    """
    schema = stream_schema(stream)
    base = root / stream
    pattern = f"exchange={exchange or '*'}/symbol={symbol or '*'}/date=*/hour=*/part-*.parquet"
    files = sorted(base.glob(pattern)) if base.exists() else []
    if not files:
        return pl.DataFrame(schema=schema)
    ts_col = "ts_event" if "ts_event" in schema else "ts_open"
    return pl.scan_parquet(files).sort(ts_col).collect()


# ---------------------------------------------------------------------------
# Reconciliation: own recording vs Vision
# ---------------------------------------------------------------------------


def _rel_diff(a: str, b: str) -> pl.Expr:
    denom = pl.max_horizontal(pl.col(a).abs(), pl.col(b).abs(), pl.lit(1e-12))
    return (pl.col(a) - pl.col(b)).abs() / denom


def reconcile_trades(
    own: pl.DataFrame,
    vision: pl.DataFrame,
    *,
    count_tol: float = 1e-9,
    volume_tol: float = 1e-9,
) -> pl.DataFrame:
    """Per-minute trade count and volume relative diffs between two trade frames."""

    def per_minute(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
        return (
            df.with_columns(
                (pl.col("ts_event") - pl.col("ts_event") % NS_PER_MIN).alias("minute")
            )
            .group_by("minute")
            .agg(
                pl.len().cast(pl.Float64).alias(f"{prefix}_count"),
                pl.col("qty").sum().alias(f"{prefix}_volume"),
            )
        )

    joined = (
        per_minute(own, "own")
        .join(per_minute(vision, "vision"), on="minute", how="full", coalesce=True)
        .fill_null(0.0)
        .sort("minute")
    )
    return joined.with_columns(
        _rel_diff("own_count", "vision_count").alias("count_rel_diff"),
        _rel_diff("own_volume", "vision_volume").alias("volume_rel_diff"),
    ).with_columns(
        (
            (pl.col("count_rel_diff") <= count_tol) & (pl.col("volume_rel_diff") <= volume_tol)
        ).alias("ok")
    )


_KLINE_FIELDS = ("open", "high", "low", "close", "volume")


def reconcile_klines(
    own: pl.DataFrame, vision: pl.DataFrame, *, rel_tol: float = 1e-9
) -> pl.DataFrame:
    """Per-bar OHLCV relative diffs between two kline frames, joined on ts_open."""
    o = own.select("ts_open", *[pl.col(f).alias(f"own_{f}") for f in _KLINE_FIELDS])
    v = vision.select("ts_open", *[pl.col(f).alias(f"vision_{f}") for f in _KLINE_FIELDS])
    joined = o.join(v, on="ts_open", how="full", coalesce=True).sort("ts_open")
    diffs = [_rel_diff(f"own_{f}", f"vision_{f}").alias(f"{f}_rel_diff") for f in _KLINE_FIELDS]
    joined = joined.with_columns(diffs)
    missing = pl.any_horizontal(
        *[pl.col(f"own_{f}").is_null() | pl.col(f"vision_{f}").is_null() for f in _KLINE_FIELDS]
    )
    within = pl.all_horizontal(*[pl.col(f"{f}_rel_diff") <= rel_tol for f in _KLINE_FIELDS])
    return joined.with_columns((~missing & within.fill_null(False)).alias("ok"))


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    trades: pl.DataFrame
    klines: pl.DataFrame
    passed: bool


def reconcile(
    own_trades: pl.DataFrame,
    vision_trades: pl.DataFrame,
    own_klines: pl.DataFrame,
    vision_klines: pl.DataFrame,
    *,
    count_tol: float = 1e-9,
    volume_tol: float = 1e-9,
    kline_rel_tol: float = 1e-9,
) -> ReconcileResult:
    """Own recording vs Vision over the overlapping window: frames + boolean verdict."""
    t = reconcile_trades(own_trades, vision_trades, count_tol=count_tol, volume_tol=volume_tol)
    k = reconcile_klines(own_klines, vision_klines, rel_tol=kline_rel_tol)
    passed = bool(t.get_column("ok").all()) and bool(k.get_column("ok").all())
    return ReconcileResult(trades=t, klines=k, passed=passed)


# ---------------------------------------------------------------------------
# Dataset catalog
# ---------------------------------------------------------------------------

CATALOG_SCHEMA: dict[str, pl.DataType] = {
    "stream": pl.Utf8,
    "exchange": pl.Utf8,
    "symbol": pl.Utf8,
    "date": pl.Utf8,
    "hours_present": pl.Int64,
    "rows": pl.Int64,
    "quality": pl.Utf8,
}
CATALOG_REL_PATH = Path("_catalog") / "catalog.parquet"


def build_catalog(
    lake_root: Path, expected_hours: dict[str, int] | None = None
) -> pl.DataFrame:
    """Scan the parquet lake -> one row per (stream, exchange, symbol, date).

    quality = complete when hours_present >= expected hours for the stream
    (default 24), else partial.
    """
    import pyarrow.parquet as pq

    expected_hours = expected_hours or {}
    rows: list[dict] = []
    if lake_root.exists():
        for stream_dir in sorted(p for p in lake_root.iterdir() if p.is_dir()):
            stream = stream_dir.name
            if stream.startswith("_"):
                continue
            for f in sorted(stream_dir.glob("exchange=*/symbol=*/date=*/hour=*/part-*.parquet")):
                hour_d = f.parent
                date_d = hour_d.parent
                symbol_d = date_d.parent
                exchange_d = symbol_d.parent
                rows.append(
                    {
                        "stream": stream,
                        "exchange": exchange_d.name.split("=", 1)[1],
                        "symbol": symbol_d.name.split("=", 1)[1],
                        "date": date_d.name.split("=", 1)[1],
                        "hour": int(hour_d.name.split("=", 1)[1]),
                        "rows": pq.ParquetFile(f).metadata.num_rows,
                    }
                )
    if not rows:
        return pl.DataFrame(schema=CATALOG_SCHEMA)
    per_file = pl.DataFrame(rows)
    catalog = (
        per_file.group_by("stream", "exchange", "symbol", "date")
        .agg(
            pl.col("hour").n_unique().cast(pl.Int64).alias("hours_present"),
            pl.col("rows").sum().cast(pl.Int64).alias("rows"),
        )
        .sort("stream", "exchange", "symbol", "date")
    )
    expected = pl.col("stream").replace_strict(
        expected_hours, default=24, return_dtype=pl.Int64
    )
    catalog = catalog.with_columns(
        pl.when(pl.col("hours_present") >= expected)
        .then(pl.lit("complete"))
        .otherwise(pl.lit("partial"))
        .alias("quality")
    )
    return _conform(catalog, CATALOG_SCHEMA)


def write_catalog(
    lake_root: Path,
    catalog: pl.DataFrame | None = None,
    expected_hours: dict[str, int] | None = None,
) -> Path:
    """Persist the catalog as parquet under the lake root; returns the file path."""
    catalog = catalog if catalog is not None else build_catalog(lake_root, expected_hours)
    path = lake_root / CATALOG_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_parquet(path)
    return path


def read_catalog(lake_root: Path) -> pl.DataFrame:
    path = lake_root / CATALOG_REL_PATH
    if not path.exists():
        return pl.DataFrame(schema=CATALOG_SCHEMA)
    return pl.read_parquet(path)


# ---------------------------------------------------------------------------
# Demo reports (checklist "виз" for 1.2)
# ---------------------------------------------------------------------------


def _synth_catalog(seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    streams = ["trade", "kline", "liquidation", "open_interest", "ratio", "mark_price"]
    start = dt.date(2024, 1, 1)
    rows = []
    for symbol in ("BTCUSDT", "SOLUSDT", "DOGEUSDT"):
        for stream in streams:
            for i in range(14):
                hours = 24 if rng.random() > 0.2 else int(rng.integers(1, 24))
                rows.append(
                    {
                        "stream": stream,
                        "exchange": EXCHANGE,
                        "symbol": symbol,
                        "date": (start + dt.timedelta(days=i)).isoformat(),
                        "hours_present": hours,
                        "rows": int(rng.integers(1_000, 500_000)) * hours // 24,
                        "quality": "complete" if hours == 24 else "partial",
                    }
                )
    return pl.DataFrame(rows, schema=CATALOG_SCHEMA)


def demo_reports(out_dir: Path, seed: int = 42) -> list[Path]:
    """Generate the 1.2 figures from synthetic data; returns saved png paths.

    Figures: catalog coverage heatmap (days x stream), rows-per-stream bars,
    reconciliation relative-diff distributions (identical vs perturbed).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    from trading_system.core.schema import records_to_frame
    from trading_system.core.synth import synth_trades
    from trading_system.viz.style import PALETTE, apply_style, save_fig

    apply_style()
    out_dir = Path(out_dir)
    paths: list[Path] = []

    catalog = _synth_catalog(seed)
    cov = (
        catalog.filter(pl.col("symbol") == "BTCUSDT")
        .pivot(on="stream", index="date", values="hours_present")
        .sort("date")
    )
    streams = [c for c in cov.columns if c != "date"]
    matrix = cov.select(streams).to_numpy().astype(float)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=PALETTE["heat"],
        vmin=0,
        vmax=24,
        annot=True,
        fmt=".0f",
        xticklabels=streams,
        yticklabels=cov.get_column("date").to_list(),
        cbar_kws={"label": "hours present"},
    )
    ax.set_title("Vision lake coverage — hours per day by stream (BTCUSDT)")
    ax.set_xlabel("stream")
    ax.set_ylabel("date")
    paths.append(save_fig(fig, "vision_catalog_coverage", out_dir))

    per_stream = catalog.group_by("stream").agg(pl.col("rows").sum()).sort("stream")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        x=per_stream.get_column("stream").to_list(),
        y=per_stream.get_column("rows").to_numpy(),
        color=PALETTE["neutral"],
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title("Vision lake — total rows per stream (all symbols, 14 days)")
    ax.set_xlabel("stream")
    ax.set_ylabel("rows (log)")
    paths.append(save_fig(fig, "vision_catalog_rows", out_dir))

    own = records_to_frame(synth_trades(n=30_000, seed=seed), "trade")
    identical = reconcile_trades(own, own)
    rng = np.random.default_rng(seed)
    keep = rng.random(own.height) > 0.01
    perturbed_frame = own.filter(pl.Series(keep)).with_columns(
        (pl.col("qty") * (1 + 0.02 * rng.standard_normal(int(keep.sum())))).alias("qty")
    )
    perturbed = reconcile_trades(own, perturbed_frame)
    tol = 0.001
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    id_diffs = identical.get_column("volume_rel_diff").to_numpy()
    pe_diffs = perturbed.get_column("volume_rel_diff").to_numpy()
    sns.histplot(id_diffs, bins=30, color=PALETTE["long"], stat="probability", ax=axes[0])
    axes[0].set_title(f"identical recording (max diff {id_diffs.max():.1e})")
    axes[0].set_xlabel("per-minute volume relative diff")
    hi = float(max(pe_diffs.max(), tol) * 1.1)
    sns.histplot(
        pe_diffs,
        bins=30,
        binrange=(0.0, hi),
        color=PALETTE["short"],
        stat="probability",
        ax=axes[1],
    )
    axes[1].axvline(tol, color=PALETTE["accent"], linestyle="--", label=f"tolerance {tol}")
    axes[1].legend()
    axes[1].set_title("perturbed recording (1% drops + qty noise)")
    axes[1].set_xlabel("per-minute volume relative diff")
    fig.suptitle("Reconciliation own vs Vision — relative-diff distribution")
    paths.append(save_fig(fig, "vision_reconcile_reldiff", out_dir))
    return paths
