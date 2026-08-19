"""Dataset catalog over the parquet lake (1.2)."""

from __future__ import annotations

import polars as pl

from trading_system.collectors.vision import (
    CATALOG_SCHEMA,
    build_catalog,
    read_catalog,
    write_catalog,
)
from trading_system.core.io import write_batch
from trading_system.core.schema import POLARS_SCHEMAS
from trading_system.core.timeutils import NS_PER_S

DAY0 = 1_704_067_200 * NS_PER_S  # 2024-01-01 00:00:00 UTC
HOUR = 3_600 * NS_PER_S


def _trade_rows(ts_list: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "exchange": "binance_usdm",
                "symbol": "BTCUSDT",
                "ts_event": ts,
                "ts_recv": ts,
                "price": 100.0,
                "qty": 1.0,
                "qty_usd": 100.0,
                "side": 1,
                "trade_id": i + 1,
            }
            for i, ts in enumerate(ts_list)
        ],
        schema=POLARS_SCHEMAS["trade"],
    )


def _kline_rows(hours: int) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "exchange": "binance_usdm",
                "symbol": "BTCUSDT",
                "ts_open": DAY0 + h * HOUR,
                "ts_close": DAY0 + (h + 1) * HOUR - 1,
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 1.0,
                "quote_volume": 1.5,
                "taker_buy_volume": 0.5,
                "taker_buy_quote_volume": 0.75,
                "n_trades": 10,
                "closed": True,
            }
            for h in range(hours)
        ],
        schema=POLARS_SCHEMAS["kline"],
    )


def _build_lake(root):
    # trade: 2 hours on day 1 + 1 hour on day 2 -> two partial days
    write_batch(root, "trade", _trade_rows([DAY0 + 10, DAY0 + 20, DAY0 + HOUR + 30]))
    write_batch(root, "trade", _trade_rows([DAY0 + 24 * HOUR + 40]))
    # kline: one row in each of the 24 hours -> complete day
    write_batch(root, "kline", _kline_rows(24))


def test_catalog_rows_hours_and_quality(tmp_path):
    _build_lake(tmp_path)
    cat = build_catalog(tmp_path)
    assert cat.schema == pl.Schema(CATALOG_SCHEMA)
    assert cat.height == 3
    trade_d1 = cat.filter((pl.col("stream") == "trade") & (pl.col("date") == "2024-01-01"))
    assert trade_d1.row(0, named=True) == {
        "stream": "trade",
        "exchange": "binance_usdm",
        "symbol": "BTCUSDT",
        "date": "2024-01-01",
        "hours_present": 2,
        "rows": 3,
        "quality": "partial",
    }
    trade_d2 = cat.filter((pl.col("stream") == "trade") & (pl.col("date") == "2024-01-02"))
    assert trade_d2.get_column("hours_present")[0] == 1
    kline = cat.filter(pl.col("stream") == "kline")
    assert kline.get_column("hours_present")[0] == 24
    assert kline.get_column("quality")[0] == "complete"
    assert kline.get_column("rows")[0] == 24


def test_expected_hours_override(tmp_path):
    _build_lake(tmp_path)
    cat = build_catalog(tmp_path, expected_hours={"trade": 2})
    trade_d1 = cat.filter((pl.col("stream") == "trade") & (pl.col("date") == "2024-01-01"))
    assert trade_d1.get_column("quality")[0] == "complete"


def test_catalog_persisted_under_lake_and_ignored_on_rescan(tmp_path):
    _build_lake(tmp_path)
    path = write_catalog(tmp_path)
    assert path == tmp_path / "_catalog" / "catalog.parquet"
    assert path.exists()
    round_trip = read_catalog(tmp_path)
    assert round_trip.equals(build_catalog(tmp_path))
    # the _catalog directory itself must not appear as a stream
    again = build_catalog(tmp_path)
    assert "_catalog" not in again.get_column("stream").to_list()


def test_empty_lake(tmp_path):
    cat = build_catalog(tmp_path / "nowhere")
    assert cat.is_empty()
    assert cat.schema == pl.Schema(CATALOG_SCHEMA)
    assert read_catalog(tmp_path).is_empty()
