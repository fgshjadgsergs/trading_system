"""Archive CSV -> unified schema normalizers (1.2)."""

from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path

import polars as pl
import pytest

from trading_system.collectors.vision import (
    BOOK_TICKER_SCHEMA,
    KIND_STREAMS,
    ingest_zip,
    normalize_csv,
    read_local_stream,
)
from trading_system.core.io import read_stream
from trading_system.core.schema import POLARS_SCHEMAS

FIXTURES = Path(__file__).parent / "fixtures" / "vision"
NS = 1_000_000  # ms -> ns


def fx(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_agg_trades_exact_values():
    frames = normalize_csv("aggTrades", "BTCUSDT", fx("aggTrades.csv"))
    df = frames["trade"]
    assert df.schema == pl.Schema(POLARS_SCHEMAS["trade"])
    assert df.height == 3
    row = df.row(0, named=True)
    assert row["exchange"] == "binance_usdm"
    assert row["symbol"] == "BTCUSDT"
    assert row["ts_event"] == 1704067200123 * NS
    assert row["ts_recv"] == row["ts_event"]  # archives carry no local recv time
    assert row["price"] == 50000.5
    assert row["qty"] == 0.002
    assert row["qty_usd"] == pytest.approx(100.001)
    assert row["trade_id"] == 1000
    # is_buyer_maker=true => taker sold
    assert df.get_column("side").to_list() == [-1, 1, -1]


def test_agg_trades_headerless_matches_header_variant():
    with_header = normalize_csv("aggTrades", "BTCUSDT", fx("aggTrades.csv"))["trade"]
    without = normalize_csv("aggTrades", "BTCUSDT", fx("aggTrades_noheader.csv"))["trade"]
    assert with_header.equals(without)


def test_timestamp_unit_defense_seconds_and_micros():
    base = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
    secs = normalize_csv("aggTrades", "X", (base + "1,1.0,1.0,1,1,1704067200,false\n").encode())
    micros = normalize_csv(
        "aggTrades", "X", (base + "1,1.0,1.0,1,1,1704067200123456,false\n").encode()
    )
    assert secs["trade"].get_column("ts_event")[0] == 1_704_067_200 * 1_000_000_000
    assert micros["trade"].get_column("ts_event")[0] == 1_704_067_200_123_456_000


def test_trades_kind():
    df = normalize_csv("trades", "BTCUSDT", fx("trades.csv"))["trade"]
    assert df.height == 2
    assert df.get_column("trade_id").to_list() == [501, 502]
    assert df.get_column("side").to_list() == [1, -1]
    assert df.get_column("ts_event").to_list() == [1704067200100 * NS, 1704067200200 * NS]


def test_klines_exact_values():
    df = normalize_csv("klines", "BTCUSDT", fx("klines.csv"))["kline"]
    assert df.schema == pl.Schema(POLARS_SCHEMAS["kline"])
    row = df.row(0, named=True)
    assert row["ts_open"] == 1704067200000 * NS
    assert row["ts_close"] == 1704067259999 * NS
    assert (row["open"], row["high"], row["low"], row["close"]) == (
        50000.0,
        50100.0,
        49950.0,
        50050.0,
    )
    assert row["volume"] == 12.5
    assert row["quote_volume"] == 625625.0
    assert row["taker_buy_volume"] == 7.5
    assert row["n_trades"] == 240
    assert row["closed"] is True


def test_premium_index_klines_use_local_stream_name():
    frames = normalize_csv("premiumIndexKlines", "BTCUSDT", fx("klines.csv"))
    assert set(frames) == {"premium_index_kline"}
    assert frames["premium_index_kline"].schema == pl.Schema(POLARS_SCHEMAS["kline"])


def test_liquidation_snapshot_mapping():
    df = normalize_csv("liquidationSnapshot", "BTCUSDT", fx("liquidationSnapshot.csv"))[
        "liquidation"
    ]
    assert df.schema == pl.Schema(POLARS_SCHEMAS["liquidation"])
    r0, r1 = df.row(0, named=True), df.row(1, named=True)
    assert r0["ts_event"] == 1704067205000 * NS
    assert r0["side"] == -1  # SELL order = long position liquidated
    assert r0["price"] == 49855.5  # average fill price preferred
    assert r0["qty"] == 0.014
    assert r0["qty_usd"] == pytest.approx(49855.5 * 0.014)
    assert r1["side"] == 1
    assert r1["qty"] == 0.200  # accumulated fill, not last fill


def test_metrics_open_interest_and_three_ratio_rows():
    frames = normalize_csv("metrics", "BTCUSDT", fx("metrics.csv"))
    oi, ratios = frames["open_interest"], frames["ratio"]
    assert oi.schema == pl.Schema(POLARS_SCHEMAS["open_interest"])
    assert ratios.schema == pl.Schema(POLARS_SCHEMAS["ratio"])
    assert oi.height == 2
    # 2024-01-01 00:05:00 UTC
    assert oi.get_column("ts_event")[0] == 1_704_067_500 * 1_000_000_000
    assert oi.get_column("open_interest")[0] == 80000.5
    assert oi.get_column("open_interest_usd")[0] == 4000025000.0
    assert ratios.height == 6  # three metrics per timestamp
    first = ratios.filter(pl.col("ts_event") == 1_704_067_500 * 1_000_000_000).sort("metric")
    assert first.get_column("metric").to_list() == [
        "global_ls_account",
        "taker_ls",
        "top_ls_position",
    ]
    by_metric = {r["metric"]: r for r in first.iter_rows(named=True)}
    assert by_metric["global_ls_account"]["ratio"] == 2.10
    assert by_metric["global_ls_account"]["long_share"] == pytest.approx(2.10 / 3.10)
    assert by_metric["global_ls_account"]["short_share"] == pytest.approx(1.0 / 3.10)
    assert by_metric["top_ls_position"]["ratio"] == 1.42
    assert by_metric["taker_ls"]["ratio"] == 0.95


def test_funding_rate_to_mark_price_rows():
    df = normalize_csv("fundingRate", "BTCUSDT", fx("fundingRate.csv"))["mark_price"]
    assert df.schema == pl.Schema(POLARS_SCHEMAS["mark_price"])
    r0 = df.row(0, named=True)
    assert r0["ts_event"] == 1704067200000 * NS
    assert r0["funding_rate"] == 0.0001
    assert math.isnan(r0["mark_price"]) and math.isnan(r0["index_price"])
    assert r0["next_funding_ts"] == r0["ts_event"] + 8 * 3600 * 1_000_000_000
    assert df.get_column("funding_rate")[1] == -0.00005


def test_book_ticker_local_schema():
    df = normalize_csv("bookTicker", "BTCUSDT", fx("bookTicker.csv"))["book_ticker"]
    assert df.schema == pl.Schema(BOOK_TICKER_SCHEMA)
    r0 = df.row(0, named=True)
    assert r0["ts_event"] == 1704067200050 * NS
    assert r0["ts_recv"] == 1704067200055 * NS
    assert (r0["bid_price"], r0["bid_qty"]) == (49999.9, 1.5)
    assert (r0["ask_price"], r0["ask_qty"]) == (50000.1, 2.0)
    assert r0["update_id"] == 9001


def test_book_depth_datetime_strings():
    df = normalize_csv("bookDepth", "BTCUSDT", fx("bookDepth.csv"))["book_depth"]
    assert df.height == 2
    assert df.get_column("ts_event").unique().to_list() == [1_704_067_260 * 1_000_000_000]
    assert df.get_column("percentage").to_list() == [1.0, -1.0]


def test_empty_csv_gives_empty_conformant_frames():
    for kind, streams in KIND_STREAMS.items():
        frames = normalize_csv(kind, "BTCUSDT", b"")
        assert set(frames) == set(streams)
        assert all(f.is_empty() for f in frames.values())


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="no normalizer"):
        normalize_csv("depthUpdate", "BTCUSDT", b"1,2,3\n")


def _zip_of(name: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name.replace(".csv", "") + ".csv", fx(name))
    return buf.getvalue()


def test_ingest_zip_writes_unified_lake(tmp_path):
    lake = tmp_path / "lake"
    written = ingest_zip(_zip_of("aggTrades.csv"), "aggTrades", "BTCUSDT", lake)
    assert set(written) == {"trade"}
    df = read_stream(lake, "trade", exchange="binance_usdm", symbol="BTCUSDT")
    assert df.height == 3
    assert df.get_column("ts_event").is_sorted()


def test_ingest_zip_book_ticker_readable_via_local_reader(tmp_path):
    lake = tmp_path / "lake"
    written = ingest_zip(_zip_of("bookTicker.csv"), "bookTicker", "BTCUSDT", lake)
    assert set(written) == {"book_ticker"}
    df = read_local_stream(lake, "book_ticker", symbol="BTCUSDT")
    assert df.height == 2
    assert df.schema == pl.Schema(BOOK_TICKER_SCHEMA)
