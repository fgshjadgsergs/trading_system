"""Gate 0: the empty pipeline (module stubs) assembles and runs under CI."""

from __future__ import annotations

import importlib

import polars as pl

MODULES = [
    "trading_system.core",
    "trading_system.core.schema",
    "trading_system.core.adapter",
    "trading_system.core.io",
    "trading_system.core.synth",
    "trading_system.collectors",
    "trading_system.book",
    "trading_system.features",
    "trading_system.liqmap",
    "trading_system.profile",
    "trading_system.spoof",
    "trading_system.signals",
    "trading_system.backtest",
    "trading_system.risk",
    "trading_system.viz",
    "trading_system.monitoring",
    "trading_system.calibration",
]


def test_all_modules_import():
    for name in MODULES:
        importlib.import_module(name)


def test_schema_roundtrip(tmp_data):
    from trading_system.core.io import read_stream, write_batch
    from trading_system.core.schema import records_to_frame
    from trading_system.core.synth import synth_trades

    trades = synth_trades(n=500, seed=42)
    frame = records_to_frame(trades, "trade")
    assert frame["qty_usd"].min() > 0
    write_batch(tmp_data, "trade", frame)
    back = read_stream(tmp_data, "trade", symbol="BTCUSDT")
    assert back.height == 500
    assert back["ts_event"].is_sorted()
    assert back.select(pl.col("side").is_in([-1, 1]).all()).item()


def test_synth_book_stream_is_sequential():
    from trading_system.core.synth import synth_book_stream

    s = synth_book_stream(n_diffs=200, seed=42)
    prev = s.snapshot.last_update_id
    for d in s.diffs:
        assert d.prev_final_update_id == prev
        assert d.first_update_id == prev + 1
        assert d.final_update_id >= d.first_update_id
        prev = d.final_update_id


def test_config_and_seeds(cfg):
    assert cfg["project"]["seed"] == 42
    assert set(cfg["symbols"]) == {"BTCUSDT", "SOLUSDT", "DOGEUSDT"}
