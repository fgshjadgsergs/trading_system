"""M1: quality report — uptime, gaps parquet stream, latency data, figures."""

from __future__ import annotations

import numpy as np

from trading_system.collectors.quality import (
    GAPS_SCHEMA,
    coverage_gaps,
    daily_quality_report,
    gap_events_to_frame,
    plot_gap_timeline,
    plot_latency_histograms,
    read_gaps,
    uptime_pct,
    write_gap_events,
)
from trading_system.collectors.recorder import BatchWriter
from trading_system.collectors.sequencer import DepthSequencer
from trading_system.core.synth import (
    EXCHANGE,
    synth_book_stream,
    synth_mark_prices,
    synth_trades,
)
from trading_system.core.timeutils import NS_PER_S

SYMBOL = "BTCUSDT"


# --------------------------------------------------------------------------- #
# coverage primitives
# --------------------------------------------------------------------------- #
def test_coverage_gaps_empty_and_dense():
    t0, t1 = 0, 100 * NS_PER_S
    assert coverage_gaps(np.empty(0, np.int64), t0, t1, 5 * NS_PER_S) == [(t0, t1)]
    assert uptime_pct([(t0, t1)], t0, t1) == 0.0
    dense = np.arange(t0, t1, NS_PER_S, dtype=np.int64)
    assert coverage_gaps(dense, t0, t1, 5 * NS_PER_S) == []
    assert uptime_pct([], t0, t1) == 100.0


def test_coverage_gaps_finds_injected_silence():
    t0, t1 = 0, 1_000 * NS_PER_S
    ts = np.arange(t0, t1, NS_PER_S, dtype=np.int64)
    ts = ts[(ts < 300 * NS_PER_S) | (ts >= 400 * NS_PER_S)]  # 100 s of silence
    gaps = coverage_gaps(ts, t0, t1, 10 * NS_PER_S)
    assert len(gaps) == 1
    s, e = gaps[0]
    assert s == 299 * NS_PER_S and e == 400 * NS_PER_S
    assert 89.0 < uptime_pct(gaps, t0, t1) < 91.0


# --------------------------------------------------------------------------- #
# full report over a synthetic recording with an injected gap
# --------------------------------------------------------------------------- #
def _build_lake(root):
    trades = synth_trades(n=2_000, symbol=SYMBOL, seed=11)
    t0 = trades[0].ts_event
    t1 = trades[-1].ts_event + NS_PER_S
    span = t1 - t0
    cut_from, cut_to = t0 + int(0.30 * span), t0 + int(0.45 * span)
    kept = [t for t in trades if not cut_from <= t.ts_event < cut_to]
    marks = synth_mark_prices(trades)  # full coverage: contrast with the cut trades

    book = synth_book_stream(n_diffs=2_000, symbol=SYMBOL, start_ts=t0, seed=11)
    seq = DepthSequencer(SYMBOL)
    seq.set_snapshot(book.snapshot)
    applied = []
    for i, d in enumerate(book.diffs):
        if 900 <= i < 910:
            continue  # simulated disconnect: 10 lost diffs
        applied.extend(seq.add_diff(d))

    writer = BatchWriter(root, max_rows=500, max_age_s=3600.0, clock=lambda: t1)
    for rec in [*kept, *marks, *applied]:
        writer.add(rec)
    writer.flush_all()
    assert seq.gaps
    write_gap_events(root, seq.gaps, EXCHANGE, "depth_diff")
    return t0, t1, (cut_from, cut_to)


def test_daily_report_detects_gap_and_uptime_below_100(tmp_data):
    t0, t1, (cut_from, cut_to) = _build_lake(tmp_data)
    report = daily_quality_report(
        tmp_data,
        EXCHANGE,
        SYMBOL,
        t0,
        t1,
        max_silence_s={"trade": 10.0, "mark_price": 10.0},
    )
    trade_q = report.streams["trade"]
    assert trade_q.n_records > 0
    assert trade_q.uptime_pct < 100.0
    assert any(s <= cut_from and e >= cut_to for s, e in trade_q.silence_gaps)
    assert report.streams["mark_price"].uptime_pct == 100.0
    assert not report.clean

    assert len(report.seq_gaps) == 1
    row = report.seq_gaps.row(0, named=True)
    assert row["symbol"] == SYMBOL
    assert row["stream"] == "depth_diff"
    assert row["got"] != row["expected"]

    for q in report.streams.values():
        assert (q.latency_ms >= 0).all()
        assert q.latency_ms.size == q.n_records


def test_gaps_frame_schema_and_roundtrip(tmp_data):
    _build_lake(tmp_data)
    frame = read_gaps(tmp_data, exchange=EXCHANGE, symbol=SYMBOL)
    assert dict(frame.schema) == dict(GAPS_SCHEMA)
    assert len(frame) == 1
    empty = gap_events_to_frame([], EXCHANGE, "depth_diff")
    assert empty.is_empty() and dict(empty.schema) == dict(GAPS_SCHEMA)
    assert write_gap_events(tmp_data, [], EXCHANGE, "depth_diff") == []
    assert read_gaps(tmp_data / "nowhere").is_empty()


def test_quality_figures_render(tmp_data, tmp_reports):
    t0, t1, _ = _build_lake(tmp_data)
    report = daily_quality_report(
        tmp_data, EXCHANGE, SYMBOL, t0, t1, max_silence_s={"trade": 10.0, "depth_diff": 5.0}
    )
    latencies = {s: q.latency_ms for s, q in report.streams.items()}
    p1 = plot_latency_histograms(latencies, tmp_reports)
    gaps_by_stream = {s: q.silence_gaps for s, q in report.streams.items()}
    p2 = plot_gap_timeline(gaps_by_stream, t0, t1, tmp_reports)
    for p in (p1, p2):
        assert p.exists()
        assert p.stat().st_size > 5 * 1024
        assert p.suffix == ".png"
