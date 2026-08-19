"""M1 demo figures from synthetic data: latency histograms, weekly gap timeline.

demo_reports records a synthetic session through the real pipeline
(normalized records -> BatchWriter -> parquet lake -> daily_quality_report)
with an injected depth-stream drop, then renders the checklist figures.
The real weekly live run is an ops task; this exercises the same code paths.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from trading_system.collectors.quality import (
    Interval,
    coverage_gaps,
    daily_quality_report,
    plot_gap_timeline,
    plot_latency_histograms,
    write_gap_events,
)
from trading_system.collectors.recorder import BatchWriter
from trading_system.collectors.sequencer import DepthSequencer
from trading_system.core.synth import (
    EXCHANGE,
    synth_book_stream,
    synth_mark_prices,
    synth_open_interest,
    synth_trades,
)
from trading_system.core.timeutils import NS_PER_S
from trading_system.viz.style import apply_style

SYMBOL = "BTCUSDT"
WEEK_NS = 7 * 86_400 * NS_PER_S


def _record_synthetic_day(root: Path, seed: int) -> tuple[int, int]:
    """Write a synthetic recording with a trade silence and a depth gap."""
    trades = synth_trades(n=4_000, symbol=SYMBOL, seed=seed)
    t0 = trades[0].ts_event
    t1 = trades[-1].ts_event + NS_PER_S
    span = t1 - t0
    # cut trades in a sub-window -> uptime < 100% for the trade stream
    cut_from, cut_to = t0 + int(0.42 * span), t0 + int(0.50 * span)
    kept_trades = [t for t in trades if not cut_from <= t.ts_event < cut_to]

    marks = synth_mark_prices(trades)
    oi = synth_open_interest(symbol=SYMBOL, start_ts=t0, n=int(span / (7 * NS_PER_S)), seed=seed)
    book = synth_book_stream(n_diffs=4_000, symbol=SYMBOL, start_ts=t0, seed=seed)

    # sequencer sees a dropped chunk of diffs -> GapEvent persisted to 'gaps'
    seq = DepthSequencer(SYMBOL)
    seq.set_snapshot(book.snapshot)
    applied = []
    for i, d in enumerate(book.diffs):
        if 700 <= i < 720:
            continue  # simulated websocket drop
        applied.extend(seq.add_diff(d))

    writer = BatchWriter(root, max_rows=1_000, max_age_s=3600.0, clock=lambda: t1)
    for rec_list in (kept_trades, marks, oi, applied):
        for rec in rec_list:
            writer.add(rec)
    writer.flush_all()
    write_gap_events(root, seq.gaps, EXCHANGE, "depth_diff")
    return t0, t1


def _weekly_coverage(seed: int, t0: int) -> tuple[dict[str, list[Interval]], int, int]:
    """Synthetic 7-day per-stream coverage with seeded outages."""
    rng = np.random.default_rng(seed + 1)
    t1 = t0 + WEEK_NS
    cadence_ns = 10 * NS_PER_S
    gaps: dict[str, list[Interval]] = {}
    for stream in ("depth_diff", "trade", "mark_price", "open_interest"):
        ts = np.arange(t0, t1, cadence_ns, dtype=np.int64)
        for _ in range(int(rng.integers(1, 4))):
            start = t0 + int(rng.integers(0, WEEK_NS - 6 * 3_600 * NS_PER_S))
            dur = int(rng.integers(15 * 60, 6 * 3_600)) * NS_PER_S
            ts = ts[(ts < start) | (ts >= start + dur)]
        gaps[stream] = coverage_gaps(ts, t0, t1, max_silence_ns=60 * NS_PER_S)
    return gaps, t0, t1


def demo_reports(out_dir: Path, seed: int = 42) -> list[Path]:
    """Generate all M1 checklist figures from synthetic data; returns png paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    paths: list[Path] = []
    with tempfile.TemporaryDirectory(dir=out_dir, prefix="_demo_lake_") as lake:
        root = Path(lake)
        t0, t1 = _record_synthetic_day(root, seed)
        report = daily_quality_report(root, EXCHANGE, SYMBOL, t0, t1)
        latencies = {s: q.latency_ms for s, q in report.streams.items()}
        paths.append(plot_latency_histograms(latencies, out_dir))
    weekly_gaps, w0, w1 = _weekly_coverage(seed, t0)
    paths.append(plot_gap_timeline(weekly_gaps, w0, w1, out_dir))
    return paths
