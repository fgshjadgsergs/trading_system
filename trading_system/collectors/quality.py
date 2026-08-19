"""Daily recording-quality report over the parquet lake.

Per stream: uptime %, silence gaps (no events longer than the stream's
freshness limit), latency histogram data (ts_recv - ts_event). Sequencer
GapEvents are persisted as their own parquet stream 'gaps' with a local
schema (core POLARS_SCHEMAS intentionally knows nothing about it).

Viz: seaborn latency histograms per stream; weekly gap/uptime timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import structlog

from trading_system.collectors.sequencer import GapEvent
from trading_system.core.io import read_stream, write_batch
from trading_system.core.timeutils import NS_PER_S, ns_to_dt
from trading_system.viz.style import PALETTE, apply_style, save_fig

log = structlog.get_logger(__name__)

GAPS_STREAM = "gaps"

# Local schema: sequence-gap records are an M1 concern, not part of core.
GAPS_SCHEMA: dict[str, pl.DataType] = {
    "exchange": pl.Utf8,
    "symbol": pl.Utf8,
    "ts_event": pl.Int64,
    "ts_recv": pl.Int64,
    "stream": pl.Utf8,  # source stream the gap was detected on, e.g. "depth_diff"
    "expected": pl.Int64,
    "got": pl.Int64,
}

# Freshness limit per stream: silence longer than this counts as a gap.
DEFAULT_MAX_SILENCE_S: dict[str, float] = {
    "trade": 10.0,
    "depth_diff": 5.0,
    "mark_price": 5.0,
    "open_interest": 30.0,
}

Interval = tuple[int, int]  # (start_ns, end_ns)


# --------------------------------------------------------------------------- #
# gap persistence
# --------------------------------------------------------------------------- #
def gap_events_to_frame(
    events: list[GapEvent], exchange: str, source_stream: str
) -> pl.DataFrame:
    rows = [
        {
            "exchange": exchange,
            "symbol": ev.symbol,
            "ts_event": ev.ts,
            "ts_recv": ev.ts,
            "stream": source_stream,
            "expected": ev.expected,
            "got": ev.got,
        }
        for ev in events
    ]
    if not rows:
        return pl.DataFrame(schema=GAPS_SCHEMA)
    return pl.DataFrame(rows, schema=GAPS_SCHEMA, orient="row")


def write_gap_events(
    root: Path, events: list[GapEvent], exchange: str, source_stream: str
) -> list[Path]:
    """Persist sequencer gaps under the partitioned 'gaps' parquet stream."""
    frame = gap_events_to_frame(events, exchange, source_stream)
    if frame.is_empty():
        return []
    return write_batch(Path(root), GAPS_STREAM, frame)


def read_gaps(
    root: Path,
    exchange: str | None = None,
    symbol: str | None = None,
    ts_from: int | None = None,
    ts_to: int | None = None,
) -> pl.DataFrame:
    """Local reader for the 'gaps' stream (core reader only knows core schemas)."""
    base = Path(root) / GAPS_STREAM
    pattern = f"exchange={exchange or '*'}/symbol={symbol or '*'}/date=*/hour=*/part-*.parquet"
    files = sorted(base.glob(pattern)) if base.exists() else []
    if not files:
        return pl.DataFrame(schema=GAPS_SCHEMA)
    lf = pl.scan_parquet(files)
    if ts_from is not None:
        lf = lf.filter(pl.col("ts_event") >= ts_from)
    if ts_to is not None:
        lf = lf.filter(pl.col("ts_event") < ts_to)
    return lf.sort("ts_event").collect()


# --------------------------------------------------------------------------- #
# coverage / uptime
# --------------------------------------------------------------------------- #
def coverage_gaps(ts: np.ndarray, t0: int, t1: int, max_silence_ns: int) -> list[Interval]:
    """Silence intervals within [t0, t1) where no event arrives for too long."""
    if t1 <= t0:
        raise ValueError("empty window")
    ts = np.sort(np.asarray(ts, dtype=np.int64))
    ts = ts[(ts >= t0) & (ts < t1)]
    if ts.size == 0:
        return [(t0, t1)]
    edges = np.concatenate(([t0], ts, [t1]))
    starts = edges[:-1]
    ends = edges[1:]
    mask = (ends - starts) > max_silence_ns
    return [(int(s), int(e)) for s, e in zip(starts[mask], ends[mask], strict=True)]


def uptime_pct(gaps: list[Interval], t0: int, t1: int) -> float:
    """Share of the window not covered by silence gaps, in percent."""
    silent = sum(e - s for s, e in gaps)
    return 100.0 * (1.0 - silent / (t1 - t0))


@dataclass(frozen=True, slots=True)
class StreamQuality:
    stream: str
    n_records: int
    uptime_pct: float
    silence_gaps: list[Interval]
    latency_ms: np.ndarray  # ts_recv - ts_event per record


@dataclass(frozen=True, slots=True)
class QualityReport:
    exchange: str
    symbol: str
    t0: int
    t1: int
    streams: dict[str, StreamQuality] = field(default_factory=dict)
    seq_gaps: pl.DataFrame = field(default_factory=lambda: pl.DataFrame(schema=GAPS_SCHEMA))

    @property
    def clean(self) -> bool:
        """True when every stream is at 100% uptime and no sequence gaps exist."""
        full = all(q.uptime_pct >= 100.0 for q in self.streams.values())
        return full and self.seq_gaps.is_empty()


def daily_quality_report(
    root: Path,
    exchange: str,
    symbol: str,
    t0: int,
    t1: int,
    max_silence_s: dict[str, float] | None = None,
) -> QualityReport:
    """Quality report for one symbol over [t0, t1) from the parquet lake."""
    limits = max_silence_s or DEFAULT_MAX_SILENCE_S
    streams: dict[str, StreamQuality] = {}
    for stream, limit_s in limits.items():
        frame = read_stream(Path(root), stream, exchange=exchange, symbol=symbol, ts_from=t0, ts_to=t1)
        ts = frame["ts_event"].to_numpy() if not frame.is_empty() else np.empty(0, np.int64)
        lat = (
            ((frame["ts_recv"] - frame["ts_event"]) / 1e6).to_numpy()
            if not frame.is_empty()
            else np.empty(0, np.float64)
        )
        gaps = coverage_gaps(ts, t0, t1, int(limit_s * NS_PER_S))
        streams[stream] = StreamQuality(
            stream=stream,
            n_records=len(frame),
            uptime_pct=uptime_pct(gaps, t0, t1),
            silence_gaps=gaps,
            latency_ms=lat,
        )
    seq_gaps = read_gaps(Path(root), exchange=exchange, symbol=symbol, ts_from=t0, ts_to=t1)
    report = QualityReport(
        exchange=exchange, symbol=symbol, t0=t0, t1=t1, streams=streams, seq_gaps=seq_gaps
    )
    log.info(
        "quality_report",
        symbol=symbol,
        clean=report.clean,
        seq_gaps=len(seq_gaps),
        uptime={s: round(q.uptime_pct, 3) for s, q in streams.items()},
    )
    return report


# --------------------------------------------------------------------------- #
# viz (checklist M1: seaborn latency histograms; weekly gap timeline)
# --------------------------------------------------------------------------- #
def plot_latency_histograms(
    latencies: dict[str, np.ndarray], out_dir: Path, name: str = "m1_latency_hist"
) -> Path:
    """Seaborn histogram of ts_recv - ts_event (ms) per stream."""
    apply_style()
    items = [(s, lat) for s, lat in latencies.items()]
    n = max(len(items), 1)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
    colors = [PALETTE["neutral"], PALETTE["long"], PALETTE["short"], PALETTE["accent"]]
    for i, (stream, lat) in enumerate(items):
        ax = axes[i // ncols][i % ncols]
        if lat.size:
            sns.histplot(x=lat, bins=40, ax=ax, color=colors[i % len(colors)])
            ax.axvline(float(np.median(lat)), color="black", ls="--", lw=1, label="median")
            ax.legend(fontsize=8)
        ax.set_title(f"{stream} (n={lat.size})")
        ax.set_xlabel("latency ts_recv - ts_event, ms")
    for j in range(len(items), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("M1: per-stream ingest latency", y=1.02)
    fig.tight_layout()
    return save_fig(fig, name, out_dir)


def plot_gap_timeline(
    gaps_by_stream: dict[str, list[Interval]],
    t0: int,
    t1: int,
    out_dir: Path,
    name: str = "m1_gap_timeline",
) -> Path:
    """Uptime/gap timeline: one row per stream, red spans mark silence gaps."""
    apply_style()
    fig, ax = plt.subplots(figsize=(13, 0.9 * max(len(gaps_by_stream), 1) + 2))
    x0 = mdates.date2num(ns_to_dt(t0))
    x1 = mdates.date2num(ns_to_dt(t1))
    labels: list[str] = []
    for row, (stream, gaps) in enumerate(gaps_by_stream.items()):
        ax.broken_barh([(x0, x1 - x0)], (row - 0.35, 0.7), facecolors=PALETTE["long"], alpha=0.45)
        spans = [
            (mdates.date2num(ns_to_dt(s)), mdates.date2num(ns_to_dt(e)) - mdates.date2num(ns_to_dt(s)))
            for s, e in gaps
        ]
        if spans:
            ax.broken_barh(spans, (row - 0.35, 0.7), facecolors=PALETTE["short"])
        labels.append(f"{stream}  {uptime_pct(gaps, t0, t1):.2f}%")
    ax.set_yticks(range(len(gaps_by_stream)), labels=labels)
    ax.set_xlim(x0, x1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.set_xlabel("UTC")
    ax.set_title("M1: stream uptime and gaps (green = recording, red = gap)")
    fig.autofmt_xdate()
    fig.tight_layout()
    return save_fig(fig, name, out_dir)
