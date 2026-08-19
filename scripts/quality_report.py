"""Daily data-quality report (этап 1.1): uptime, gaps, latency histograms.

Reads the parquet lake, prints per-stream uptime and gap lists, and saves the
checklist figures (latency histograms per stream, gap/uptime timeline).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import structlog

from trading_system.collectors.quality import (
    daily_quality_report,
    plot_gap_timeline,
    plot_latency_histograms,
)
from trading_system.core.config import load_config
from trading_system.core.timeutils import NS_PER_S, dt_to_ns

log = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--lake", default="data/lake")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--date", default=None, help="UTC date YYYY-MM-DD (default: today)")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    cfg = load_config(args.config)
    date = (
        datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=UTC)
        if args.date
        else datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    )
    t0 = dt_to_ns(date)
    t1 = t0 + 86_400 * NS_PER_S
    # config keys are ws-stream names; the lake uses unified stream names
    key_map = {
        "agg_trade": "trade",
        "depth": "depth_diff",
        "force_order": "liquidation",
        "kline_1m": "kline",
    }
    limits = {
        key_map.get(k, k): float(v) for k, v in cfg["monitoring"]["freshness_limits_s"].items()
    }
    report = daily_quality_report(
        Path(args.lake), "binance_usdm", args.symbol, t0, t1, max_silence_s=limits
    )
    out = Path(args.out)
    latencies = {s: q.latency_ms for s, q in report.streams.items()}
    gaps_by_stream = {s: q.silence_gaps for s, q in report.streams.items()}
    figs = [
        plot_latency_histograms(latencies, out_dir=out),
        plot_gap_timeline(gaps_by_stream, t0, t1, out_dir=out),
    ]
    for s, q in report.streams.items():
        log.info(
            "stream_quality",
            stream=s,
            records=q.n_records,
            uptime_pct=round(q.uptime_pct, 3),
            gaps=len(q.silence_gaps),
        )
    log.info("quality_report_done", clean=report.clean, figures=[str(f) for f in figs])


if __name__ == "__main__":
    main()
