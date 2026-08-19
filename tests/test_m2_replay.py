"""M2 replay tests: golden replay vs naive reference, frames path, grid sampling."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest

from trading_system.book import BookReplayer, stream_frames
from trading_system.core.synth import synth_book_stream
from trading_system.core.timeutils import NS_PER_S

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "m2"
GOLDEN_HASH_FILE = FIXTURE_DIR / "golden_top50.sha256"


def naive_final_state(stream, top: int = 50):
    """Independent reference: plain-dict application of the diff stream."""
    bids = {p: q for p, q in stream.snapshot.bids}
    asks = {p: q for p, q in stream.snapshot.asks}
    last = stream.snapshot.last_update_id
    for d in stream.diffs:
        if d.final_update_id < last:
            continue
        for p, q in d.bids:
            if q == 0.0:
                bids.pop(p, None)
            else:
                bids[p] = q
        for p, q in d.asks:
            if q == 0.0:
                asks.pop(p, None)
            else:
                asks[p] = q
        last = d.final_update_id
    top_bids = tuple(sorted(bids.items(), key=lambda x: -x[0])[:top])
    top_asks = tuple(sorted(asks.items())[:top])
    return top_bids, top_asks


def top_digest(bids, asks) -> str:
    payload = "|".join(
        f"{tag}:{p!r}:{q!r}"
        for tag, levels in (("B", bids), ("A", asks))
        for p, q in levels
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class TestGoldenReplay:
    def test_final_state_matches_naive_reference(self):
        stream = synth_book_stream(n_diffs=2000, seed=42)
        book = BookReplayer(stream.snapshot, stream.diffs).final_book()
        assert book.top_n(50) == naive_final_state(stream, top=50)
        assert book.last_update_id == stream.diffs[-1].final_update_id

    def test_final_top50_hash_matches_fixture(self):
        stream = synth_book_stream(n_diffs=2000, seed=42)
        book = BookReplayer(stream.snapshot, stream.diffs).final_book()
        bids, asks = book.top_n(50)
        expected = GOLDEN_HASH_FILE.read_text().strip()
        assert top_digest(bids, asks) == expected

    def test_replay_yields_after_each_applied_diff(self):
        stream = synth_book_stream(n_diffs=200, seed=7)
        seen_ts = []
        for ts, book in BookReplayer(stream.snapshot, stream.diffs).replay():
            # the yielded book is live: its state is as of the yielded ts
            assert book.ts_event == ts
            seen_ts.append(ts)
        assert seen_ts == [d.ts_event for d in stream.diffs]


class TestFramesPath:
    def test_frame_replay_equals_dataclass_replay(self):
        stream = synth_book_stream(n_diffs=500, seed=3)
        snap_df, diff_df = stream_frames(stream.snapshot, stream.diffs)
        from_frames = BookReplayer.from_frames(snap_df, diff_df).final_book()
        from_records = BookReplayer(stream.snapshot, stream.diffs).final_book()
        assert from_frames.top_n(1000) == from_records.top_n(1000)
        assert from_frames.last_update_id == from_records.last_update_id

    def test_empty_snapshot_frame_rejected(self):
        stream = synth_book_stream(n_diffs=10, seed=1)
        snap_df, diff_df = stream_frames(stream.snapshot, stream.diffs)
        with pytest.raises(ValueError):
            BookReplayer.from_frames(snap_df.clear(), diff_df)


class TestStateAt:
    def test_state_at_matches_prefix_replay(self):
        stream = synth_book_stream(n_diffs=300, seed=11)
        replayer = BookReplayer(stream.snapshot, stream.diffs)
        k = 137
        t = stream.diffs[k].ts_event
        prefix = BookReplayer(stream.snapshot, stream.diffs[: k + 1]).final_book()
        assert replayer.state_at(t, n=25) == prefix.top_n(25)

    def test_state_at_snapshot_time_is_snapshot(self):
        stream = synth_book_stream(n_diffs=50, seed=5)
        replayer = BookReplayer(stream.snapshot, stream.diffs)
        bids, asks = replayer.state_at(stream.snapshot.ts_event, n=10)
        assert bids == stream.snapshot.bids[:10]
        assert asks == stream.snapshot.asks[:10]

    def test_state_before_snapshot_rejected(self):
        stream = synth_book_stream(n_diffs=10, seed=5)
        replayer = BookReplayer(stream.snapshot, stream.diffs)
        with pytest.raises(ValueError):
            replayer.state_at(stream.snapshot.ts_event - 1)


class TestSampleGrid:
    def test_grid_shape_and_alignment(self):
        stream = synth_book_stream(n_diffs=400, seed=9)
        replayer = BookReplayer(stream.snapshot, stream.diffs)
        interval = 5 * NS_PER_S
        grid = replayer.sample_grid(interval_ns=interval, n=20)
        assert grid.columns == ["ts", "side", "price", "qty"]
        ts_vals = grid["ts"].unique().sort().to_list()
        t0 = stream.snapshot.ts_event
        end = stream.diffs[-1].ts_event
        expected_ts = list(range(t0, end + 1, interval))
        assert ts_vals == expected_ts
        assert set(grid["side"].unique().to_list()) == {"bid", "ask"}
        assert (grid["qty"] > 0).all()

    def test_grid_rows_match_state_at(self):
        stream = synth_book_stream(n_diffs=400, seed=9)
        replayer = BookReplayer(stream.snapshot, stream.diffs)
        interval = 7 * NS_PER_S
        grid = replayer.sample_grid(interval_ns=interval, n=15)
        t = grid["ts"].unique().sort().to_list()[3]
        bids, asks = replayer.state_at(t, n=15)
        sub = grid.filter(pl.col("ts") == t)
        got_bids = [
            (r["price"], r["qty"]) for r in sub.filter(pl.col("side") == "bid").iter_rows(named=True)
        ]
        got_asks = [
            (r["price"], r["qty"]) for r in sub.filter(pl.col("side") == "ask").iter_rows(named=True)
        ]
        assert tuple(got_bids) == bids
        assert tuple(got_asks) == asks

    def test_sample_metrics_columns_and_positive_depth(self):
        stream = synth_book_stream(n_diffs=400, seed=9)
        replayer = BookReplayer(stream.snapshot, stream.diffs)
        metrics = replayer.sample_metrics(interval_ns=5 * NS_PER_S, depth_pct=0.005)
        assert metrics.columns == ["ts", "mid", "spread", "bid_depth", "ask_depth"]
        assert (metrics["spread"] > 0).all()
        assert (metrics["bid_depth"] >= 0).all()
        assert (metrics["ask_depth"] >= 0).all()

    def test_bad_interval_rejected(self):
        stream = synth_book_stream(n_diffs=10, seed=2)
        replayer = BookReplayer(stream.snapshot, stream.diffs)
        with pytest.raises(ValueError):
            replayer.sample_grid(interval_ns=0)
