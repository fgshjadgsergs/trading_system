"""M1: demo_reports generates every checklist figure from synthetic data."""

from __future__ import annotations

from trading_system.collectors.reports import demo_reports


def test_demo_reports_generate_all_figures(tmp_path):
    paths = demo_reports(tmp_path, seed=42)
    assert len(paths) == 2
    names = {p.name for p in paths}
    assert names == {"m1_latency_hist.png", "m1_gap_timeline.png"}
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 5 * 1024  # non-trivial png
    # temp demo lake is cleaned up; only figures remain in out_dir
    assert not list(tmp_path.glob("_demo_lake_*"))


def test_demo_reports_deterministic_across_runs(tmp_path):
    a = demo_reports(tmp_path / "a", seed=7)
    b = demo_reports(tmp_path / "b", seed=7)
    assert [p.name for p in a] == [p.name for p in b]
    for pa, pb in zip(a, b, strict=True):
        assert pa.read_bytes() == pb.read_bytes()
