"""M6: demo report figures exist and are non-trivial."""

from __future__ import annotations

from trading_system.spoof.reports import demo_reports


def test_demo_reports_produce_pngs(tmp_path):
    paths = demo_reports(tmp_path, seed=42)
    assert len(paths) == 2
    names = {p.name for p in paths}
    assert names == {"m6_stability_heatmap.png", "m6_lifetimes_exec_vs_cancel.png"}
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 5 * 1024
