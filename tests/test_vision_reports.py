"""Demo report figures for checklist section 1.2 (виз)."""

from __future__ import annotations

from trading_system.collectors.vision import demo_reports


def test_demo_reports_generates_nontrivial_pngs(tmp_path):
    paths = demo_reports(tmp_path, seed=42)
    assert [p.name for p in paths] == [
        "vision_catalog_coverage.png",
        "vision_catalog_rows.png",
        "vision_reconcile_reldiff.png",
    ]
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 5 * 1024, f"{p.name} too small"
        assert p.parent == tmp_path


def test_demo_reports_deterministic_across_seeds(tmp_path):
    a = demo_reports(tmp_path / "a", seed=42)
    b = demo_reports(tmp_path / "b", seed=42)
    assert [p.read_bytes() for p in a] == [p.read_bytes() for p in b]
