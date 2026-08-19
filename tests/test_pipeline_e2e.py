"""Gate 2 smoke: the full pipeline runs as one script with no manual steps."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_pipeline_end_to_end(tmp_path):
    out = tmp_path / "reports"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "run_pipeline.py"),
            "--out",
            str(out),
            "--n-trades",
            "60000",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=420,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = out / "pipeline" / "README.md"
    assert report.exists()
    pngs = list((out / "pipeline").glob("*.png"))
    assert len(pngs) >= 5
    for p in pngs:
        assert p.stat().st_size > 5_000
