"""Stage report builder: collect figure pngs + captions into reports/<stage>/README.md."""

from __future__ import annotations

import shutil
from pathlib import Path

from trading_system.core.config import reports_dir


def build_report(
    stage: str,
    figures: list[tuple[Path, str]],
    out_root: Path | None = None,
    intro: str = "",
) -> Path:
    """Copy figures under reports/<stage>/ and write an index README.md."""
    root = Path(out_root) if out_root is not None else reports_dir()
    stage_dir = root / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Отчёт: {stage}", ""]
    if intro:
        lines += [intro, ""]
    for path, caption in figures:
        path = Path(path)
        target = stage_dir / path.name
        if path.resolve() != target.resolve():
            shutil.copyfile(path, target)
        lines += [f"## {caption}", "", f"![{caption}]({path.name})", ""]
    index = stage_dir / "README.md"
    index.write_text("\n".join(lines), encoding="utf-8")
    return index
