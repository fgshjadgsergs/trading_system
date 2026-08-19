"""Single style/palette for every chart in the project."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless everywhere; png export only

import matplotlib.pyplot as plt
import seaborn as sns

from trading_system.core.config import reports_dir

PALETTE = {
    "long": "#2e7d32",
    "short": "#c62828",
    "neutral": "#455a64",
    "accent": "#f9a825",
    "heat": "magma",
    "grid": "#e0e0e0",
}


def apply_style() -> None:
    sns.set_theme(
        style="whitegrid",
        palette=[PALETTE["neutral"], PALETTE["long"], PALETTE["short"], PALETTE["accent"]],
        rc={
            "figure.figsize": (12, 6),
            "figure.dpi": 110,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "grid.color": PALETTE["grid"],
        },
    )


def save_fig(fig: plt.Figure, name: str, out_dir: Path | None = None) -> Path:
    """Save a figure as png under reports/ (or out_dir) and close it."""
    out = Path(out_dir) if out_dir is not None else reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
