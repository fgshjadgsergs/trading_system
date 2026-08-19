"""M10: unified visualization style and report templates.

Convention: seaborn for all analytics; candles and price-overlay heatmaps via
mplfinance/plotly. Every figure is exported as png into reports/.
"""

from trading_system.viz.overlay import overlay_chart
from trading_system.viz.report import build_report
from trading_system.viz.style import apply_style, save_fig
from trading_system.viz.templates import (
    calibration_curve,
    corr_heatmap,
    dist_plot,
    event_study_plot,
)

__all__ = [
    "apply_style",
    "build_report",
    "calibration_curve",
    "corr_heatmap",
    "dist_plot",
    "event_study_plot",
    "overlay_chart",
    "save_fig",
]
