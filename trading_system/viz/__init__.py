"""M10: unified visualization style and report templates.

Convention: seaborn for all analytics; candles and price-overlay heatmaps via
mplfinance/plotly. Every figure is exported as png into reports/.
"""

from trading_system.viz.style import apply_style, save_fig

__all__ = ["apply_style", "save_fig"]
