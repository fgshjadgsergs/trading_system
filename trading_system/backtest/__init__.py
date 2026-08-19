"""M8: event-driven backtester with realistic fills, latency, fees, funding."""

from trading_system.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    Bar,
    BookProvider,
    Context,
    Fill,
    Order,
    OrderType,
    Strategy,
    TradePrint,
    run_backtest,
)
from trading_system.backtest.fills import (
    LatencyModel,
    impact_bps,
    limit_crossed,
    market_fill_price,
    walk_book,
)
from trading_system.backtest.metrics import (
    cost_waterfall,
    hit_rate,
    max_drawdown,
    pnl_decomposition,
    summary,
    trades_from_fills,
)
from trading_system.backtest.reports import demo_reports
from trading_system.backtest.strategies import (
    MACrossStrategy,
    RandomStrategy,
    TargetPositionStrategy,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Bar",
    "BookProvider",
    "Context",
    "Fill",
    "LatencyModel",
    "MACrossStrategy",
    "Order",
    "OrderType",
    "RandomStrategy",
    "Strategy",
    "TargetPositionStrategy",
    "TradePrint",
    "cost_waterfall",
    "demo_reports",
    "hit_rate",
    "impact_bps",
    "limit_crossed",
    "market_fill_price",
    "max_drawdown",
    "pnl_decomposition",
    "run_backtest",
    "summary",
    "trades_from_fills",
    "walk_book",
]
