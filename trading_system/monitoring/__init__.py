"""M11: monitoring.

Per-stream freshness tracking against configured age limits, gap alerting
with burst escalation, live-PnL tracking against the backtest expectation
envelope, alert sinks (structlog, JSONL file, Telegram via an injected
transport) with dedup/rate-limit, a Grafana dashboard asset with a structural
validator, and a scripted fire-drill that breaks every component on a fake
clock and checks each alert fires before a human would notice.
"""

from trading_system.monitoring.alerts import (
    Alert,
    AlertSink,
    DedupSink,
    FanoutSink,
    FileSink,
    ListSink,
    LogSink,
    Severity,
    TelegramSink,
    format_telegram_text,
)
from trading_system.monitoring.dashboard import (
    GRAFANA_DASHBOARD_PATH,
    load_dashboard,
    validate_dashboard,
)
from trading_system.monitoring.drill import DrillEvent, DrillResult, run_drill
from trading_system.monitoring.freshness import (
    FreshnessTracker,
    GapEvent,
    GapMonitor,
    StaleStream,
)
from trading_system.monitoring.pnl_tracker import PnlStatus, PnlTracker
from trading_system.monitoring.reports import demo_reports

__all__ = [
    "GRAFANA_DASHBOARD_PATH",
    "Alert",
    "AlertSink",
    "DedupSink",
    "DrillEvent",
    "DrillResult",
    "FanoutSink",
    "FileSink",
    "FreshnessTracker",
    "GapEvent",
    "GapMonitor",
    "ListSink",
    "LogSink",
    "PnlStatus",
    "PnlTracker",
    "Severity",
    "StaleStream",
    "TelegramSink",
    "demo_reports",
    "format_telegram_text",
    "load_dashboard",
    "run_drill",
    "validate_dashboard",
]
