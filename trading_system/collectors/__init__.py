"""M1: exchange collectors — websocket/REST ingest, gap detection, batch parquet writer.

Public API:
    BinanceUsdmAdapter        — Binance USDT-M ExchangeAdapter implementation
    ReconnectingWSClient      — reconnect/backoff/heartbeat message pump
    DepthSequencer, GapEvent  — U/u/pu book-sync gatekeeper
    BatchWriter, RestPoller   — parquet batch recording + generic REST polling
    daily_quality_report      — uptime/gaps/latency report over the lake
    demo_reports              — checklist figures from synthetic data
"""

from trading_system.collectors.binance import BinanceUsdmAdapter
from trading_system.collectors.quality import (
    QualityReport,
    StreamQuality,
    daily_quality_report,
)
from trading_system.collectors.recorder import BatchWriter, RestPoller
from trading_system.collectors.reports import demo_reports
from trading_system.collectors.sequencer import DepthSequencer, GapEvent
from trading_system.collectors.ws_client import ConnectionEvent, ReconnectingWSClient

__all__ = [
    "BatchWriter",
    "BinanceUsdmAdapter",
    "ConnectionEvent",
    "DepthSequencer",
    "GapEvent",
    "QualityReport",
    "ReconnectingWSClient",
    "RestPoller",
    "StreamQuality",
    "daily_quality_report",
    "demo_reports",
]
