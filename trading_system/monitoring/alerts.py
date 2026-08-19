"""Alerts: dataclass, sinks (structlog, JSONL file, Telegram) and dedup.

TelegramSink only FORMATS the Bot API sendMessage call and hands it to an
injectable transport — no network is ever touched from this module; tests and
drills inject a recording fake.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import structlog

from trading_system.core.timeutils import NS_PER_S, ns_to_dt


class Severity(enum.StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Alert:
    severity: Severity
    source: str  # component that raised it, e.g. "freshness", "gaps", "pnl"
    message: str
    ts: int  # UTC ns
    key: str = ""  # dedup key; defaults to severity:source

    @property
    def dedup_key(self) -> str:
        return self.key or f"{self.severity}:{self.source}"


class AlertSink(Protocol):
    """Anything that can receive an alert."""

    def emit(self, alert: Alert) -> Any: ...


class ListSink:
    """Collects alerts in memory; the sink used by tests and drills."""

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        self.alerts.append(alert)


class LogSink:
    """Routes alerts to structlog at a level matching their severity."""

    def __init__(self, logger: Any | None = None) -> None:
        self._log = logger if logger is not None else structlog.get_logger("alerts")

    def emit(self, alert: Alert) -> None:
        method = {
            Severity.INFO: self._log.info,
            Severity.WARNING: self._log.warning,
            Severity.CRITICAL: self._log.error,
        }[alert.severity]
        method(alert.message, source=alert.source, severity=str(alert.severity), ts=alert.ts)


class FileSink:
    """Appends alerts as JSON lines to a file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, alert: Alert) -> None:
        row = asdict(alert)
        row["severity"] = str(alert.severity)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def format_telegram_text(alert: Alert) -> str:
    """Human-readable one-liner for Telegram."""
    stamp = ns_to_dt(alert.ts).strftime("%Y-%m-%d %H:%M:%S UTC")
    badge = {Severity.INFO: "INFO", Severity.WARNING: "WARN", Severity.CRITICAL: "CRIT"}[
        alert.severity
    ]
    return f"[{badge}] {alert.source}: {alert.message} ({stamp})"


class TelegramSink:
    """Formats a Bot API sendMessage request; delivery is the transport's job.

    transport(url, payload) is injected — offline by construction. The real
    transport (requests/aiohttp POST) is wired in ops code, never here.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        transport: Callable[[str, dict], Any],
    ) -> None:
        self.url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.chat_id = chat_id
        self._transport = transport

    def emit(self, alert: Alert) -> None:
        payload = {
            "chat_id": self.chat_id,
            "text": format_telegram_text(alert),
            "disable_web_page_preview": True,
            "disable_notification": alert.severity is Severity.INFO,
        }
        self._transport(self.url, payload)


class DedupSink:
    """Suppresses repeats of the same dedup_key within a cooldown window.

    Time comes from alert.ts (event time), never the wall clock, so behavior
    is deterministic offline. emit() returns True when delivered downstream.
    """

    def __init__(self, inner: AlertSink, cooldown_s: float) -> None:
        if cooldown_s <= 0:
            raise ValueError("cooldown_s must be positive")
        self.inner = inner
        self.cooldown_ns = int(cooldown_s * NS_PER_S)
        self._last_ts: dict[str, int] = {}
        self.suppressed = 0

    def emit(self, alert: Alert) -> bool:
        key = alert.dedup_key
        last = self._last_ts.get(key)
        if last is not None and alert.ts - last < self.cooldown_ns:
            self.suppressed += 1
            return False
        self._last_ts[key] = alert.ts
        self.inner.emit(alert)
        return True

    def reset(self) -> None:
        self._last_ts.clear()
        self.suppressed = 0


class FanoutSink:
    """Delivers each alert to every child sink."""

    def __init__(self, sinks: list[AlertSink]) -> None:
        self.sinks = list(sinks)

    def emit(self, alert: Alert) -> None:
        for s in self.sinks:
            s.emit(alert)
