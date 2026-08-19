"""M11 alert sinks: dedup/rate-limit, Telegram payload via fake transport, files."""

from __future__ import annotations

import json

import pytest

from trading_system.monitoring.alerts import (
    Alert,
    DedupSink,
    FanoutSink,
    FileSink,
    ListSink,
    LogSink,
    Severity,
    TelegramSink,
    format_telegram_text,
)

NS_PER_S = 1_000_000_000
T0 = 1_755_600_000 * NS_PER_S  # 2025-08-19 10:40:00 UTC


def alert(t_s: float = 0.0, sev: Severity = Severity.WARNING, key: str = "k") -> Alert:
    return Alert(
        severity=sev,
        source="freshness",
        message="stream depth stale: 6.0s > 5s limit",
        ts=T0 + int(t_s * NS_PER_S),
        key=key,
    )


def test_dedup_key_defaults_to_severity_source():
    a = Alert(severity=Severity.INFO, source="pnl", message="m", ts=T0)
    assert a.dedup_key == "info:pnl"
    assert alert(key="custom").dedup_key == "custom"


# --------------------------------------------------------------------------
# DedupSink
# --------------------------------------------------------------------------


def test_dedup_suppresses_repeats_within_cooldown():
    inner = ListSink()
    dedup = DedupSink(inner, cooldown_s=300.0)
    assert dedup.emit(alert(0.0)) is True
    assert dedup.emit(alert(10.0)) is False  # same key, inside cooldown
    assert dedup.emit(alert(299.9)) is False
    assert dedup.emit(alert(300.0)) is True  # cooldown expired
    assert len(inner.alerts) == 2
    assert dedup.suppressed == 2


def test_dedup_different_keys_pass_independently():
    inner = ListSink()
    dedup = DedupSink(inner, cooldown_s=300.0)
    assert dedup.emit(alert(0.0, key="a"))
    assert dedup.emit(alert(1.0, key="b"))
    assert not dedup.emit(alert(2.0, key="a"))
    assert len(inner.alerts) == 2


def test_dedup_reset_clears_history():
    dedup = DedupSink(ListSink(), cooldown_s=300.0)
    dedup.emit(alert(0.0))
    dedup.emit(alert(1.0))
    dedup.reset()
    assert dedup.suppressed == 0
    assert dedup.emit(alert(2.0)) is True


def test_dedup_bad_cooldown_raises():
    with pytest.raises(ValueError):
        DedupSink(ListSink(), cooldown_s=0.0)


# --------------------------------------------------------------------------
# TelegramSink — payload formatted correctly, delivered to a fake transport
# --------------------------------------------------------------------------


def test_telegram_payload_is_a_correct_sendmessage_call():
    sent: list[tuple[str, dict]] = []
    sink = TelegramSink("123:ABC", "-100200300", transport=lambda url, p: sent.append((url, p)))
    a = alert(sev=Severity.CRITICAL)
    sink.emit(a)

    assert len(sent) == 1
    url, payload = sent[0]
    assert url == "https://api.telegram.org/bot123:ABC/sendMessage"
    assert payload["chat_id"] == "-100200300"
    assert payload["disable_web_page_preview"] is True
    assert payload["disable_notification"] is False  # CRITICAL must ping
    assert payload["text"] == format_telegram_text(a)
    assert payload["text"].startswith("[CRIT] freshness:")
    assert "stream depth stale" in payload["text"]
    assert "2025-08-19 10:40:00 UTC" in payload["text"]  # ts rendered in UTC
    json.dumps(payload)  # JSON-serializable as the Bot API requires


def test_telegram_info_is_silent_warning_pings():
    sent: list[dict] = []
    sink = TelegramSink("t", "c", transport=lambda url, p: sent.append(p))
    sink.emit(alert(sev=Severity.INFO))
    sink.emit(alert(sev=Severity.WARNING))
    assert sent[0]["disable_notification"] is True
    assert sent[1]["disable_notification"] is False
    assert sent[0]["text"].startswith("[INFO]")
    assert sent[1]["text"].startswith("[WARN]")


# --------------------------------------------------------------------------
# Other sinks
# --------------------------------------------------------------------------


def test_file_sink_appends_parseable_jsonl(tmp_path):
    path = tmp_path / "alerts" / "log.jsonl"
    sink = FileSink(path)
    sink.emit(alert(0.0))
    sink.emit(alert(1.0, sev=Severity.CRITICAL))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["severity"] == "warning" and rows[1]["severity"] == "critical"
    assert rows[0]["source"] == "freshness"
    assert rows[0]["ts"] == T0


def test_fanout_delivers_to_every_sink():
    a_sink, b_sink = ListSink(), ListSink()
    FanoutSink([a_sink, b_sink]).emit(alert())
    assert len(a_sink.alerts) == 1 and len(b_sink.alerts) == 1


class RecordingLogger:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def info(self, msg, **kw):
        self.calls.append(("info", msg))

    def warning(self, msg, **kw):
        self.calls.append(("warning", msg))

    def error(self, msg, **kw):
        self.calls.append(("error", msg))


def test_log_sink_routes_by_severity():
    logger = RecordingLogger()
    sink = LogSink(logger)
    sink.emit(alert(sev=Severity.INFO))
    sink.emit(alert(sev=Severity.WARNING))
    sink.emit(alert(sev=Severity.CRITICAL))
    assert [level for level, _ in logger.calls] == ["info", "warning", "error"]
