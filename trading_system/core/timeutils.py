"""UTC-nanosecond time helpers. All timestamps in the system are UTC ns."""

from __future__ import annotations

import time
from datetime import UTC, datetime

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000
NS_PER_MIN = 60 * NS_PER_S

TIMEFRAME_NS: dict[str, int] = {
    "1s": NS_PER_S,
    "1m": 60 * NS_PER_S,
    "5m": 300 * NS_PER_S,
    "15m": 900 * NS_PER_S,
    "1h": 3_600 * NS_PER_S,
    "4h": 14_400 * NS_PER_S,
    "1d": 86_400 * NS_PER_S,
}


def now_ns() -> int:
    return time.time_ns()


def ms_to_ns(ms: int | float) -> int:
    return int(ms * NS_PER_MS)


def ns_to_dt(ts: int) -> datetime:
    return datetime.fromtimestamp(ts / NS_PER_S, tz=UTC)


def dt_to_ns(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise ValueError("naive datetime: all system time is explicit UTC")
    return int(dt.timestamp() * NS_PER_S)


def floor_ts(ts: int, timeframe: str) -> int:
    """Bar-open timestamp of the timeframe bucket containing ts."""
    step = TIMEFRAME_NS[timeframe]
    return ts - ts % step


def date_str(ts: int) -> str:
    return ns_to_dt(ts).strftime("%Y-%m-%d")
