"""M9: risk and execution.

Volatility-target position sizing (EWMA vol estimator, cap, exchange step
rounding), hard limits (sticky daily stop on UTC-day PnL, kill switch on
consecutive errors or stale market data), an order state machine with a
write-ahead JSONL journal, and restart reconciliation that converges the
journal-replayed local state to exchange truth. Everything is driven by
injected timestamps (UTC ns) — no wall clock, no network; the exchange sits
behind the ExchangeClient protocol (FakeExchange for tests and drills).
"""

from trading_system.risk.limits import (
    DailyStop,
    HaltState,
    KillState,
    KillSwitch,
    utc_day_key,
)
from trading_system.risk.orders import (
    ALLOWED_TRANSITIONS,
    OPEN_STATES,
    TERMINAL_STATES,
    InvalidTransition,
    JournalCorrupted,
    OrderJournal,
    OrderState,
    OrderStateMachine,
    ReplayedOrder,
    replay_journal,
)
from trading_system.risk.reconcile import (
    Action,
    ExchangeClient,
    ExchangeOrder,
    ExchangePosition,
    FakeExchange,
    Mismatch,
    ReconcileReport,
    reconcile_on_restart,
)
from trading_system.risk.reports import demo_reports
from trading_system.risk.sizing import (
    EwmaVol,
    SizeResult,
    VolTargetSizer,
    round_qty_to_step,
    vol_target_position_usd,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "OPEN_STATES",
    "TERMINAL_STATES",
    "Action",
    "DailyStop",
    "EwmaVol",
    "ExchangeClient",
    "ExchangeOrder",
    "ExchangePosition",
    "FakeExchange",
    "HaltState",
    "InvalidTransition",
    "JournalCorrupted",
    "KillState",
    "KillSwitch",
    "Mismatch",
    "OrderJournal",
    "OrderState",
    "OrderStateMachine",
    "ReconcileReport",
    "ReplayedOrder",
    "SizeResult",
    "VolTargetSizer",
    "demo_reports",
    "reconcile_on_restart",
    "replay_journal",
    "round_qty_to_step",
    "utc_day_key",
    "vol_target_position_usd",
]
