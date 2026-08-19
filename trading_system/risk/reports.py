"""M9 demo reports: order state machine diagram and crash-drill incident log.

The state diagram is drawn programmatically from orders.ALLOWED_TRANSITIONS —
the exact table the machine validates against — so code and picture cannot
drift apart. The incident log replays a scripted crash drill: an order flow is
journaled, the process "dies" mid-order, and the restart reconciles the
journal against a deliberately divergent FakeExchange. No network, no wall
clock; everything is seeded and offline.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from trading_system.core.timeutils import NS_PER_S
from trading_system.risk.orders import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    OrderJournal,
    OrderState,
    OrderStateMachine,
    replay_journal,
)
from trading_system.risk.reconcile import (
    ExchangeOrder,
    ExchangePosition,
    FakeExchange,
    ReconcileReport,
    reconcile_on_restart,
)
from trading_system.viz.style import PALETTE, apply_style, save_fig

# --------------------------------------------------------------------------
# Figure (a): order state machine, drawn from ALLOWED_TRANSITIONS
# --------------------------------------------------------------------------

_STATE_POS: dict[OrderState, tuple[float, float]] = {
    OrderState.IDLE: (0.0, 2.0),
    OrderState.PENDING_NEW: (2.3, 2.0),
    OrderState.OPEN: (4.6, 3.1),
    OrderState.PARTIALLY_FILLED: (7.0, 3.1),
    OrderState.FILLED: (9.4, 2.0),
    OrderState.PENDING_CANCEL: (5.8, 0.9),
    OrderState.CANCELED: (3.5, 0.6),
    OrderState.REJECTED: (1.4, 0.6),
}

# Manual curvature per edge so arrows do not overlap; default is a light arc.
_EDGE_RAD: dict[tuple[OrderState, OrderState], float] = {
    (OrderState.PENDING_NEW, OrderState.FILLED): -0.35,
    (OrderState.PENDING_NEW, OrderState.PARTIALLY_FILLED): 0.25,
    (OrderState.OPEN, OrderState.FILLED): 0.2,
    (OrderState.OPEN, OrderState.CANCELED): -0.15,
    (OrderState.PARTIALLY_FILLED, OrderState.CANCELED): 0.25,
    (OrderState.PENDING_CANCEL, OrderState.FILLED): -0.2,
}

_BOX_W, _BOX_H = 1.9, 0.62


def _state_color(state: OrderState) -> str:
    if state is OrderState.FILLED:
        return PALETTE["long"]
    if state in TERMINAL_STATES:  # CANCELED / REJECTED
        return PALETTE["short"]
    if state in (OrderState.PENDING_NEW, OrderState.PENDING_CANCEL):
        return PALETTE["accent"]
    return PALETTE["neutral"]


def fig_state_machine(out_dir: Path) -> Path:
    """Draw the order state machine from the live transition table."""
    fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    ax.set_xlim(-1.3, 10.9)
    ax.set_ylim(-0.4, 4.3)
    ax.axis("off")

    n_edges = 0
    for src, targets in ALLOWED_TRANSITIONS.items():
        for dst in sorted(targets, key=str):
            n_edges += 1
            x0, y0 = _STATE_POS[src]
            if src == dst:  # self-loop above the box
                loop = FancyArrowPatch(
                    (x0 - 0.3, y0 + _BOX_H / 2),
                    (x0 + 0.3, y0 + _BOX_H / 2),
                    connectionstyle="arc3,rad=-1.9",
                    arrowstyle="-|>",
                    mutation_scale=13,
                    color=PALETTE["neutral"],
                    lw=1.2,
                    shrinkA=2,
                    shrinkB=2,
                )
                ax.add_patch(loop)
                continue
            x1, y1 = _STATE_POS[dst]
            rad = _EDGE_RAD.get((src, dst), 0.12)
            arrow = FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=14,
                color=_state_color(dst),
                alpha=0.75,
                lw=1.4,
                shrinkA=26,
                shrinkB=26,
            )
            ax.add_patch(arrow)

    for state, (x, y) in _STATE_POS.items():
        color = _state_color(state)
        box = FancyBboxPatch(
            (x - _BOX_W / 2, y - _BOX_H / 2),
            _BOX_W,
            _BOX_H,
            boxstyle="round,pad=0.06",
            linewidth=2.2 if state in TERMINAL_STATES else 1.4,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.annotate(
            str(state),
            (x, y),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )
        if state in TERMINAL_STATES:
            ax.annotate(
                "terminal",
                (x, y - _BOX_H / 2 - 0.14),
                ha="center",
                va="top",
                fontsize=8,
                color=color,
            )

    ax.set_title(
        f"M9 order state machine — {len(ALLOWED_TRANSITIONS)} states, "
        f"{n_edges} allowed transitions (drawn from orders.ALLOWED_TRANSITIONS)"
    )
    ax.annotate(
        "self-loops: repeated partial fills / fills while a cancel is in flight",
        (0.0, 4.05),
        fontsize=9,
        color=PALETTE["neutral"],
    )
    return save_fig(fig, "m9_order_state_machine", out_dir)


# --------------------------------------------------------------------------
# Figure (b): crash-drill incident log
# --------------------------------------------------------------------------

_T0 = 1_755_600_000 * NS_PER_S  # same synthetic day the rest of the repo uses


@dataclass(frozen=True, slots=True)
class _LogRow:
    t_s: float
    lane: str
    label: str
    color: str


def _run_crash_drill(journal_path: Path) -> tuple[list[_LogRow], ReconcileReport, float]:
    """Scripted order flow -> crash -> journal replay + reconciliation."""
    journal = OrderJournal(journal_path)
    rows: list[_LogRow] = []

    def ts(t_s: float) -> int:
        return _T0 + int(t_s * NS_PER_S)

    def log(t_s: float, lane: str, label: str, color: str) -> None:
        rows.append(_LogRow(t_s=t_s, lane=lane, label=label, color=color))

    a = OrderStateMachine("A-1", journal, symbol="BTCUSDT", side="BUY", qty=1.0, ts=ts(0.0))
    a.transition(OrderState.PENDING_NEW, ts(0.1))
    log(0.1, "A-1", "PENDING_NEW", PALETTE["accent"])
    a.transition(OrderState.OPEN, ts(0.4))
    log(0.4, "A-1", "OPEN", PALETTE["neutral"])
    a.transition(OrderState.PARTIALLY_FILLED, ts(1.2), fill_qty=0.4, fill_price=50_000.0)
    log(1.2, "A-1", "PART 0.4", PALETTE["long"])

    b = OrderStateMachine("B-2", journal, symbol="BTCUSDT", side="SELL", qty=0.5, ts=ts(1.5))
    b.transition(OrderState.PENDING_NEW, ts(1.5))
    log(1.5, "B-2", "PENDING_NEW", PALETTE["accent"])

    c = OrderStateMachine("C-3", journal, symbol="BTCUSDT", side="BUY", qty=0.6, ts=ts(1.8))
    c.transition(OrderState.PENDING_NEW, ts(1.8))
    log(1.8, "C-3", "PENDING_NEW", PALETTE["accent"])
    c.transition(OrderState.OPEN, ts(2.0))
    log(2.0, "C-3", "OPEN", PALETTE["neutral"])
    c.transition(OrderState.PENDING_CANCEL, ts(2.4))
    log(2.4, "C-3", "PENDING_CANCEL", PALETTE["accent"])

    crash_s = 2.6  # process dies here; in-memory machines are dropped
    del a, b, c

    # Exchange truth diverged while we were down.
    exchange = FakeExchange(
        position=ExchangePosition("BTCUSDT", 0.9, 50_010.0),
        open_orders=[
            ExchangeOrder("A-1", "BTCUSDT", "BUY", 1.0, 50_000.0, 0.7, "PARTIALLY_FILLED"),
            ExchangeOrder("C-3", "BTCUSDT", "BUY", 0.6, 49_900.0, 0.2, "PARTIALLY_FILLED"),
            ExchangeOrder("G-9", "BTCUSDT", "SELL", 0.3, 50_500.0, 0.0, "NEW"),
        ],
    )
    restart_s = 3.5
    replayed = replay_journal(journal_path)
    report = reconcile_on_restart(
        replayed,
        exchange,
        "BTCUSDT",
        ts=ts(restart_s),
        local_position_qty=0.4,
        journal=journal,
    )
    for i, action in enumerate(report.actions):
        lane = action.order_id or "position"
        log(restart_s + 0.12 * i, lane, action.kind, PALETTE["short"])
    return rows, report, crash_s


def fig_crash_drill(out_dir: Path) -> Path:
    """Incident log of the M9 crash drill as a lane timeline + findings."""
    with tempfile.TemporaryDirectory() as tmp:
        rows, report, crash_s = _run_crash_drill(Path(tmp) / "journal.jsonl")

    lanes = ["A-1", "B-2", "C-3", "G-9", "position"]
    lane_y = {lane: len(lanes) - 1 - i for i, lane in enumerate(lanes)}
    fig, (ax, ax_txt) = plt.subplots(
        2, 1, figsize=(13, 7.5), height_ratios=[2.1, 1], constrained_layout=True
    )

    for lane in lanes:
        ax.axhline(lane_y[lane], color=PALETTE["grid"], lw=0.8, zorder=0)
    for row in rows:
        if row.lane not in lane_y:
            continue
        y = lane_y[row.lane]
        ax.scatter(row.t_s, y, s=60, color=row.color, zorder=3)
        ax.annotate(
            row.label,
            (row.t_s, y),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            rotation=25,
            color=row.color,
        )
    restart_s = (report.ts - _T0) / NS_PER_S
    ax.axvline(crash_s, color=PALETTE["short"], lw=2.0, ls="--")
    ax.annotate(
        "CRASH: process killed mid-order ",
        (crash_s, len(lanes) - 0.45),
        ha="right",
        fontsize=9,
        color=PALETTE["short"],
        fontweight="bold",
    )
    ax.axvline(restart_s, color=PALETTE["long"], lw=2.0, ls="--")
    ax.annotate(
        " RESTART: journal replay + reconcile",
        (restart_s, len(lanes) - 0.45),
        ha="left",
        fontsize=9,
        color=PALETTE["long"],
        fontweight="bold",
    )
    ax.set_yticks([lane_y[lane] for lane in lanes], lanes)
    ax.set_xlabel("seconds since drill start")
    ax.set_ylim(-0.6, len(lanes))
    converged = "CONVERGED" if report.converged else "NOT CONVERGED"
    ax.set_title(
        f"M9 crash drill — journaled flow, kill, restart from journal vs divergent "
        f"exchange: {converged}"
    )

    ax_txt.axis("off")
    lines = [f"divergences found: {len(report.mismatches)}   corrective actions: {len(report.actions)}"]
    for m in report.mismatches:
        who = m.order_id or "position"
        lines.append(f"  mismatch [{m.kind}] {who}: local {m.local} vs exchange {m.exchange}")
    for a in report.actions:
        who = a.order_id or "position"
        lines.append(f"  action   [{a.kind}] {who}: {a.detail}")
    ax_txt.annotate(
        "\n".join(lines),
        (0.0, 1.0),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
    )
    return save_fig(fig, "m9_crash_drill_log", out_dir)


def demo_reports(out_dir: Path, seed: int = 42) -> list[Path]:
    """Generate the M9 checklist figures (deterministic; seed kept for API parity)."""
    apply_style()
    return [fig_state_machine(out_dir), fig_crash_drill(out_dir)]
