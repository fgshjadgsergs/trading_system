"""Адверсариальная батарея для спуфинг-детектора M6 (сценарии A-K).

Цель — не «работает/не работает», а числовые ГРАНИЦЫ детектора
(lifecycle -> metrics -> score): при каких параметрах адверсариальная стена
перестаёт отличаться от честной. Генераторы сценариев сидированы и следуют
конвенциям spoof/synth.py (фон на нечётных тиках, плант-паттерны на чётных,
принты за полкаданса до состояния). Формула скоринга:

    base  = (w_life*pct + w_fill*fill + w_ice*r/(r+1)) / SUM(w),  w = 0.25/0.50/0.25
    score = base * 0.5 ** (flicker_count / 1.0)

Ключевые найденные границы (см. tests/stress/test_stress_m6_report.py):
- n_flick* = 0 по score (одна отмена уже даёт score ~0.11 < 0.4), флаг flicker
  с n_flick >= 2 (3 рождения при k=3);
- p* не существует: «терпеливый» спуфер (без подкормки) не проходит даже на
  p99 времени жизни честных (score <= ~0.12) — сильная сторона;
- f* ~ 0.40 (без рефиллов) для гейта 0.4; обрыв x2 ровно на max_flicker_fill=0.2;
- минимальная стоимость маскировки: f чуть выше 0.2 + мгновенные рефиллы ->
  score выше минимума честных и ярлык iceberg — слепое пятно (задокументировано).

Дизайн-слабости НЕ исправляются — фиксируются ассертами на фактическое
поведение с комментарием и попадают в отчёт reports/stress-m6/.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from functools import cache

import numpy as np
import polars as pl
import pytest

from trading_system.core.schema import Side, Trade
from trading_system.core.synth import EXCHANGE
from trading_system.core.timeutils import NS_PER_MS, NS_PER_S
from trading_system.spoof.lifecycle import BookState, LevelJournal
from trading_system.spoof.metrics import annotate_episodes
from trading_system.spoof.score import W_LIFE, score_episodes
from trading_system.spoof.synth import T0, TRUTH_SCHEMA, evaluate_spoof_flags
from trading_system.spoof.walls import ZONE_SCHEMA, merge_zones, wall_zones_at

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))

CADENCE_MS = 200
CAD_NS = CADENCE_MS * NS_PER_MS
HALF = CAD_NS // 2
MID = 50_000.0
TICK = 1.0
LARGE_K = 3.0
FLICKER_K = 3
# Операционный порог "стена настоящая": walls.wall_zones_at(min_score=0.4).
SCORE_GATE = 0.4


def _ts(i: int) -> int:
    return T0 + i * CAD_NS


# -- генераторы сценариев (конвенции spoof/synth.py) --------------------------


@dataclass
class _Plant:
    pattern_id: int
    label: str
    side: str
    price: float
    qty_at: dict[int, float] = field(default_factory=dict)
    trades: list[tuple[int, float]] = field(default_factory=list)

    @property
    def ts_start(self) -> int:
        return _ts(min(self.qty_at))

    @property
    def ts_end(self) -> int:
        return _ts(max(self.qty_at) + 1)


def plant_honest(pid, side, price, i0, hold, q=6.0, n_bites=4) -> _Plant:
    """Честная стена: стоит hold состояний, съедается принтами до нуля."""
    p = _Plant(pid, "honest", side, price)
    for j in range(hold):
        p.qty_at[i0 + j] = q
    bite = q / n_bites
    for b in range(n_bites):
        i = i0 + hold + b
        left = q - bite * (b + 1)
        if left > 1e-9:
            p.qty_at[i] = left
        p.trades.append((_ts(i) - HALF, bite))
    return p


def plant_flicker(pid, side, price, i0, n_flick, live=5, gap=10, q=6.0) -> _Plant:
    """Мерцающий спуф: n_flick+1 появлений, каждое снимается без принтов."""
    p = _Plant(pid, "spoof", side, price)
    cur = i0
    for _ in range(n_flick + 1):
        for j in range(live):
            p.qty_at[cur + j] = q
        cur += live + gap
    return p


def plant_patient(pid, side, price, i0, hold, q=6.0) -> _Plant:
    """«Терпеливый» спуф: одно появление, долго стоит, снимается без принтов."""
    p = _Plant(pid, "spoof", side, price)
    for j in range(hold):
        p.qty_at[i0 + j] = q
    return p


def plant_feeder(pid, side, price, i0, hold, f, n_bites=4, q=6.0, refill=False) -> _Plant:
    """Спуф с подкормкой: даёт исполниться доле f объёма, остаток снимает.

    refill=True — после каждого укуса мгновенно (следующее состояние, < 300 мс)
    восстанавливает объём: маскировка под айсберг; итоговый fill_frac == f.
    """
    p = _Plant(pid, "spoof", side, price)
    feed_total = q * f / (1.0 - f) if refill else q * f
    bite = feed_total / n_bites if n_bites else 0.0
    feed_states = set(np.linspace(3, hold - 3, n_bites).astype(int)) if n_bites else set()
    cur_q = q
    for j in range(hold):
        i = i0 + j
        if j in feed_states and bite > 0:
            if not refill:
                cur_q -= bite
            p.qty_at[i] = (cur_q - bite) if refill else cur_q
            p.trades.append((_ts(i) - HALF, bite))
        else:
            p.qty_at[i] = cur_q
    return p


def plant_iceberg(pid, side, price, i0, r, q=6.0, warmup=15, refill_lag=1) -> _Plant:
    """Честный айсберг: r циклов «принт съедает 75% -> рефилл», финал — добит.

    refill_lag=1 — рефилл на следующем состоянии (200 мс < окна 300 мс);
    refill_lag=2 — «поздний» рефилл (400 мс), цепочка рефиллов не строится.
    """
    p = _Plant(pid, "iceberg", side, price)
    for j in range(warmup):
        p.qty_at[i0 + j] = q
    i = i0 + warmup
    for _ in range(r):
        p.qty_at[i] = q * 0.25
        p.trades.append((_ts(i) - HALF, q * 0.75))
        for j in range(refill_lag, refill_lag + 2):
            p.qty_at[i + j] = q
        i += refill_lag + 2
    p.trades.append((_ts(i) - HALF, q))
    return p


def build_session(
    plants: list[_Plant],
    n_states: int,
    seed: int,
    n_bg: int = 29,
    bg_sigma: float = 0.3,
    churn: float = 0.0,
    qty_cap: float | None = None,
) -> tuple[list[BookState], list[Trade]]:
    """Фон на нечётных тиках (qty ~ логнормаль) + планты; сидированно."""
    rng = np.random.default_rng(seed)

    def draw_qty() -> float:
        q = float(rng.lognormal(0.0, bg_sigma))
        return min(q, qty_cap) if qty_cap is not None else q

    bg = {
        side: {float(MID + s * off * TICK): draw_qty() for off in range(1, 2 * n_bg, 2)}
        for side, s in (("bid", -1), ("ask", 1))
    }
    trades: list[Trade] = []
    tid = 0
    for p in plants:
        taker = Side.SELL if p.side == "bid" else Side.BUY
        for t, qq in p.trades:
            tid += 1
            trades.append(
                Trade(EXCHANGE, "BTCUSDT", t, t + NS_PER_MS, p.price, qq, qq * p.price, taker, tid)
            )
    trades.sort(key=lambda t: t.ts_event)
    bg_prices = {s: list(v) for s, v in bg.items()}
    states: list[BookState] = []
    for i in range(n_states):
        for _ in range(3):  # лёгкий джиттер/чёрн нескольких фоновых уровней
            side = "bid" if rng.random() < 0.5 else "ask"
            pr = bg_prices[side][int(rng.integers(0, len(bg_prices[side])))]
            if churn > 0 and rng.random() < churn:
                bg[side][pr] = 0.0 if bg[side][pr] > 0 else draw_qty()
            elif bg[side][pr] > 0:  # выключенный уровень не воскрешает джиттер
                q = max(0.05, bg[side][pr] * float(rng.uniform(0.9, 1.1)))
                bg[side][pr] = min(q, qty_cap) if qty_cap is not None else q
        levels = {
            "bid": [(p, q) for p, q in bg["bid"].items() if q > 0],
            "ask": [(p, q) for p, q in bg["ask"].items() if q > 0],
        }
        for p in plants:
            q = p.qty_at.get(i)
            if q is not None:
                levels[p.side].append((p.price, q))
        states.append(
            BookState(
                ts=_ts(i),
                bids=tuple(sorted(levels["bid"], key=lambda x: -x[0])),
                asks=tuple(sorted(levels["ask"])),
            )
        )
    return states, trades


def run_detector(states, trades, large_k=LARGE_K, flicker_k=FLICKER_K, record_grid=False):
    """Полный проход детектора: lifecycle -> metrics -> score."""
    j = LevelJournal(large_k=large_k, iceberg_refill_ms=300, record_grid=record_grid)
    j.run(states, trades)
    ann = annotate_episodes(j, flicker_k=flicker_k, flicker_window_s=60)
    return j, score_episodes(ann)


def truth_frame(plants: list[_Plant]) -> pl.DataFrame:
    return pl.DataFrame(
        [(p.pattern_id, p.label, p.side, p.price, p.ts_start, p.ts_end) for p in plants],
        schema=TRUTH_SCHEMA,
        orient="row",
    )


def plant_price(side: str, slot: int) -> float:
    """Чётные тиковые оффсеты для плантов (не пересекаются с фоном)."""
    return MID + (-1 if side == "bid" else 1) * (4 + 2 * slot) * TICK


def wall_score(scored: pl.DataFrame, price: float) -> float | None:
    rows = scored.filter((pl.col("price") == price) & pl.col("was_large"))
    return float(rows["score"].max()) if not rows.is_empty() else None


def _honest_population(rng, n=20, i0_lo=10, i0_hi=40):
    """Список (plant, hold) честных стен на чередующихся сторонах."""
    plants, holds = [], []
    slot = {"bid": 0, "ask": 0}
    for i in range(n):
        side = "bid" if i % 2 == 0 else "ask"
        pr = plant_price(side, slot[side])
        slot[side] += 1
        hold = int(rng.integers(50, 150))
        holds.append(hold)
        plants.append(
            plant_honest(i, side, pr, int(rng.integers(i0_lo, i0_hi)), hold=hold)
        )
    return plants, holds, slot


def _adversary_slots(slot, k):
    side = "bid" if k % 2 == 0 else "ask"
    pr = plant_price(side, slot[side])
    slot[side] += 1
    return side, pr


# -- батареи (кэшируются; их же использует test_stress_m6_report) -------------


@cache
def battery_populations() -> dict:
    """A: 200 честных против 200 мерцающих спуфов (батчами по 20+20)."""
    n_batches = max(2, round(10 * SCALE))
    agg = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    honest_scores: list[float] = []
    spoof_scores: list[float] = []
    for b in range(n_batches):
        plants, states, trades = _population_batch(seed=100 + b)
        _, scored = run_detector(states, trades)
        res = evaluate_spoof_flags(truth_frame(plants), scored, flag="flicker")
        for k in agg:
            agg[k] += res[k]
        for p in plants:
            s = wall_score(scored, p.price)
            if s is not None:
                (honest_scores if p.label == "honest" else spoof_scores).append(s)
    precision = agg["tp"] / (agg["tp"] + agg["fp"]) if agg["tp"] + agg["fp"] else 1.0
    recall = agg["tp"] / (agg["tp"] + agg["fn"]) if agg["tp"] + agg["fn"] else 1.0
    return {
        "precision": precision,
        "recall": recall,
        **agg,
        "honest_scores": honest_scores,
        "spoof_scores": spoof_scores,
    }


def _population_batch(seed: int, n_honest: int = 20, n_spoof: int = 20):
    rng = np.random.default_rng(seed)
    plants: list[_Plant] = []
    slot = {"bid": 0, "ask": 0}
    for pid, kind in enumerate(["honest"] * n_honest + ["spoof"] * n_spoof):
        side = "bid" if pid % 2 == 0 else "ask"
        pr = plant_price(side, slot[side])
        slot[side] += 1
        i0 = 10 + int(rng.integers(0, 40))
        q = float(rng.uniform(5.0, 9.0))
        if kind == "honest":
            plants.append(
                plant_honest(pid, side, pr, i0, hold=int(rng.integers(50, 150)), q=q,
                             n_bites=int(rng.integers(3, 7)))
            )
        else:
            plants.append(
                plant_flicker(pid, side, pr, i0, n_flick=int(rng.integers(3, 7)),
                              live=int(rng.integers(3, 10)), gap=int(rng.integers(5, 20)), q=q)
            )
    states, trades = build_session(plants, n_states=300, seed=seed + 1000)
    return plants, states, trades


@cache
def battery_flicker_sweep() -> dict:
    """B: свип числа переставлений n_flick на фоне честной популяции."""
    rng = np.random.default_rng(11)
    plants, _, slot = _honest_population(rng)
    grid = [0, 1, 2, 4, 8, 16, 32, 64]
    prices = {}
    for k, n in enumerate(grid):
        side, pr = _adversary_slots(slot, k)
        prices[n] = pr
        plants.append(plant_flicker(100 + k, side, pr, i0=12, n_flick=n))
    states, trades = build_session(plants, n_states=1000, seed=12)
    _, scored = run_detector(states, trades)
    rows = []
    for n in grid:
        eps = scored.filter(pl.col("price") == prices[n])
        rows.append(
            {
                "n_flick": n,
                "score": float(eps["score"].max()),
                "flicker_count": int(eps["flicker_count"].max()),
                "flagged": bool(eps["flicker"].any()),
            }
        )
    return {"rows": rows}


@cache
def battery_patient_sweep() -> dict:
    """C: «умный по времени жизни» спуфер на p-перцентиле честных hold'ов."""
    rng = np.random.default_rng(21)
    plants, holds, slot = _honest_population(rng)
    grid = [10, 25, 50, 75, 90, 95, 99]
    prices = {}
    for k, p in enumerate(grid):
        side, pr = _adversary_slots(slot, k)
        prices[p] = pr
        plants.append(plant_patient(100 + k, side, pr, i0=12, hold=int(np.percentile(holds, p))))
    states, trades = build_session(plants, n_states=220, seed=22)
    _, scored = run_detector(states, trades)
    honest = [wall_score(scored, p.price) for p in plants if p.label == "honest"]
    honest = [s for s in honest if s is not None]
    rows = [{"p": p, "score": wall_score(scored, prices[p])} for p in grid]
    return {"rows": rows, "honest_min": min(honest), "honest_max": max(honest)}


FEED_GRID = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.21, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]


@cache
def battery_feed_sweep(refill: bool = False) -> dict:
    """D (refill=False) / F-кривая (refill=True): свип доли подкормки f."""
    rng = np.random.default_rng(31 if not refill else 32)
    plants, holds, slot = _honest_population(rng)
    hold = int(np.percentile(holds, 95 if refill else 75))
    grid = [f for f in FEED_GRID if f <= 0.5] if refill else FEED_GRID
    prices = {}
    for k, f in enumerate(grid):
        side, pr = _adversary_slots(slot, k)
        prices[f] = pr
        plants.append(
            plant_feeder(100 + k, side, pr, i0=12, hold=hold, f=f, refill=refill)
        )
    states, trades = build_session(plants, n_states=240, seed=33)
    _, scored = run_detector(states, trades)
    honest = [wall_score(scored, p.price) for p in plants if p.label == "honest"]
    honest = [s for s in honest if s is not None]
    rows = []
    for f in grid:
        eps = scored.filter(pl.col("price") == prices[f])
        rows.append(
            {
                "f": f,
                "score": float(eps["score"].max()),
                "fill_frac": float(eps["fill_frac"].max()),
                "flicker_count": int(eps["flicker_count"].max()),
                "iceberg": bool(eps["iceberg"].any()),
                "chain_refills": int(eps["chain_refills"].max()),
            }
        )
    f_gate = next((r["f"] for r in rows if r["score"] >= SCORE_GATE), None)
    f_honest = next((r["f"] for r in rows if r["score"] >= min(honest)), None)
    return {"rows": rows, "honest_min": min(honest), "f_gate": f_gate, "f_honest": f_honest}


@cache
def battery_iceberg_sweep() -> dict:
    """E: честный айсберг при r рефиллах + «поздний» рефилл (вне окна 300 мс)."""
    rng = np.random.default_rng(41)
    plants, _, slot = _honest_population(rng, n=10)
    grid = [1, 2, 4, 8, 16]
    prices = {}
    for k, r in enumerate(grid):
        side, pr = _adversary_slots(slot, k)
        prices[r] = pr
        plants.append(plant_iceberg(100 + k, side, pr, i0=12, r=r))
    side, late_pr = _adversary_slots(slot, len(grid))
    plants.append(plant_iceberg(200, side, late_pr, i0=12, r=4, refill_lag=2))
    states, trades = build_session(plants, n_states=220, seed=42)
    _, scored = run_detector(states, trades)
    rows = []
    for r in grid:
        eps = scored.filter(pl.col("price") == prices[r])
        rows.append(
            {
                "r": r,
                "score": float(eps["score"].max()),
                "flicker": bool(eps["flicker"].any()),
                "iceberg": bool(eps["iceberg"].any()),
                "chain_refills": int(eps["chain_refills"].max()),
            }
        )
    late = scored.filter(pl.col("price") == late_pr)
    return {
        "rows": rows,
        "late": {
            "score": float(late["score"].max()),
            "flicker": bool(late["flicker"].any()),
            "chain_refills": int(late["chain_refills"].max()),
        },
    }


@cache
def battery_masked_population() -> dict:
    """F-популяция: 30 «идеальных спуферов» (долго стоит + кормит f>0.2 через
    рефиллы) против честных — доля прошедших гейт 0.4."""
    masked_scores: list[float] = []
    honest_scores: list[float] = []
    for b in range(2):
        rng = np.random.default_rng(51 + b)
        plants, holds, slot = _honest_population(rng)
        for k in range(15):
            side, pr = _adversary_slots(slot, k)
            hold = int(np.percentile(holds, float(rng.uniform(60, 95))))
            plants.append(
                plant_feeder(100 + k, side, pr, i0=12, hold=hold,
                             f=float(rng.uniform(0.21, 0.30)),
                             n_bites=int(rng.integers(3, 7)), refill=True)
            )
        states, trades = build_session(plants, n_states=240, seed=52 + b)
        _, scored = run_detector(states, trades)
        for p in plants:
            s = wall_score(scored, p.price)
            if s is None:
                continue
            (honest_scores if p.label == "honest" else masked_scores).append(s)
    passed = sum(1 for s in masked_scores if s >= SCORE_GATE)
    return {
        "masked_scores": masked_scores,
        "honest_scores": honest_scores,
        "pass_rate": passed / len(masked_scores),
    }


@cache
def battery_noise_fpr() -> dict:
    """G: шумовые дни без плантов. strict — крупных уровней нет вообще
    (qty <= 2 < порога); sigmaXX — чёрн-фон с хвостами разной толщины:
    ищем, с какой sigma flicker начинает ложно срабатывать."""
    n_days = max(20, int(100 * SCALE))
    out = {"n_days": n_days}
    for name, kw in (
        ("strict", {"bg_sigma": 0.3, "churn": 0.4, "qty_cap": 2.0}),
        ("sigma05", {"bg_sigma": 0.5, "churn": 0.4}),
        ("sigma07", {"bg_sigma": 0.7, "churn": 0.4}),
        ("sigma10", {"bg_sigma": 1.0, "churn": 0.4}),
    ):
        flagged_days = flicker_eps = large_eps = 0
        for d in range(n_days):
            states, trades = build_session([], n_states=150, seed=5000 + d, **kw)
            _, scored = run_detector(states, trades)
            nf = int(scored["flicker"].sum())
            flicker_eps += nf
            large_eps += int(scored["was_large"].sum())
            flagged_days += bool(nf)
        out[name] = {
            "flagged_days": flagged_days,
            "flicker_eps": flicker_eps,
            "large_eps": large_eps,
            "fpr_days": flagged_days / n_days,
        }
    return out


@cache
def battery_threshold_sweep() -> dict:
    """H: порог «крупного уровня» x0.5 / x1 / x2 -> precision/recall."""
    rows = []
    n_batches = 4
    for lk in (LARGE_K * 0.5, LARGE_K, LARGE_K * 2.0):
        agg = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
        for b in range(n_batches):
            plants, states, trades = _population_batch(seed=100 + b)
            _, scored = run_detector(states, trades, large_k=lk)
            res = evaluate_spoof_flags(truth_frame(plants), scored, flag="flicker")
            for k in agg:
                agg[k] += res[k]
        precision = agg["tp"] / (agg["tp"] + agg["fp"]) if agg["tp"] + agg["fp"] else 1.0
        recall = agg["tp"] / (agg["tp"] + agg["fn"]) if agg["tp"] + agg["fn"] else 1.0
        rows.append({"large_k": lk, "precision": precision, "recall": recall, **agg})
    return {"rows": rows}


@cache
def battery_perf() -> dict:
    """I: 500 одновременных стен, ~100k строк уровней (книжных событий)."""
    rng = np.random.default_rng(61)
    plants = []
    for i in range(500):
        side = "bid" if i % 2 == 0 else "ask"
        pr = MID + (-1 if side == "bid" else 1) * (4 + 2 * (i // 2)) * TICK
        plants.append(
            plant_honest(i, side, pr, i0=int(rng.integers(2, 20)),
                         hold=int(rng.integers(40, 70)))
        )
    t0 = time.perf_counter()
    states, trades = build_session(plants, n_states=100, seed=62, n_bg=550)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    j, scored = run_detector(states, trades, record_grid=True)
    t_detect = time.perf_counter() - t0
    large = scored.filter(pl.col("was_large"))
    return {
        "t_build_s": t_build,
        "t_detect_s": t_detect,
        "n_level_rows": j.grid_frame().height,
        "n_episodes": len(j.episodes),
        "n_large": large.height,
        "large_min_score": float(large["score"].min()) if large.height else None,
    }


# -- A: базовые популяции -----------------------------------------------------


def test_a_populations_precision_recall():
    res = battery_populations()
    assert res["tp"] + res["fn"] >= 40  # спуфы реально посажены
    # уровни из test_m6_labeled_day
    assert res["precision"] >= 0.8
    assert res["recall"] >= 0.8


def test_a_population_score_separation():
    res = battery_populations()
    honest, spoof = res["honest_scores"], res["spoof_scores"]
    assert len(honest) >= 40 and len(spoof) >= 40
    assert min(honest) >= 0.5  # конвенция labeled_day: честные >= 0.5
    assert max(spoof) < 0.2  # спуферы < 0.2
    assert min(honest) - max(spoof) > 0.3


# -- B: свип мерцаний ---------------------------------------------------------


def test_b_flicker_sweep_monotone_and_boundary():
    rows = battery_flicker_sweep()["rows"]
    scores = [r["score"] for r in rows]
    assert all(a >= b - 1e-12 for a, b in zip(scores, scores[1:], strict=False))  # не растёт
    by_n = {r["n_flick"]: r for r in rows}
    # ГРАНИЦА по score: уже n_flick=0 (одна отмена без принтов) ниже гейта:
    # cancel-смерть обнуляет fill-компонент (w=0.50) и сама считается одним
    # фликером (самоучёт в окне) -> демпфер 0.5.
    assert by_n[0]["score"] < SCORE_GATE
    assert by_n[0]["flicker_count"] == 1
    # закон затухания: каждый дополнительный фликер режет score вдвое
    assert by_n[1]["score"] == pytest.approx(by_n[0]["score"] * 0.5, rel=0.35)
    # ГРАНИЦА по флагу flicker (k=3): с n_flick=2 (3 рождения)
    assert not by_n[0]["flagged"] and not by_n[1]["flagged"]
    assert all(by_n[n]["flagged"] for n in (2, 4, 8, 16, 32, 64))
    assert by_n[8]["score"] < 5e-3  # глубокое подавление


# -- C: терпеливый спуфер -----------------------------------------------------


def test_c_patient_spoofer_never_passes():
    res = battery_patient_sweep()
    scores = [r["score"] for r in res["rows"]]
    assert all(a <= b + 1e-12 for a, b in zip(scores, scores[1:], strict=False))  # растёт с p
    # СИЛЬНАЯ СТОРОНА: p* не существует — даже жизнь на p99 честных времен
    # даёт score <= w_life/2 (=0.125): вес времени жизни всего 0.25, отмена
    # без исполнений забирает fill-компонент и включает демпфер 0.5.
    assert max(scores) <= W_LIFE / 2 + 1e-9
    assert res["honest_min"] >= 0.5
    assert res["honest_min"] - max(scores) > 0.3


# -- D: подкормка -------------------------------------------------------------


def test_d_feed_sweep_cliff_and_gate():
    res = battery_feed_sweep(refill=False)
    rows = {r["f"]: r for r in res["rows"]}
    scores = [r["score"] for r in res["rows"]]
    assert all(a <= b + 1e-12 for a, b in zip(scores, scores[1:], strict=False))  # растёт с f
    # ДИЗАЙН-СЛАБОСТЬ (документируем, не чиним): обрыв ровно на
    # max_flicker_fill=0.2 — при f<=0.2 эпизод квалифицируется как фликер
    # (демпфер 0.5), при f>0.2 демпфер исчезает -> score прыгает ~x2.
    assert rows[0.20]["flicker_count"] == 1 and rows[0.21]["flicker_count"] == 0
    assert rows[0.21]["score"] / rows[0.20]["score"] > 1.5
    # ГРАНИЦА гейта 0.4: подкормка ~40% объёма проходит гейт стены
    assert res["f_gate"] is not None and 0.30 < res["f_gate"] <= 0.45
    # неотличим от честной (>= min честных) только при f >= ~0.6
    assert res["f_honest"] is not None and 0.50 < res["f_honest"] <= 0.75


# -- E: айсберг против спуфа --------------------------------------------------


def test_e_iceberg_not_confused_with_flicker():
    res = battery_iceberg_sweep()
    for row in res["rows"]:
        assert not row["flicker"], f"айсберг r={row['r']} принят за мерцание"
        assert row["score"] >= 0.6
        assert row["chain_refills"] == row["r"]
    by_r = {r["r"]: r for r in res["rows"]}
    assert not by_r[1]["iceberg"]  # min_refills=2: граница флага iceberg
    assert all(by_r[r]["iceberg"] for r in (2, 4, 8, 16))
    scores = [r["score"] for r in res["rows"]]
    assert all(a <= b + 1e-12 for a, b in zip(scores, scores[1:], strict=False))
    # поздний рефилл (400 мс > окна 300 мс): цепочка не строится, но стена
    # умирает исполненной -> высокого score не теряет и фликером не считается
    assert res["late"]["chain_refills"] == 0
    assert not res["late"]["flicker"]
    assert res["late"]["score"] >= 0.5


# -- F: идеальный спуфер (маскировка) -----------------------------------------


def test_f_masking_min_cost_boundary():
    res = battery_feed_sweep(refill=True)
    rows = {r["f"]: r for r in res["rows"]}
    # ДИЗАЙН-СЛАБОСТЬ (документируем, не чиним): минимальная стоимость
    # маскировки = f чуть выше max_flicker_fill (0.2) + мгновенные рефиллы.
    # При f=0.21 спуф собирает iceberg-бонус r/(r+1) и w_fill*0.21 и
    # обгоняет минимум честных стен; вдобавок помечается флагом iceberg.
    assert rows[0.20]["score"] < 0.3  # на волосок ниже границы — ещё ловится
    assert rows[0.21]["score"] >= SCORE_GATE + 0.1
    assert rows[0.21]["score"] >= res["honest_min"] - 0.02  # неотличим
    assert rows[0.21]["iceberg"]  # и «одет» в честный айсберг
    assert rows[0.21]["fill_frac"] == pytest.approx(0.21, abs=0.02)
    # сам по себе рефилл без f>0.2 НЕ спасает: демпфер фликера сильнее бонуса
    assert rows[0.05]["score"] < SCORE_GATE


def test_f_masked_population_blinds_detector():
    res = battery_masked_population()
    # ГЛАВНЫЙ ВЫВОД: >=90% замаскированных спуферов проходят гейт 0.4,
    # т.е. детектор слеп к «долго стоит + кормит >20% через рефиллы».
    assert res["pass_rate"] >= 0.9
    assert min(res["honest_scores"]) >= 0.5


# -- G: шум без крупных уровней ----------------------------------------------


def test_g_noise_false_positive_rate():
    res = battery_noise_fpr()
    # строгие шумовые дни (крупных уровней нет вообще): ноль ложных флагов
    assert res["strict"]["large_eps"] == 0
    assert res["strict"]["flicker_eps"] == 0
    # умеренные хвосты (sigma=0.5): крупные уровни в шуме уже рождаются
    # (сотни эпизодов на 100 дней), но flicker молчит — FPR = 0
    assert res["sigma05"]["large_eps"] > 0
    assert res["sigma05"]["flicker_eps"] == 0
    # ДИЗАЙН-СЛАБОСТЬ (документируем): с sigma ~0.7 чёрн изредка рождает
    # «крупный» уровень, умирающий отменой >=3 раз в окне, — появляются
    # ложные flicker-флаги; к sigma=1.0 FPR растёт до единиц процентов дней.
    assert res["sigma07"]["fpr_days"] <= 0.05
    assert res["sigma10"]["fpr_days"] <= 0.15
    if res["n_days"] >= 100:
        assert res["sigma10"]["flagged_days"] >= 1  # слепым нулём не является


# -- H: чувствительность порога «крупного уровня» -----------------------------


def test_h_large_threshold_sensitivity():
    rows = battery_threshold_sweep()["rows"]
    by_k = {round(r["large_k"], 2): r for r in rows}
    for r in rows:
        assert r["precision"] >= 0.8
    assert by_k[1.5]["recall"] >= 0.9
    assert by_k[3.0]["recall"] >= 0.9
    # ДИЗАЙН-СЛАБОСТЬ (документируем): порог x2 (large_k=6) выкидывает часть
    # стен (qty < 6*медианы) из «крупных» -> recall проседает.
    assert by_k[6.0]["recall"] <= 0.8


# -- I: масштаб и перф --------------------------------------------------------


def test_i_scale_500_walls_100k_events():
    res = battery_perf()
    assert res["n_level_rows"] >= 100_000  # «день» из >=100k книжных событий
    assert res["n_large"] == 500  # все стены распознаны крупными
    assert res["large_min_score"] > SCORE_GATE  # и остаются честными
    assert res["n_episodes"] < 2_000  # эпизоды не размножаются
    assert res["t_detect_s"] < 30.0 * max(1.0, SCALE)


# -- J: wall_zones_at / merge_zones (трек B) ----------------------------------


def _ep_row(price, birth_s, death_s, score, max_qty=10.0, lifetime_ms=None):
    death = None if death_s is None else T0 + int(death_s * NS_PER_S)
    birth = T0 + int(birth_s * NS_PER_S)
    return {
        "price": price,
        "birth_ts": birth,
        "death_ts": death if death is not None else T0 + 10**15,
        "alive": death_s is None,
        "lifetime_ms": lifetime_ms
        if lifetime_ms is not None
        else ((death - birth) / 1e6 if death_s is not None else 0.0),
        "max_qty": max_qty,
        "score": score,
    }


def test_j_live_wall_first_tick_prior():
    # живая стена в ПЕРВЫЙ тик жизни, разрешённых стен ещё нет: перцентиль
    # откатывается к 0.5 -> прайор = W_LIFE * 0.5 = 0.125 (не ноль!).
    eps = pl.DataFrame([_ep_row(100.0, 10, None, score=1.0)])
    ts = T0 + 10 * NS_PER_S
    z = wall_zones_at(eps, ts, band=2.0, min_score=0.0)
    assert z.height == 1
    assert z["heat_usd"][0] == pytest.approx(100.0 * 10.0 * W_LIFE * 0.5)
    # дефолтный гейт 0.4 такую стену не пропускает
    assert wall_zones_at(eps, ts, band=2.0).height == 0
    # при наличии разрешённой стены-референса перцентиль нулевого возраста = 0:
    # зона включается с heat == 0 при min_score=0 (граница '>=', документируем)
    eps2 = pl.DataFrame(
        [_ep_row(100.0, 10, None, 1.0), _ep_row(90.0, -500, -100, 0.8, lifetime_ms=400_000)]
    )
    z2 = wall_zones_at(eps2, ts, band=2.0, min_score=0.0, dead_half_life_s=1e12)
    live_zone = z2.filter(pl.col("lo") == 99.0)
    assert live_zone.height == 1
    assert live_zone["heat_usd"][0] == 0.0


def test_j_dead_wall_half_life_boundary():
    eps = pl.DataFrame([_ep_row(100.0, 0, 60, score=0.8)])
    ts = T0 + (60 + 3_600) * NS_PER_S  # возраст ровно dead_half_life
    z = wall_zones_at(eps, ts, band=2.0, min_score=0.4, dead_half_life_s=3_600.0)
    assert z.height == 1  # eff == 0.8 * 0.5 == min_score: граница включительна
    assert z["heat_usd"][0] == pytest.approx(100.0 * 10.0 * 0.4, rel=1e-9)
    assert wall_zones_at(eps, ts + NS_PER_S, band=2.0, min_score=0.4).height == 0
    # смерть ровно в ts: уже «разрешённая» стена, полный score без распада
    z0 = wall_zones_at(eps, T0 + 60 * NS_PER_S, band=2.0, min_score=0.4)
    assert z0["heat_usd"][0] == pytest.approx(100.0 * 10.0 * 0.8)


def test_j_thousand_zones_merge_fast_and_sorted():
    rows = [_ep_row(100.0 + i, 0, 50, score=0.9) for i in range(1000)]
    eps = pl.DataFrame(rows)
    t0 = time.perf_counter()
    zones = wall_zones_at(eps, T0 + 100 * NS_PER_S, band=1.0, min_score=0.1)
    t_zones = time.perf_counter() - t0
    assert zones.height == 1000
    assert bool((zones["lo"].diff().drop_nulls() >= 0).all())
    t0 = time.perf_counter()
    merged = merge_zones(zones, zones.with_columns(pl.col("heat_usd") * 0.5))
    t_merge = time.perf_counter() - t0
    assert merged.height == 2000
    assert bool((merged["lo"].diff().drop_nulls() >= 0).all())
    assert list(merged.schema) == list(ZONE_SCHEMA)
    assert t_zones < 2.0 and t_merge < 1.0
    assert merge_zones().height == 0


# -- K: патологии входа -------------------------------------------------------


def _flat_state(i: int, extra_bid=None) -> BookState:
    bids = ((99.0, 1.0), (98.0, 1.0), (97.0, 1.0)) + ((extra_bid,) if extra_bid else ())
    return BookState(ts=T0 + i * CAD_NS, bids=bids, asks=((101.0, 1.0), (102.0, 1.0)))


def test_k_out_of_order_states_raise():
    states = [_flat_state(0), _flat_state(1, (96.0, 5.0)), _flat_state(2)]
    with pytest.raises(ValueError, match="non-decreasing"):
        LevelJournal(large_k=3.0, iceberg_refill_ms=300).run(list(reversed(states)), [])


def test_k_equal_timestamps_allowed():
    same = [
        _flat_state(0),
        BookState(ts=T0, bids=((99.0, 1.0), (96.0, 5.0)), asks=((101.0, 1.0),)),
        BookState(ts=T0, bids=((99.0, 1.0),), asks=((101.0, 1.0),)),
    ]
    j = LevelJournal(large_k=3.0, iceberg_refill_ms=300).run(same, [])
    (ep,) = [e for e in j.episodes if e.price == 96.0]
    assert ep.lifetime_ns == 0 and not ep.alive


def test_k_zero_qty_level_is_absence():
    # L2-контракт: qty=0 означает отсутствие уровня -> эпизода нет вообще
    states = [_flat_state(0), _flat_state(1, (96.0, 0.0)), _flat_state(2)]
    j = LevelJournal(large_k=3.0, iceberg_refill_ms=300).run(states, [])
    assert not [e for e in j.episodes if e.price == 96.0]


def test_k_zero_price_wall_scores_but_zone_heat_is_zero():
    states = [_flat_state(0), _flat_state(1, (0.0, 5.0)), _flat_state(2)]
    j = LevelJournal(large_k=3.0, iceberg_refill_ms=300).run(states, [])
    ann = annotate_episodes(j, flicker_k=3, flicker_window_s=60)
    scored = score_episodes(ann)
    row = scored.filter(pl.col("price") == 0.0)
    assert row.height == 1
    assert np.isfinite(row["score"][0])
    # notional = price*qty = 0 -> зона отсекается любым min_notional_usd > 0
    zones = wall_zones_at(
        scored.select("price", "birth_ts", "death_ts", "alive", "lifetime_ms",
                      "max_qty", "score"),
        T0 + 2 * CAD_NS, band=1.0, min_score=0.0, min_notional_usd=1e-9,
    )
    assert zones.filter(pl.col("lo") <= 0.0).height == 0


def test_k_unsorted_trades_are_sorted_internally():
    states = [
        _flat_state(0),
        _flat_state(1, (96.0, 6.0)),
        _flat_state(2, (96.0, 3.0)),
        _flat_state(3),
    ]

    def trade(i, qty, tid):
        ts = T0 + i * CAD_NS - HALF
        return Trade(EXCHANGE, "BTCUSDT", ts, ts + NS_PER_MS, 96.0, qty, 96.0 * qty,
                     Side.SELL, tid)

    trades = [trade(3, 3.0, 2), trade(2, 3.0, 1)]  # задом наперёд
    j1 = LevelJournal(large_k=3.0, iceberg_refill_ms=300).run(states, trades)
    j2 = LevelJournal(large_k=3.0, iceberg_refill_ms=300).run(states, trades[::-1])
    assert j1.events == j2.events
    (ep,) = [e for e in j1.episodes if e.price == 96.0]
    assert ep.filled_qty == pytest.approx(6.0)
    assert ep.canceled_qty == pytest.approx(0.0)
