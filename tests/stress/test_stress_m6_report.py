"""Отчёт стресс-кампании M6: reports/stress-m6/README.md + 3 PNG.

Генерирует (по конвенциям *_reports-тестов и viz/style) карту сильных/слабых
сторон спуфинг-детектора из чисел адверсариальной батареи
tests/stress/test_stress_m6_spoof.py и таймингов M7 из
tests/stress/test_stress_m7_signals.py. Батареи кэшированы (lru_cache), так
что при совместном прогоне сценарии не пересчитываются.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from trading_system.viz.style import PALETTE, apply_style, save_fig

pytestmark = pytest.mark.stress

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:  # прямой запуск одного файла
    sys.path.insert(0, str(_HERE))

import test_stress_m6_spoof as m6b  # noqa: E402
import test_stress_m7_signals as m7b  # noqa: E402

REPO = _HERE.parents[1]
OUT_DIR = REPO / "reports" / "stress-m6"
GATE = m6b.SCORE_GATE


def _fig_populations(pops: dict, masked: dict) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(0.0, 1.0, 26)
    for scores, label, color in (
        (pops["honest_scores"], "честные стены", PALETTE["long"]),
        (pops["spoof_scores"], "мерцающий спуф", PALETTE["short"]),
        (masked["masked_scores"], "«умный» спуфер (f>0.2 + рефиллы)", PALETTE["accent"]),
    ):
        ax.hist(scores, bins=bins, alpha=0.65, label=f"{label} (n={len(scores)})",
                color=color, edgecolor="white")
    ax.axvline(GATE, color=PALETTE["neutral"], ls="--", lw=1.2,
               label=f"гейт стены min_score={GATE}")
    ax.set_title("Распределения stability score: честные vs спуф vs «умный» спуфер")
    ax.set_xlabel("stability score")
    ax.set_ylabel("число стен")
    ax.legend()
    return save_fig(fig, "stress_m6_scores_populations", OUT_DIR)


def _fig_patient(patient: dict) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    ps = [r["p"] for r in patient["rows"]]
    scores = [r["score"] for r in patient["rows"]]
    ax.plot(ps, scores, marker="o", color=PALETTE["short"], label="score «терпеливого» спуфа")
    ax.axhline(patient["honest_min"], color=PALETTE["long"], ls="--",
               label=f"минимум честных ({patient['honest_min']:.2f})")
    ax.axhline(GATE, color=PALETTE["neutral"], ls="--", label=f"гейт стены ({GATE})")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Кривая детекции: спуф живёт на p-перцентиле времени жизни честных стен")
    ax.set_xlabel("p — перцентиль hold-времени честных стен")
    ax.set_ylabel("stability score")
    ax.legend()
    return save_fig(fig, "stress_m6_patient_curve", OUT_DIR)


def _fig_feed(feed: dict, feed_refill: dict) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    for res, label, color in (
        (feed, "подкормка без рефиллов", PALETTE["neutral"]),
        (feed_refill, "подкормка через мгновенные рефиллы", PALETTE["accent"]),
    ):
        ax.plot([r["f"] for r in res["rows"]], [r["score"] for r in res["rows"]],
                marker="o", color=color, label=label)
    ax.axhline(feed["honest_min"], color=PALETTE["long"], ls="--",
               label=f"минимум честных ({feed['honest_min']:.2f})")
    ax.axhline(GATE, color=PALETTE["short"], ls="--", label=f"гейт стены ({GATE})")
    ax.axvline(0.2, color=PALETTE["short"], ls=":",
               label="max_flicker_fill=0.2 (обрыв демпфера)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Кривая по доле подкормки f: сколько объёма спуфу надо исполнить")
    ax.set_xlabel("f — доля исполненного объёма (fill_frac)")
    ax.set_ylabel("stability score")
    ax.legend()
    return save_fig(fig, "stress_m6_feed_curve", OUT_DIR)


def _readme(ctx: dict) -> str:
    pops, patient, feed, feed_r = ctx["pops"], ctx["patient"], ctx["feed"], ctx["feed_r"]
    flick, ice, masked = ctx["flick"], ctx["ice"], ctx["masked"]
    noise, thr, perf, m7 = ctx["noise"], ctx["thr"], ctx["perf"], ctx["m7"]
    by_n = {r["n_flick"]: r for r in flick["rows"]}
    fd = {r["f"]: r for r in feed["rows"]}
    fr = {r["f"]: r for r in feed_r["rows"]}
    n_flick_flag = min(r["n_flick"] for r in flick["rows"] if r["flagged"])
    thr_rows = "\n".join(
        f"| H: large_k={r['large_k']:.1f} | {r['precision']:.3f} | {r['recall']:.3f} | "
        f"tp={r['tp']:.0f} fp={r['fp']:.0f} fn={r['fn']:.0f} tn={r['tn']:.0f} |"
        for r in thr["rows"]
    )
    ice_rows = "\n".join(
        f"| r={r['r']} | {r['score']:.3f} | {'да' if r['iceberg'] else 'нет'} | "
        f"{'ДА (ошибка!)' if r['flicker'] else 'нет'} |"
        for r in ice["rows"]
    )
    lines = f"""# Отчёт: stress-m6 — адверсариальная батарея спуфинг-детектора

Границы детектора M6 (lifecycle → metrics → score) числами: где он силён,
где слеп и сколько стоит его обмануть. Все сценарии сидированы
(tests/stress/test_stress_m6_spoof.py); формула: base = (0.25·pct + 0.50·fill
+ 0.25·r/(r+1)), score = base·0.5^flick. Операционный гейт «стена настоящая»
— min_score={GATE} (walls.wall_zones_at); конвенция качества — precision/recall
≥ 0.8 (test_m6_labeled_day).

## Precision / recall по сценариям (флаг flicker)

| Сценарий | precision | recall | Детали |
|---|---|---|---|
| A: {len(pops['honest_scores'])} честных vs {len(pops['spoof_scores'])} мерцающих спуфов | {pops['precision']:.3f} | {pops['recall']:.3f} | tp={pops['tp']:.0f} fp={pops['fp']:.0f} fn={pops['fn']:.0f} tn={pops['tn']:.0f} |
{thr_rows}

Порог «крупного уровня» ×0.5 (large_k=1.5) качества не портит; ×2 (large_k=6)
выкидывает стены qty < 6·медианы из «крупных» — recall падает до
{thr['rows'][-1]['recall']:.2f} (слабость калибровки порога).

## Найденные граничные числа

| Граница | Значение | Смысл |
|---|---|---|
| n_flick* (score < {GATE}) | **0** | уже ОДНО появление с отменой без принтов даёт score {by_n[0]['score']:.3f}: отмена забирает fill-компонент (вес 0.50) и сама считается фликером (демпфер 0.5) |
| n_flick* (флаг flicker, k=3) | **{n_flick_flag}** | {n_flick_flag + 1} рождения в окне 60 с; дальше score гаснет вдвое за фликер (n=8 → {by_n[8]['score']:.4f}) |
| p* (терпеливый спуфер) | **не существует** | стена, живущая даже на p99 честных времен, набирает лишь {max(r['score'] for r in patient['rows']):.3f} ≤ w_life/2 = 0.125; минимум честных {patient['honest_min']:.3f} |
| обрыв на max_flicker_fill=0.2 | score ×{fd[0.21]['score'] / fd[0.20]['score']:.1f} | f=0.20 → {fd[0.20]['score']:.3f}, f=0.21 → {fd[0.21]['score']:.3f}: жёсткая граница квалификации фликера — дешёвая точка обхода |
| f* (гейт {GATE}, без рефиллов) | **{feed['f_gate']:.2f}** | подкормка ~40% объёма проходит операционный гейт стены |
| f* (неотличим от честных, без рефиллов) | **{feed['f_honest']:.2f}** | score ≥ минимума честных только при f ≥ {feed['f_honest']:.2f} |
| минимальная стоимость маскировки | **f ≈ {feed_r['f_gate']:.2f}** (с рефиллами) | f чуть выше 0.2 + мгновенные рефиллы: score {fr[0.21]['score']:.3f} ≥ минимума честных {feed_r['honest_min']:.3f}, плюс ярлык iceberg |
| слепота на «умных» спуферах | **{masked['pass_rate']:.0%}** проходят гейт | популяция из {len(masked['masked_scores'])} спуферов «долго стоит + кормит f∈[0.21,0.30] рефиллами» |

## Айсберг vs спуф (E): рефилл ≠ флик

| Айсберг | score | флаг iceberg | принят за мерцание? |
|---|---|---|---|
{ice_rows}

Граница флага iceberg — r ≥ 2 (min_refills). «Поздний» рефилл (400 мс > окна
300 мс) цепочку не строит (chain_refills={ice['late']['chain_refills']}), но стена умирает исполненной:
score {ice['late']['score']:.3f}, фликером не считается. Путаницы айсберг/спуф нет.

## Ложные срабатывания на шуме (G), {noise['n_days']} сидированных дней на вариант

| Фон | дней с ложным flicker | ложных эпизодов | «крупных» эпизодов в шуме |
|---|---|---|---|
| без крупных уровней вообще (qty ≤ 2) | {noise['strict']['flagged_days']} | {noise['strict']['flicker_eps']} | {noise['strict']['large_eps']} |
| хвосты σ=0.5 | {noise['sigma05']['flagged_days']} | {noise['sigma05']['flicker_eps']} | {noise['sigma05']['large_eps']} |
| хвосты σ=0.7 | {noise['sigma07']['flagged_days']} | {noise['sigma07']['flicker_eps']} | {noise['sigma07']['large_eps']} |
| хвосты σ=1.0 | {noise['sigma10']['flagged_days']} | {noise['sigma10']['flicker_eps']} | {noise['sigma10']['large_eps']} |

Ноль ложных срабатываний держится до σ≈0.5 включительно; с σ≈0.7 чёрн изредка
рождает «крупный» уровень, умирающий отменой ≥ 3 раз в окне, — FPR растёт с
толщиной хвостов ({noise['sigma10']['fpr_days']:.0%} дней при σ=1.0).

## Масштаб и перф (I)

- 500 одновременных стен, {perf['n_level_rows']:,} строк уровней (книжных событий),
  {perf['n_episodes']} эпизодов: полный проход детектора за **{perf['t_detect_s']:.2f} с**
  (генерация {perf['t_build_s']:.2f} с); все 500 стен распознаны крупными,
  минимальный score честной стены {perf['large_min_score']:.3f} > {GATE}.
- wall_zones_at на 1000 эпизодов и merge_zones на 2000 зон — миллисекунды
  (см. test_j_thousand_zones_merge_fast_and_sorted).

## Графики

![Распределения score](stress_m6_scores_populations.png)

![Кривая по перцентилю времени жизни](stress_m6_patient_curve.png)

![Кривая по доле подкормки](stress_m6_feed_curve.png)

## Выводы

**Сильные стороны**
- Классика ловится идеально: на 200/200 честных vs мерцающих precision={pops['precision']:.2f},
  recall={pops['recall']:.2f}; разрыв популяций по score огромен (мин. честных
  {min(pops['honest_scores']):.2f} vs макс. спуфа {max(pops['spoof_scores']):.2f}).
- Терпение не спасает спуфера: без исполнений даже p99-долгожитель набирает
  ≤ 0.125 — w_life (0.25) мал, а отмена в конце забирает fill-вес и включает демпфер.
- Одна-единственная отмена крупного уровня уже роняет score ниже гейта (n_flick*=0).
- Айсберги не путаются с мерцанием ни при быстрых, ни при поздних рефиллах.
- На шуме без крупных уровней ложных срабатываний ровно ноль; перф с запасом
  (100k+ событий за доли секунды).

**Слабые стороны**
- Главная: маскировка стоит всего ~21% объёма. f чуть выше max_flicker_fill=0.2
  плюс мгновенные рефиллы — и спуф неотличим от честной стены
  ({masked['pass_rate']:.0%} «умных» спуферов проходят гейт) да ещё и помечен «айсбергом».
- Жёсткий обрыв на max_flicker_fill=0.2: score прыгает ×{fd[0.21]['score'] / fd[0.20]['score']:.1f} при f 0.20→0.21 —
  адверсарию известна дешёвая точка обхода.
- w_fill=0.50 линейно оплачивает подкормку: без рефиллов гейт пробивается при f≈{feed['f_gate']:.2f}.
- Iceberg-бонус r/(r+1) зарабатывается самоисполнением: рефилл после «своего» принта
  неотличим от честного рефилла.
- large_k×2 роняет recall до {thr['rows'][-1]['recall']:.2f}; порог кратен медиане, а не notional.
- На толстохвостом фоне (σ≥0.7) flicker даёт ложные флаги.

**Чем дожать**
- Штраф за абсолютную отменённую массу: canceled_qty у «замаскированного» спуфа
  всё равно ~79% notional — метрика cancel_to_fill на уровне цены ловит то, что
  score прощает.
- Сгладить обрыв: вместо жёсткой границы max_flicker_fill демпфировать флик
  весом (1 − fill_frac) — исчезает точка обхода f=0.2+ε.
- Iceberg-бонус выдавать только эпизодам, умершим исполненными (fill_frac ≥
  exec_min_fill), — рефиллы перед финальной отменой не должны награждаться.
- Порог «крупного» калибровать по notional/ATR, а не кратностью медианы —
  устойчивость recall к large_k×2 и к толстым хвостам фона.

## M7: стресс сигнального слоя (signals/detectors)

Границы (tests/stress/test_stress_m7_signals.py):
- **s1_magnet**: пул ровно на k·ATR — сигнал (граница включительная); heat ровно
  θ·ΣH — сигнал; edge-trigger без дублей и префиксная согласованность на
  {m7['n_bars']:,} барах; пустая карта / все пулы тронуты → 0 событий; пул,
  тронутый ровно на закрытии бара, на этом баре уже не цель. ATR=0 не
  фильтруется: пул ровно на close даёт вырожденный «магнит в себя»
  (target == price, side=−1) — задокументировано.
- **s2_sweep_reversal**: прокол ровно на уровне — НЕ прокол (строгое >);
  префиксная согласованность на {m7['n_bars']:,} барах при return_bars=3;
  при return_bars ≥ 4 согласованность ЛОМАЕТСЯ на вложенных проколах
  (полный день отдаёт бар первому проколу, префикс — второму; другой ts и
  target) — задокументированная дизайн-слабость.
- **s3_filter**: зона с heat ровно на квантили — блокирует (≥); касание края
  пути — блокирует (включительно); зоны покрывают всё → вето 100%; зон нет /
  все вне пути → вето 0%.

Перф (сидированный день):
- s1: {m7['n_bars']:,} баров × 7 пулов — {m7['t_s1_day_s']:.2f} с ({m7['s1_day_events']} событий);
- s2: {m7['n_bars']:,} баров × 2 уровня — {m7['t_s2_day_s']:.2f} с ({m7['s2_day_events']} событий);
- шторм s1: 1000 пулов × {m7['s1_storm_bars']:,} баров — {m7['t_s1_storm_s']:.2f} с;
- шторм s3: 1000 зон × {m7['s3_storm_events']:,} событий — {m7['t_s3_storm_s']:.2f} с
  (заблокировано {m7['s3_storm_blocked']}).

Сложность s1/s3 — питоновские циклы O(баров×пулов) / O(событий×зон): на
штормовых размерах укладывается в доли секунды, но это первый кандидат на
векторизацию при росте карты.
"""
    return lines


def test_build_stress_report():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = {
        "pops": m6b.battery_populations(),
        "flick": m6b.battery_flicker_sweep(),
        "patient": m6b.battery_patient_sweep(),
        "feed": m6b.battery_feed_sweep(refill=False),
        "feed_r": m6b.battery_feed_sweep(refill=True),
        "ice": m6b.battery_iceberg_sweep(),
        "masked": m6b.battery_masked_population(),
        "noise": m6b.battery_noise_fpr(),
        "thr": m6b.battery_threshold_sweep(),
        "perf": m6b.battery_perf(),
        "m7": m7b.storm_numbers(),
    }
    paths = [
        _fig_populations(ctx["pops"], ctx["masked"]),
        _fig_patient(ctx["patient"]),
        _fig_feed(ctx["feed"], ctx["feed_r"]),
    ]
    for p in paths:
        assert p.exists() and p.stat().st_size > 5 * 1024, p

    # распределения score обеих популяций + «умных» — артефактом
    dist = pl.concat(
        [
            pl.DataFrame({"population": [name] * len(scores), "score": scores})
            for name, scores in (
                ("honest", ctx["pops"]["honest_scores"]),
                ("spoof", ctx["pops"]["spoof_scores"]),
                ("masked_spoof", ctx["masked"]["masked_scores"]),
            )
        ]
    )
    dist.write_csv(OUT_DIR / "score_distributions.csv")
    assert (OUT_DIR / "score_distributions.csv").stat().st_size > 1_000

    readme = OUT_DIR / "README.md"
    readme.write_text(_readme(ctx), encoding="utf-8")
    text = readme.read_text(encoding="utf-8")
    for marker in (
        "минимальная стоимость маскировки",
        "n_flick*",
        "p* (терпеливый спуфер)",
        "Сильные стороны",
        "Слабые стороны",
        "Чем дожать",
        "M7: стресс сигнального слоя",
        "stress_m6_scores_populations.png",
        "stress_m6_patient_curve.png",
        "stress_m6_feed_curve.png",
    ):
        assert marker in text, marker
