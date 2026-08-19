"""M6: spoofing heuristics over L2 level lifecycles; stability score.

Public entry points: :class:`LevelJournal` (lifecycle events + episodes),
:func:`annotate_episodes`/:func:`cancel_to_fill` (metrics),
:func:`stability_score`/:func:`journal_scores` (scoring) and
``reports.demo_reports`` (figures). Synthetic labeled sessions live in
``trading_system.spoof.synth``.
"""

from trading_system.spoof.lifecycle import (
    BookState,
    LevelEpisode,
    LevelEvent,
    LevelEventType,
    LevelJournal,
)
from trading_system.spoof.metrics import (
    annotate_episodes,
    cancel_to_fill,
    episodes_frame,
    flicker_flags,
    iceberg_flags,
    large_level_lifetimes,
)
from trading_system.spoof.score import (
    journal_scores,
    lifetime_percentiles,
    score_episodes,
    score_grid,
    stability_score,
)

__all__ = [
    "BookState",
    "LevelEpisode",
    "LevelEvent",
    "LevelEventType",
    "LevelJournal",
    "annotate_episodes",
    "cancel_to_fill",
    "episodes_frame",
    "flicker_flags",
    "iceberg_flags",
    "journal_scores",
    "large_level_lifetimes",
    "lifetime_percentiles",
    "score_episodes",
    "score_grid",
    "stability_score",
]
