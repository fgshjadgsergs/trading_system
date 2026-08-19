"""Stage 3: event studies, leverage-weight calibration ladder, walk-forward.

Public API (plain data in, plain data out — numpy arrays, polars frames,
local dataclasses):

- event_studies: forward_return_paths, bootstrap_effect, mean_path_ci,
  reversal_study, magnet_study, lvn_study, top_decile_touch_events
- weights: capture_rate, flow_divergence, calibration_curve,
  naive_baseline_heat, StaticWeightCalibrator, RollingCalibrator,
  ContextualWeights, compare_ladder
- walkforward: WalkForwardSplitter, tag_regimes, run_walkforward,
  summarize_walkforward
- synthetic: make_world, make_heat_builder (seeded ground truth for tests)
- reports: demo_reports
"""

from trading_system.calibration.event_studies import (
    BootstrapResult,
    bootstrap_effect,
    forward_return_paths,
    lvn_study,
    magnet_study,
    mean_path_ci,
    reversal_study,
    top_decile_touch_events,
)
from trading_system.calibration.walkforward import (
    SymbolData,
    WalkForwardSplit,
    WalkForwardSplitter,
    run_walkforward,
    summarize_walkforward,
    tag_regimes,
)
from trading_system.calibration.weights import (
    ContextualWeights,
    LadderResult,
    RollingCalibrator,
    StaticWeightCalibrator,
    calibration_curve,
    capture_rate,
    compare_ladder,
    flow_divergence,
    naive_baseline_heat,
)

__all__ = [
    "BootstrapResult",
    "ContextualWeights",
    "LadderResult",
    "RollingCalibrator",
    "StaticWeightCalibrator",
    "SymbolData",
    "WalkForwardSplit",
    "WalkForwardSplitter",
    "bootstrap_effect",
    "calibration_curve",
    "capture_rate",
    "compare_ladder",
    "flow_divergence",
    "forward_return_paths",
    "lvn_study",
    "magnet_study",
    "mean_path_ci",
    "naive_baseline_heat",
    "reversal_study",
    "run_walkforward",
    "summarize_walkforward",
    "tag_regimes",
    "top_decile_touch_events",
]
