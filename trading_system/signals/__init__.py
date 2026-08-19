"""M7: rule-based signal detectors S1 (magnet), S2 (sweep-reversal), S3 (filter)."""

from trading_system.signals.detectors import EVENT_SCHEMA, s1_magnet, s2_sweep_reversal, s3_filter

__all__ = ["EVENT_SCHEMA", "s1_magnet", "s2_sweep_reversal", "s3_filter"]
