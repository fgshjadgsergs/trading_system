"""End-to-end pipeline: recording -> book -> features -> map -> signals -> backtest.

Gate 2 requires this to run as a single script with no manual steps. Until the
modules land, it runs the pipeline over synthetic data through whatever stages
exist, so Gate 0 (stub pipeline assembles) is checkable from day one.
"""

from __future__ import annotations

import argparse

import structlog

from trading_system.core.config import load_config, seed_everything

log = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    log.info("pipeline.start", symbol=args.symbol, seed=cfg["project"]["seed"])
    # Stages are appended here as modules land (see checklist gates).
    log.info("pipeline.done")


if __name__ == "__main__":
    main()
