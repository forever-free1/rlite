"""eval.py — Evaluation entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rlite.config import load_config
from rlite.logging import logger, setup_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="rlite eval — evaluate a trained LoRA adapter"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to LoRA adapter checkpoint (optional)",
    )
    args = parser.parse_args(argv)

    # 1. load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)

    # 2. setup logging
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir)
    logger.info("rlite eval starting (config=%s)", config_path)
    if args.checkpoint:
        logger.info("Checkpoint: %s", args.checkpoint)

    # 3. import debug plugins
    import rlite.plugins.task  # noqa: F401

    # 4. resolve plugins
    from rlite.registry import metric_registry, reward_registry, task_registry

    task_plugin = task_registry.create(cfg.task.name)
    reward_plugin = reward_registry.create(cfg.reward.name)
    metric_plugin = metric_registry.create(cfg.reward.name)

    from rlite.core.types import Trajectory

    # 5. eval dry-run: create simulated trajectories using real types
    tasks = list(task_plugin.load_dataset(split=cfg.task.split, max_samples=cfg.task.max_samples))
    logger.info("Loaded %d tasks for evaluation", len(tasks))

    trajectories = [
        Trajectory.from_single_response(
            task_id=t.task_id,
            prompt=task_plugin.build_prompt(t),
            response=f"simulated response for {t.task_id}",
        )
        for t in tasks
    ]

    # compute rewards
    for traj, task in zip(trajectories, tasks):
        traj.reward = reward_plugin.score(task, traj)

    # compute metrics
    metrics = metric_plugin.compute(trajectories)
    logger.info("Eval metrics: %s", metrics)

    logger.info("Eval dry-run complete.")


if __name__ == "__main__":
    main()
