"""train.py — Main training entry point."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from rlite.config import RLiteConfig, load_config
from rlite.logging import logger, setup_logging


def _load_plugins(cfg: RLiteConfig):
    """Import the configured task package to trigger plugin registration.

    Always imports the debug fallback first, then the requested task package.
    """
    # Always load debug plugins as fallback
    import rlite.plugins.task  # noqa: F401

    # Dynamically import the task-specific package (e.g. rlite.tasks.gsm8k)
    task_name = cfg.task.name
    if task_name != "debug":
        try:
            importlib.import_module(f"rlite.tasks.{task_name}")
        except ImportError as exc:
            logger.warning(
                "Could not import rlite.tasks.%s: %s. Falling back to debug plugins.",
                task_name,
                exc,
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="rlite train — pluggable LoRA-GRPO/DAPO RL training"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
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
    logger.info("rlite train starting (config=%s)", config_path)

    # 3. import plugins (task-specific + debug fallback)
    _load_plugins(cfg)

    # 4. resolve plugins from registry
    from rlite.registry import metric_registry, reward_registry, task_registry

    task_cls = task_registry.get(cfg.task.name)
    reward_cls = reward_registry.get(cfg.reward.name)
    metric_cls = metric_registry.get(cfg.task.name)  # metric follows task name

    logger.info("Task plugin:    %s", task_cls.__name__)
    logger.info("Reward plugin:  %s", reward_cls.__name__)
    logger.info("Metric plugin:  %s", metric_cls.__name__)
    logger.info("Algo:           %s", cfg.algo.name)
    logger.info("Rollout engine: %s", cfg.rollout.engine)
    logger.info("Trainer:        %s", cfg.trainer.method)
    logger.info("LoRA rank:      %s", cfg.trainer.lora_rank)

    # 5. instantiate plugins
    task_plugin = task_cls()
    reward_plugin = reward_cls(**cfg.reward.kwargs)
    metric_plugin = metric_cls()

    # 6. dry-run: load tasks, build prompts, simulate rollout + reward
    tasks = list(task_plugin.load_dataset(split=cfg.task.split, max_samples=cfg.task.max_samples))
    logger.info("Loaded %d tasks", len(tasks))

    for i, task in enumerate(tasks):
        prompt = task_plugin.build_prompt(task)
        logger.debug("Task %d prompt (first 80 chars): %s...", i, prompt[:80])

    logger.info("Dry-run complete — scaffold is wired correctly.")
    logger.info("Ready for model training.")


if __name__ == "__main__":
    main()
