"""eval.py — Evaluation entry point."""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from pathlib import Path

from rlite.config import RLiteConfig, load_config
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlite.core.rollout_types import RolloutReq
from rlite.logging import logger, setup_logging


def _load_plugins(cfg: RLiteConfig):
    """Import the configured task package to trigger plugin registration."""
    import rlite.plugins.task  # noqa: F401  # debug fallback

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

    # 3. import plugins
    _load_plugins(cfg)

    # 4. resolve plugins from registry
    from rlite.registry import metric_registry, reward_registry, task_registry

    task_plugin = task_registry.create(cfg.task.name)
    reward_plugin = reward_registry.create(cfg.reward.name, **cfg.reward.kwargs)
    metric_plugin = metric_registry.create(cfg.task.name)

    random.seed(cfg.trainer.seed)
    torch.manual_seed(cfg.trainer.seed)

    # 5. Load the real policy and rollout backend.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.rollout.model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.rollout.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if cfg.rollout.engine == "vllm":
        from rlite.rollout.vllm_engine import VLLMRolloutEngine
        engine = VLLMRolloutEngine(
            cfg.rollout.model_name,
            tokenizer,
            lora_rank=cfg.trainer.lora_rank,
            tensor_parallel_size=cfg.rollout.tensor_parallel_size,
            gpu_memory_utilization=cfg.rollout.gpu_memory_utilization,
            dtype=cfg.rollout.dtype,
            enable_prefix_caching=cfg.rollout.enable_prefix_caching,
            group_admission=cfg.rollout.group_admission,
            max_model_len=cfg.rollout.max_model_len,
        )
        if args.checkpoint:
            engine.reload_adapter(args.checkpoint)
    else:
        if args.checkpoint:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.checkpoint)
        from rlite.rollout.hf_engine import HFRolloutEngine
        engine = HFRolloutEngine(model, tokenizer)

    # 6. Generate and score real responses.
    tasks = list(task_plugin.load_dataset(split=cfg.task.split, max_samples=cfg.task.max_samples))
    logger.info("Loaded %d tasks for evaluation", len(tasks))
    prompts = [task_plugin.build_prompt(t) for t in tasks]
    response = engine.generate(RolloutReq(
        batch_id="eval",
        tasks=tasks,
        prompts=prompts,
        n_samples=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=cfg.rollout.max_tokens,
        policy_version=-1,
    ))
    task_by_id = {t.task_id: t for t in tasks}
    for trajectory in response.trajectories:
        trajectory.reward = reward_plugin.score(task_by_id[trajectory.task_id], trajectory)

    metrics = metric_plugin.compute(response.trajectories)
    logger.info("Eval metrics: %s", metrics)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
