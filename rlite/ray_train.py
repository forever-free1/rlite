"""Ray-orchestrated vLLM -> Buffer -> GRPO Trainer -> LoRA sync loop."""

from __future__ import annotations

import argparse
import importlib
import random
import time
from pathlib import Path

import ray

from rlite.config import load_config
from rlite.core.rollout_types import RolloutReq, RolloutResp
from rlite.logging import logger, setup_logging
from rlite.registry import metric_registry, reward_registry, task_registry
from rlite.runtime.ray_runtime import ExperienceBuffer, RolloutActor, TrainerActor
from rlite.train import ShuffledTaskSampler, score_trajectories


def _training_rollout(cfg, step, policy_version, sampler, task_plugin,
                      reward_plugin, rollout):
    """Generate one logical batch; DAPO replenishes zero-variance groups."""
    accepted = []
    rounds = 0
    target_groups = cfg.trainer.batch_size
    while len({trajectory.task_id for trajectory in accepted}) < target_groups:
        rounds += 1
        missing = target_groups - len({trajectory.task_id for trajectory in accepted})
        tasks = sampler.sample(missing)
        request = RolloutReq(
            batch_id=f"ray_step_{step:04d}_r{rounds}", tasks=tasks,
            prompts=[task_plugin.build_prompt(task) for task in tasks],
            n_samples=cfg.rollout.n_samples,
            temperature=cfg.rollout.temperature, top_p=cfg.rollout.top_p,
            max_tokens=cfg.rollout.max_tokens, policy_version=policy_version,
        )
        response = ray.get(rollout.generate.remote(request))
        score_trajectories(response.trajectories, tasks, reward_plugin)
        if cfg.algo.name != "dapo":
            accepted.extend(response.trajectories)
            break
        by_task: dict[str, list] = {}
        for trajectory in response.trajectories:
            by_task.setdefault(trajectory.task_id, []).append(trajectory)
        for trajectories in by_task.values():
            rewards = {trajectory.reward for trajectory in trajectories}
            if len(rewards) > 1:
                accepted.extend(trajectories)
        if rounds >= cfg.algo.dynamic_sampling_max_rounds:
            raise RuntimeError(
                f"DAPO dynamic sampling found only "
                f"{len({t.task_id for t in accepted})}/{target_groups} effective groups "
                f"after {rounds} rounds"
            )
    return RolloutResp(
        batch_id=f"ray_step_{step:04d}", trajectories=accepted,
        policy_version=policy_version,
        metadata={"dynamic_sampling_rounds": rounds},
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ray-disaggregated GRPO training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--log-dir")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.train_steps is not None:
        cfg.trainer.train_steps = args.train_steps
    if args.max_samples is not None:
        cfg.task.max_samples = args.max_samples
    if args.max_tokens is not None:
        cfg.rollout.max_tokens = args.max_tokens
    if args.output_dir is not None:
        cfg.trainer.output_dir = args.output_dir
    if args.log_dir is not None:
        cfg.logging.log_dir = args.log_dir
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    random.seed(cfg.trainer.seed)
    importlib.import_module(f"rlite.tasks.{cfg.task.name}")
    task_plugin = task_registry.create(cfg.task.name)
    reward_plugin = reward_registry.create(cfg.reward.name, **cfg.reward.kwargs)
    metric_plugin = metric_registry.create(cfg.task.name)

    train_tasks = list(task_plugin.load_dataset(cfg.task.split, cfg.task.max_samples))
    eval_tasks = list(task_plugin.load_dataset("test", min(cfg.task.max_samples or 100, 100)))
    sampler = ShuffledTaskSampler(train_tasks, cfg.trainer.seed)

    ray.init(num_gpus=2, ignore_reinit_error=True)
    buffer = ExperienceBuffer.remote(capacity=2)
    trainer = TrainerActor.remote(cfg)
    rollout = RolloutActor.remote(cfg)
    identities = ray.get([trainer.identity.remote(), rollout.identity.remote()])
    logger.info("Ray topology ready: trainer=%s rollout=%s", *identities)

    policy_version = 0
    output_dir = Path(cfg.trainer.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_request = RolloutReq(
        batch_id="ray_eval_0000", tasks=eval_tasks,
        prompts=[task_plugin.build_prompt(task) for task in eval_tasks],
        n_samples=1, temperature=0.0, top_p=1.0,
        max_tokens=cfg.rollout.max_tokens, policy_version=policy_version,
    )
    baseline = ray.get(rollout.generate.remote(baseline_request))
    for trajectory, task in zip(baseline.trajectories, eval_tasks):
        trajectory.reward = reward_plugin.score(task, trajectory)
    baseline_values = metric_plugin.compute(baseline.trajectories)
    logger.info("[ray eval %4d] exact_match=%.3f invalid_rate=%.3f", 0,
                baseline_values.get("exact_match", 0.0),
                baseline_values.get("invalid_format_rate", 0.0))

    for step in range(1, cfg.trainer.train_steps + 1):
        started = time.perf_counter()
        response = _training_rollout(
            cfg, step, policy_version, sampler, task_plugin, reward_plugin, rollout
        )
        ray.get(buffer.put.remote(response))
        experience = ray.get(buffer.get.remote(policy_version))
        metrics = ray.get(trainer.update.remote(experience))
        if metrics["updated"]:
            policy_version = metrics["policy_version"]
            ray.get(rollout.load_adapter.remote(
                metrics["adapter_path"], policy_version
            ))
        logger.info(
            "[ray step %4d/%d] policy=v%d loss=%7.4f reward=%6.3f "
            "microbatches=%d tokens=%d sampling_rounds=%d peak_mem=%.2fGB time=%.1fs",
            step, cfg.trainer.train_steps, policy_version,
            metrics.get("loss", 0.0), metrics["reward_mean"],
            metrics["microbatches"], metrics["tokens"],
            response.metadata["dynamic_sampling_rounds"],
            metrics.get("peak_memory_gb", 0.0), time.perf_counter() - started,
        )
        if step % cfg.trainer.eval_steps == 0:
            eval_request = RolloutReq(
                batch_id=f"ray_eval_{step:04d}", tasks=eval_tasks,
                prompts=[task_plugin.build_prompt(task) for task in eval_tasks],
                n_samples=1, temperature=0.0, top_p=1.0,
                max_tokens=cfg.rollout.max_tokens, policy_version=policy_version,
            )
            evaluated = ray.get(rollout.generate.remote(eval_request))
            for trajectory, task in zip(evaluated.trajectories, eval_tasks):
                trajectory.reward = reward_plugin.score(task, trajectory)
            values = metric_plugin.compute(evaluated.trajectories)
            logger.info("[ray eval %4d] exact_match=%.3f invalid_rate=%.3f",
                        step, values.get("exact_match", 0.0),
                        values.get("invalid_format_rate", 0.0))

    ray.shutdown()


if __name__ == "__main__":
    main()
