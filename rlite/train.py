"""train.py — Full LoRA-GRPO training loop.

Usage:
    python -m rlite.train --config configs/gsm8k_grpo_lora_hf.yaml
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlite.algos.grpo import grpo_loss
from rlite.config import load_config
from rlite.core.batch import iter_grpo_microbatches, prepare_grpo_experience
from rlite.algos.advantages import compute_group_advantages
from rlite.core.rollout_types import RolloutReq
from rlite.logging import logger, setup_logging
from rlite.registry import metric_registry, reward_registry, task_registry
from rlite.trainers.lora_trainer import LoRATrainer


class ShuffledTaskSampler:
    """Seeded cyclic sampler that loads a dataset once and reshuffles per epoch."""

    def __init__(self, tasks, seed: int):
        if not tasks:
            raise ValueError("Training dataset is empty")
        self.tasks = list(tasks)
        self.rng = random.Random(seed)
        self.order: list[int] = []
        self.position = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self.order = list(range(len(self.tasks)))
        self.rng.shuffle(self.order)
        self.position = 0

    def sample(self, size: int):
        result = []
        while len(result) < size:
            if self.position == len(self.order):
                self._reshuffle()
            take = min(size - len(result), len(self.order) - self.position)
            indices = self.order[self.position:self.position + take]
            result.extend(self.tasks[i] for i in indices)
            self.position += take
        return result


def score_trajectories(trajectories, tasks, reward_plugin) -> None:
    """Score by explicit task id; trajectory ordering is never assumed."""
    task_by_id = {task.task_id: task for task in tasks}
    for trajectory in trajectories:
        try:
            task = task_by_id[trajectory.task_id]
        except KeyError as exc:
            raise ValueError(
                f"Rollout returned unknown task_id {trajectory.task_id!r}"
            ) from exc
        trajectory.reward = reward_plugin.score(task, trajectory)


def _load_plugins(task_name: str) -> None:
    """Import the configured task package to trigger plugin registration."""
    import rlite.plugins.task  # noqa: F401  # debug fallback
    if task_name != "debug":
        try:
            importlib.import_module(f"rlite.tasks.{task_name}")
        except ImportError as exc:
            logger.warning("Could not import rlite.tasks.%s: %s", task_name, exc)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="rlite train - lightweight LoRA-GRPO training"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    random.seed(cfg.trainer.seed)
    torch.manual_seed(cfg.trainer.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.trainer.seed)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir)
    logger.info("=" * 60)
    logger.info("rlite train — LoRA-GRPO on %s", cfg.task.name)
    logger.info("Config: %s", config_path)
    logger.info("=" * 60)

    # ---- Device -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Plugins ----------------------------------------------------------
    _load_plugins(cfg.task.name)
    task_plugin = task_registry.create(cfg.task.name)
    reward_plugin = reward_registry.create(cfg.reward.name, **cfg.reward.kwargs)
    metric_plugin = metric_registry.create(cfg.task.name)

    logger.info("Task:   %s", type(task_plugin).__name__)
    logger.info("Reward: %s", type(reward_plugin).__name__)
    logger.info("Metric: %s", type(metric_plugin).__name__)

    # ---- Model & tokenizer ------------------------------------------------
    model_name = cfg.rollout.model_name
    logger.info("Loading base model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # decoder-only models need left padding for batched generation

    # ---- Trainer (LoRA) ---------------------------------------------------
    from peft import LoraConfig

    lora_config = LoraConfig(
        r=cfg.trainer.lora_rank,
        lora_alpha=cfg.trainer.lora_alpha,
        lora_dropout=cfg.trainer.lora_dropout,
        target_modules=cfg.trainer.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer = LoRATrainer(
        model, lora_config, lr=cfg.trainer.learning_rate,
        max_grad_norm=cfg.trainer.max_grad_norm,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
    )
    # ---- Rollout engine ---------------------------------------------------
    # rollout_model is always the PeftModel (needed for build_grpo_batch)
    rollout_model = trainer.model

    if cfg.rollout.engine == "vllm":
        from rlite.rollout.vllm_engine import VLLMRolloutEngine

        rollout_engine = VLLMRolloutEngine(
            model_name=cfg.rollout.model_name or model_name,
            tokenizer=tokenizer,
            lora_rank=cfg.trainer.lora_rank,
            tensor_parallel_size=cfg.rollout.tensor_parallel_size,
            gpu_memory_utilization=cfg.rollout.gpu_memory_utilization,
            dtype=cfg.rollout.dtype,
            enable_prefix_caching=cfg.rollout.enable_prefix_caching,
            group_admission=cfg.rollout.group_admission,
            max_model_len=cfg.rollout.max_model_len,
        )
        logger.info("Rollout: VLLMRolloutEngine (engine=vllm)")
    else:
        from rlite.rollout.hf_engine import HFRolloutEngine

        rollout_engine = HFRolloutEngine(rollout_model, tokenizer)
        logger.info("Rollout: HFRolloutEngine (engine=hf)")

    # ---- Eval data (fixed subset for monitoring) --------------------------
    eval_tasks = list(task_plugin.load_dataset(
        split="test", max_samples=min(cfg.task.max_samples or 100, 100)
    ))
    logger.info("Eval tasks: %d", len(eval_tasks))

    train_tasks = list(task_plugin.load_dataset(
        split=cfg.task.split, max_samples=cfg.task.max_samples
    ))
    sampler = ShuffledTaskSampler(train_tasks, cfg.trainer.seed)
    logger.info("Training task pool: %d (seed=%d)", len(train_tasks), cfg.trainer.seed)

    # ---- Training loop ----------------------------------------------------
    total_steps = cfg.trainer.train_steps
    batch_size = cfg.trainer.batch_size  # number of prompts per step
    K = cfg.rollout.n_samples
    policy_version = 0
    output_dir = Path(cfg.trainer.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Train steps: %d | prompts/step: %d | K: %d | LR: %g",
                total_steps, batch_size, K, cfg.trainer.learning_rate)
    logger.info("Output dir: %s", output_dir)

    # A pre-training baseline is required to interpret later evaluation points.
    _run_eval(
        eval_tasks, task_plugin, reward_plugin, metric_plugin,
        rollout_engine, rollout_model, tokenizer, cfg, 0,
    )

    for step in range(1, total_steps + 1):
        t_start = time.time()

        # --- 1-3. Sample, rollout and score. ---
        tasks = sampler.sample(batch_size)
        prompts = [task_plugin.build_prompt(t) for t in tasks]
        req = RolloutReq(
            batch_id=f"step_{step:04d}", tasks=tasks, prompts=prompts,
            n_samples=K, temperature=cfg.rollout.temperature,
            top_p=cfg.rollout.top_p, max_tokens=cfg.rollout.max_tokens,
            policy_version=policy_version,
        )
        rollout_resp = rollout_engine.generate(req)
        score_trajectories(rollout_resp.trajectories, tasks, reward_plugin)

        # --- 4. Build batch & compute loss ---
        experience = prepare_grpo_experience(rollout_resp, tokenizer)
        advantages, advantage_metrics = compute_group_advantages(
            experience.rewards, experience.group_ids
        )
        total_samples = len(experience.input_ids)
        metric_sums: dict[str, float] = {}
        microbatch_count = 0
        sample_offset = 0
        trainer.begin_batch()
        for batch, micro_samples in iter_grpo_microbatches(
            experience,
            rollout_model,
            max_sequences=cfg.trainer.micro_batch_size_per_gpu,
            max_tokens=cfg.trainer.max_tokens_per_micro_batch,
        ):
            microbatch_count += 1
            micro_advantages = advantages[
                sample_offset:sample_offset + micro_samples
            ].to(batch.rewards.device)
            loss, micro_metrics = grpo_loss(
                batch.rewards, batch.old_logprobs, batch.new_logprobs,
                batch.response_mask, batch.group_ids,
                eps_clip=cfg.algo.eps, kl_coef=cfg.algo.kl_coef,
                advantages=micro_advantages,
            )
            weight = micro_samples / total_samples
            trainer.backward_microbatch(loss, weight)
            for key, value in micro_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value * weight
            sample_offset += micro_samples

        algo_metrics = {**metric_sums, **advantage_metrics}
        algo_metrics["loss"] = metric_sums.get("loss", 0.0)

        # --- 5. Train step ---
        updated = trainer.finish_batch()
        if updated:
            policy_version += 1

        # --- 5b. Sync adapter to vLLM engine (vLLM has its own model copy) ---
        if cfg.rollout.engine == "vllm" and updated:
            adapter_tmp_dir = output_dir / "adapter_current"
            adapter_tmp_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(adapter_tmp_dir))
            rollout_engine.reload_adapter(str(adapter_tmp_dir))

        # --- 6. Log ---
        t_elapsed = time.time() - t_start
        if step % cfg.trainer.log_steps == 0 or step == 1:
            reward_mean = experience.rewards.mean().item()
            logger.info(
                "[step %4d/%d] loss=%7.4f | reward=%6.3f | "
                "nonzero_adv=%5.3f | kl=%6.4f | microbatches=%d | "
                "tokens=%d | time=%5.1fs",
                step, total_steps, algo_metrics["loss"], reward_mean,
                algo_metrics["nonzero_advantage_ratio"], algo_metrics["kl"],
                microbatch_count, experience.token_count, t_elapsed,
            )

        # --- 7. Eval ---
        if step % cfg.trainer.eval_steps == 0:
            _run_eval(
                eval_tasks, task_plugin, reward_plugin, metric_plugin,
                rollout_engine, rollout_model, tokenizer, cfg, step,
            )

        # --- 8. Save checkpoint ---
        if step % cfg.trainer.save_steps == 0:
            ckpt_dir = output_dir / f"step_{step}"
            trainer.save_checkpoint(str(ckpt_dir))


def _run_eval(eval_tasks, task_plugin, reward_plugin, metric_plugin,
              rollout_engine, rollout_model, tokenizer, cfg, step):
    """Run evaluation on a fixed set of tasks."""
    prompts = [task_plugin.build_prompt(t) for t in eval_tasks]
    req = RolloutReq(
        batch_id=f"eval_step_{step}",
        tasks=eval_tasks,
        prompts=prompts,
        n_samples=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=cfg.rollout.max_tokens,
        policy_version=-1,
    )
    rollout_resp = rollout_engine.generate(req)

    for traj, task in zip(rollout_resp.trajectories, eval_tasks):
        traj.reward = reward_plugin.score(task, traj)

    metrics = metric_plugin.compute(rollout_resp.trajectories)
    logger.info(
        "[eval  %4d] exact_match=%.3f | invalid_rate=%.3f | avg_len=%d",
        step,
        metrics.get("exact_match", 0.0),
        metrics.get("invalid_format_rate", 0.0),
        int(metrics.get("avg_response_length", 0)),
    )


if __name__ == "__main__":
    main()
