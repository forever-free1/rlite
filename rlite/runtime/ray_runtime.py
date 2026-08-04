"""Minimal Ray actor runtime for disaggregated GRPO training."""

from __future__ import annotations

import time
from pathlib import Path

import ray


@ray.remote(num_cpus=1)
class ExperienceBuffer:
    """A bounded, version-aware bridge between rollout and training."""

    def __init__(self, capacity: int = 2):
        self.capacity = capacity
        self._items: list = []

    def put(self, response) -> dict:
        if len(self._items) >= self.capacity:
            raise RuntimeError("ExperienceBuffer is full; rollout backpressure required")
        self._items.append(response)
        return {"size": len(self._items), "policy_version": response.policy_version}

    def get(self, expected_policy_version: int):
        if not self._items:
            raise RuntimeError("ExperienceBuffer is empty")
        response = self._items.pop(0)
        if response.policy_version != expected_policy_version:
            raise RuntimeError(
                f"stale rollout: expected policy v{expected_policy_version}, "
                f"got v{response.policy_version}"
            )
        return response

    def stats(self) -> dict:
        return {"size": len(self._items), "capacity": self.capacity}


@ray.remote(num_gpus=1, num_cpus=2)
class RolloutActor:
    """Own vLLM and exactly one Ray-assigned GPU."""

    def __init__(self, cfg):
        from transformers import AutoTokenizer
        from rlite.rollout.vllm_engine import VLLMRolloutEngine

        self.policy_version = 0
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.rollout.model_name)
        self.engine = VLLMRolloutEngine(
            model_name=cfg.rollout.model_name,
            tokenizer=self.tokenizer,
            lora_rank=cfg.trainer.lora_rank,
            tensor_parallel_size=1,
            gpu_memory_utilization=cfg.rollout.gpu_memory_utilization,
            dtype=cfg.rollout.dtype,
            enable_prefix_caching=cfg.rollout.enable_prefix_caching,
            group_admission=cfg.rollout.group_admission,
            max_model_len=cfg.rollout.max_model_len,
        )

    def generate(self, request):
        if request.policy_version >= 0 and request.policy_version != self.policy_version:
            raise RuntimeError(
                f"rollout actor is at policy v{self.policy_version}, "
                f"request requires v{request.policy_version}"
            )
        started = time.perf_counter()
        response = self.engine.generate(request)
        response.metadata["rollout_seconds"] = time.perf_counter() - started
        response.metadata["rollout_policy_version"] = self.policy_version
        return response

    def load_adapter(self, adapter_path: str, policy_version: int) -> dict:
        if policy_version != self.policy_version + 1:
            raise RuntimeError(
                f"non-monotonic weight sync: v{self.policy_version} -> v{policy_version}"
            )
        self.engine.reload_adapter(adapter_path)
        self.policy_version = policy_version
        return {"policy_version": self.policy_version, "adapter_path": adapter_path}

    def identity(self) -> dict:
        import os
        return {"pid": os.getpid(), "gpu_ids": ray.get_gpu_ids()}


@ray.remote(num_gpus=1, num_cpus=2)
class TrainerActor:
    """Own the trainable policy and expose an FSDP-compatible backend contract."""

    def __init__(self, cfg):
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from rlite.trainers.lora_trainer import LoRATrainer

        self.cfg = cfg
        self.policy_version = 0
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.rollout.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            cfg.rollout.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        lora = LoraConfig(
            r=cfg.trainer.lora_rank,
            lora_alpha=cfg.trainer.lora_alpha,
            lora_dropout=cfg.trainer.lora_dropout,
            target_modules=cfg.trainer.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.backend = LoRATrainer(
            model, lora, lr=cfg.trainer.learning_rate,
            max_grad_norm=cfg.trainer.max_grad_norm,
            gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        )
        self.version_dir = Path(cfg.trainer.output_dir) / "adapter_versions"
        self.version_dir.mkdir(parents=True, exist_ok=True)

    def update(self, rollout_response) -> dict:
        import torch
        from rlite.algos.advantages import compute_group_advantages
        from rlite.algos.registry import get_objective
        from rlite.core.batch import iter_grpo_microbatches, prepare_grpo_experience

        if rollout_response.policy_version != self.policy_version:
            raise RuntimeError(
                f"trainer expected rollout v{self.policy_version}, "
                f"got v{rollout_response.policy_version}"
            )
        experience = prepare_grpo_experience(rollout_response, self.tokenizer)
        advantages, advantage_metrics = compute_group_advantages(
            experience.rewards, experience.group_ids
        )
        total_samples = len(experience.input_ids)
        total_response_tokens = sum(item.numel() for item in experience.old_logprobs)
        offset = micro_count = 0
        metrics: dict[str, float] = {}
        self.backend.begin_batch()
        objective = get_objective(self.cfg.algo.name)
        for batch, count in iter_grpo_microbatches(
            experience, self.backend.model,
            max_sequences=self.cfg.trainer.micro_batch_size_per_gpu,
            max_tokens=self.cfg.trainer.max_tokens_per_micro_batch,
        ):
            loss, part = objective(
                batch.rewards, batch.old_logprobs, batch.new_logprobs,
                batch.response_mask, batch.group_ids,
                eps_clip=self.cfg.algo.eps, kl_coef=self.cfg.algo.kl_coef,
                eps_high=self.cfg.algo.eps_high,
                advantages=advantages[offset:offset + count].to(batch.rewards.device),
            )
            if self.cfg.algo.name == "dapo":
                weight = batch.response_mask.sum().item() / total_response_tokens
            else:
                weight = count / total_samples
            self.backend.backward_microbatch(loss, weight)
            for key, value in part.items():
                metrics[key] = metrics.get(key, 0.0) + value * weight
            offset += count
            micro_count += 1
        updated = self.backend.finish_batch()
        adapter_path = None
        if updated:
            self.policy_version += 1
            adapter_path = self.version_dir / f"v{self.policy_version:06d}"
            self.backend.save_checkpoint(str(adapter_path))
        metrics.update(advantage_metrics)
        metrics.update({
            "updated": updated,
            "policy_version": self.policy_version,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "microbatches": micro_count,
            "tokens": experience.token_count,
            "reward_mean": experience.rewards.mean().item(),
        })
        if torch.cuda.is_available():
            metrics["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
        return metrics

    def identity(self) -> dict:
        import os
        return {"pid": os.getpid(), "gpu_ids": ray.get_gpu_ids(), "backend": "lora"}
