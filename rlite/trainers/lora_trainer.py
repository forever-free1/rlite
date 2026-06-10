"""PEFT LoRA trainer: wraps a base model with a LoRA adapter and provides
a minimal training step for GRPO.

This trainer is intentionally minimal — no dataloader, no scheduler, no
distributed wrapper.  Those belong in higher-level orchestration (Ray, etc.).
"""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

from rlite.logging import logger
from rlite.trainers.base import BaseTrainer


class LoRATrainer(BaseTrainer):
    """Adds a LoRA adapter to *model* and exposes a single ``train_step``.

    Args:
        model: Frozen base model.
        lora_config: PEFT ``LoraConfig``.  If ``None``, sensible defaults are used.
        lr: Learning rate for AdamW.
        max_grad_norm: Gradient clipping threshold.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        lora_config: LoraConfig | None = None,
        lr: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        if lora_config is None:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )

        self.model = get_peft_model(model, lora_config)
        self._trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            "LoRA trainable params: %s (%.2f M)",
            self._trainable_params,
            self._trainable_params / 1e6,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr
        )
        self.max_grad_norm = max_grad_norm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_step(self, loss: torch.Tensor) -> None:
        """Standard backward → clip → step."""
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

    def save_checkpoint(self, path: str) -> None:
        """Save LoRA adapter weights."""
        self.model.save_pretrained(path)
        logger.info("Checkpoint saved to %s", path)

    def get_trainable_params(self) -> int:
        return self._trainable_params

    def get_adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a detached copy of LoRA weights (for in-place sync)."""
        return {k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
                if "lora" in k}
