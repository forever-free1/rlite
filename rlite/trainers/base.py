"""Abstract base class for trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseTrainer(ABC):
    """Interface for training backends (LoRA, DeepSpeed LoRA, etc.)."""

    @abstractmethod
    def train_step(self, loss: torch.Tensor) -> None:
        """Backward pass + optimizer step."""
        ...

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Persist model / adapter to disk."""
        ...

    @abstractmethod
    def get_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        ...
