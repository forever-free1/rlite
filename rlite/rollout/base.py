"""Abstract base class for rollout engines."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rlite.core.rollout_types import RolloutReq, RolloutResp


class RolloutEngine(ABC):
    """Interface for generation backends (HF, vLLM, etc.)."""

    @abstractmethod
    def generate(self, req: RolloutReq) -> RolloutResp:
        """Generate responses for a batch of prompts."""
        ...

    @abstractmethod
    def reload_adapter(self, path: str) -> None:
        """Load updated LoRA adapter weights."""
        ...
