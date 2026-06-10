"""Rollout types: RolloutReq and RolloutResp.

Inspired by UniRL's typed runtime idea — a clean contract between
the training loop and the rollout engine (HF or vLLM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlite.core.types import Task, Trajectory


@dataclass
class RolloutReq:
    """A batch rollout request sent to the generation engine.

    Attributes:
        batch_id: Unique identifier for this batch (for logging / tracking).
        tasks: The tasks to generate responses for.
        prompts: Pre-built prompt strings (one per task).
        n_samples: Number of responses per prompt (K in GRPO).
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        max_tokens: Maximum new tokens per response.
        policy_version: Monotonic counter for the current policy (used for cache invalidation).
        metadata: Extra info (e.g. training step number).
    """

    batch_id: str
    tasks: list[Task] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    n_samples: int = 4
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    policy_version: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutResp:
    """A batch rollout response from the generation engine.

    Attributes:
        batch_id: Matches the request batch_id.
        trajectories: Generated trajectories (n_samples per task).
        policy_version: The policy version that generated these responses.
        metadata: Extra info (timing, GPU utilisation, etc.).
    """

    batch_id: str
    trajectories: list[Trajectory] = field(default_factory=list)
    policy_version: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
