"""Core data types: Task, Step, Trajectory, RewardResult, MetricResult.

These are the shared contracts that all plugins and the training loop agree on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """Represents one "unit of ability" the model should learn.

    Attributes:
        task_id: Unique identifier, e.g. ``"gsm8k_001"``.
        input: Free-form input dict (e.g. ``{"question": "..."}``).
        target: Expected answer / label (used by reward plugins).
        metadata: Extra info (source dataset, difficulty, etc.).
    """

    task_id: str
    input: dict[str, Any] = field(default_factory=dict)
    target: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """A single response from the model for a given prompt.

    For simple tasks (e.g. GSM8K) there is exactly one step per trajectory.
    Multi-turn / agentic tasks may have multiple steps.
    """

    prompt: str
    response: str
    token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """A model's complete response to a task (one or more steps).

    Attributes:
        task_id: Links back to the originating ``Task``.
        steps: Ordered list of model responses.
        final_response: Convenience accessor for the last response.
        reward: Scalar reward, filled in by ``RewardPlugin.score()``.
        advantage: Computed advantage, filled in by the algorithm.
        metadata: Extra info (generation config, timing, etc.).
    """

    task_id: str
    steps: list[Step] = field(default_factory=list)
    final_response: str = ""
    reward: float | None = None
    advantage: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_single_response(
        cls,
        task_id: str,
        prompt: str,
        response: str,
        token_ids: list[int] | None = None,
        logprobs: list[float] | None = None,
        **metadata: Any,
    ) -> Trajectory:
        """Factory for the common case: one prompt → one response."""
        step = Step(
            prompt=prompt,
            response=response,
            token_ids=token_ids,
            logprobs=logprobs,
        )
        return cls(
            task_id=task_id,
            steps=[step],
            final_response=response,
            metadata=dict(metadata),
        )


# ---------------------------------------------------------------------------
# RewardResult
# ---------------------------------------------------------------------------


@dataclass
class RewardResult:
    """Structured reward from a ``RewardPlugin``.

    Attributes:
        score: The total scalar reward for this trajectory.
        details: Component breakdown (e.g. ``{"exact_match": 1.0, "format": 0.5}``).
        valid: Whether the response was parseable / processable.
    """

    score: float
    details: dict[str, float] = field(default_factory=dict)
    valid: bool = True


# ---------------------------------------------------------------------------
# MetricResult
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """Named metrics computed over a batch of trajectories.

    Attributes:
        metrics: Dict mapping metric name → value.
        n_samples: Number of trajectories used for computation.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
