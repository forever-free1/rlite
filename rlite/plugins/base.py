"""Base classes for plugin interfaces: Task, Reward, Metric."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Task plugin
# ---------------------------------------------------------------------------


class TaskPlugin(ABC):
    """Responsible for loading datasets and building prompts."""

    name: str = ""

    @abstractmethod
    def load_dataset(self, split: str, max_samples: int | None = None) -> Iterable[Any]:
        """Yield tasks for the given split.

        Args:
            split: Dataset split name (e.g. ``"train"``, ``"test"``).
            max_samples: If set, stop after yielding this many tasks.

        Returns:
            An iterable of ``Task`` objects (see ``rlite.core.types``).
        """
        ...

    @abstractmethod
    def build_prompt(self, task: Any) -> str:
        """Build the prompt string that will be fed to the model.

        Args:
            task: A ``Task`` instance.

        Returns:
            Prompt text ready for tokenisation.
        """
        ...


# ---------------------------------------------------------------------------
# Reward plugin
# ---------------------------------------------------------------------------


class RewardPlugin(ABC):
    """Scores a model trajectory and validates reward distributions."""

    name: str = ""

    @abstractmethod
    def score(self, task: Any, trajectory: Any) -> float:
        """Compute a scalar reward for one trajectory.

        Args:
            task: The ``Task`` that generated this trajectory.
            trajectory: The ``Trajectory`` produced by the model.

        Returns:
            Scalar reward (typically in [0, 1] or can include penalties).
        """
        ...

    def validate(self, rewards: list[float]) -> None:
        """Optional hook: validate reward distribution across a batch.

        Called before advantage computation. Default no-op.
        """
        pass


# ---------------------------------------------------------------------------
# Metric plugin
# ---------------------------------------------------------------------------


class MetricPlugin(ABC):
    """Evaluates model performance over a set of trajectories."""

    name: str = ""

    @abstractmethod
    def compute(self, trajectories: list[Any]) -> dict[str, float]:
        """Compute aggregate metrics over trajectories.

        Args:
            trajectories: List of ``Trajectory`` objects.

        Returns:
            Dict mapping metric names to scalar values.
        """
        ...
