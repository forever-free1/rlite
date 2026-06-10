"""Debug plugin implementations for system verification.

These minimal plugins allow the scaffold (config → registry → train loop)
to run without actual model or dataset dependencies.  They now use the
real core types so that the full data flow can be exercised.
"""

from typing import Any, Iterable

from rlite.core.types import Task, Trajectory
from rlite.plugins.base import MetricPlugin, RewardPlugin, TaskPlugin
from rlite.registry import register_metric, register_reward, register_task


# ---------------------------------------------------------------------------
# Debug Task
# ---------------------------------------------------------------------------


@register_task("debug")
class DebugTask(TaskPlugin):
    name = "debug"

    def load_dataset(self, split: str, max_samples: int | None = None) -> Iterable[Task]:
        n = min(max_samples or 4, 4)
        for i in range(n):
            yield Task(
                task_id=f"debug_{i}",
                input={"question": f"debug question {i}"},
                target={"answer": str(i)},
                metadata={"source": "debug", "split": split},
            )

    def build_prompt(self, task: Task) -> str:
        return f"Question: {task.input['question']}\n\nAnswer:"


# ---------------------------------------------------------------------------
# Debug Reward
# ---------------------------------------------------------------------------


@register_reward("debug")
class DebugReward(RewardPlugin):
    name = "debug"

    def score(self, task: Task, trajectory: Trajectory) -> float:
        return 1.0


# ---------------------------------------------------------------------------
# Debug Metric
# ---------------------------------------------------------------------------


@register_metric("debug")
class DebugMetric(MetricPlugin):
    name = "debug"

    def compute(self, trajectories: list[Trajectory]) -> dict[str, float]:
        return {"debug_score": 1.0, "count": float(len(trajectories))}
