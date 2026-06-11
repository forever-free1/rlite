"""GSM8K metric plugin.

Computes evaluation metrics over a batch of trajectories:
  - ``exact_match``: fraction of answers that match the ground truth.
  - ``invalid_format_rate``: fraction of responses where no answer could be extracted.
  - ``avg_response_length``: average length of generated responses.
"""

from __future__ import annotations

from rlite.core.types import Trajectory
from rlite.plugins.base import MetricPlugin
from rlite.registry import register_metric
from rlite.tasks.gsm8k.reward import _numbers_match, extract_answer


@register_metric("gsm8k")
class GSM8KMetric(MetricPlugin):
    """Metric plugin for GSM8K evaluation."""

    name = "gsm8k"

    def compute(self, trajectories: list[Trajectory]) -> dict[str, float]:
        n = len(trajectories)
        if n == 0:
            return {"exact_match": 0.0, "invalid_format_rate": 0.0, "avg_response_length": 0.0}

        correct = 0
        invalid = 0
        total_length = 0

        for traj in trajectories:
            extracted = extract_answer(traj.final_response)
            if extracted is None:
                invalid += 1
            else:
                # Trajectory doesn't carry task reference, so we can't check match here.
                # We count correct rewards instead: if reward >= 1.0 it was an exact match.
                if traj.reward is not None and traj.reward >= 1.0:
                    correct += 1
            # Token-based length (not character count)
            token_ids = traj.steps[0].token_ids if traj.steps else []
            total_length += len(token_ids) if token_ids else 0

        return {
            "exact_match": correct / n,
            "invalid_format_rate": invalid / n,
            "avg_response_length": total_length / n,
        }
