"""GSM8K reward plugin.

Scores model trajectories via:
  - **exact match** (1.0): extracted answer matches the ground truth.
  - **format reward** (0.5): response contains a valid answer marker.
  - **invalid penalty** (-0.5): no answer could be extracted.

The reward is additive: ``score = exact_match + format_reward + invalid_penalty``.
This means a correct answer with proper format gets 1.5, while an unparseable
response gets -0.5.
"""

from __future__ import annotations

import re

from rlite.core.types import Task, Trajectory
from rlite.plugins.base import RewardPlugin
from rlite.registry import register_reward


def extract_answer(text: str) -> str | None:
    """Extract a numeric answer from model-generated text.

    Supports the following patterns (tried in order):
    1. ``#### <number>`` — GSM8K standard format
    2. ``\\boxed{<number>}`` — LaTeX format
    3. Last number found in the text

    Returns:
        The extracted number as a string (commas removed), or ``None``.
    """
    if not text:
        return None

    # Pattern 1: #### followed by a number
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")

    # Pattern 2: \boxed{...}
    m = re.search(r"\\boxed\{(-?[\d,]+(?:\.\d+)?)\}", text)
    if m:
        return m.group(1).replace(",", "")

    # Pattern 3: last number in the text
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def _numbers_match(pred: str, target: str) -> bool:
    """Compare two numeric strings, handling float equivalence."""
    try:
        return abs(float(pred) - float(target)) < 1e-6
    except (ValueError, TypeError):
        return pred.strip() == target.strip()


@register_reward("gsm8k")
class GSM8KReward(RewardPlugin):
    """Reward plugin for GSM8K math reasoning tasks."""

    name = "gsm8k"

    def __init__(
        self,
        exact_match_weight: float = 1.0,
        format_reward_weight: float = 0.5,
        invalid_penalty: float = -0.5,
    ):
        self.exact_match_weight = exact_match_weight
        self.format_reward_weight = format_reward_weight
        self.invalid_penalty = invalid_penalty

    def score(self, task: Task, trajectory: Trajectory) -> float:
        response = trajectory.final_response
        extracted = extract_answer(response)
        target = task.target.get("answer", "")

        reward = 0.0

        if extracted is None:
            # No answer found — apply penalty
            reward += self.invalid_penalty
        else:
            # Check format: does the response use a proper answer marker?
            has_format = bool(re.search(r"(####|\\boxed\{)", response))
            if has_format:
                reward += self.format_reward_weight

            # Exact match
            if _numbers_match(extracted, target):
                reward += self.exact_match_weight

        return reward

    def validate(self, rewards: list[float]) -> None:
        """Log a warning if all rewards are identical (degenerate training signal)."""
        if len(set(rewards)) == 1:
            import logging
            logger = logging.getLogger("rlite")
            logger.warning(
                "GSM8KReward: all rewards are identical (%.3f) — training signal may be degenerate.",
                rewards[0] if rewards else float("nan"),
            )
