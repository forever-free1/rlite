"""GRPO advantage computation: group-wise reward normalisation.

For each prompt / task, K responses are sampled.  Advantages are computed
*within* each group:

    A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

This removes prompt-specific difficulty bias — the model learns which of its
responses are *relatively* better, not which prompts are "hard".
"""

from __future__ import annotations

import torch


def compute_group_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute per-sample advantages via group-wise z-score normalisation.

    Args:
        rewards: 1-D tensor of shape ``[B]``, one scalar reward per response.
        group_ids: 1-D long tensor of shape ``[B]``.  Responses that share
            the same group_id belong to the same prompt (i.e. are K siblings).
        eps: Small constant to avoid division by zero when a group has
            zero standard deviation (all rewards identical).

    Returns:
        advantages: 1-D tensor of shape ``[B]`` with zero mean per group.
        metrics: Dict with ``nonzero_advantage_ratio`` and ``filtered_group_ratio``.

    Raises:
        ValueError: If ``rewards`` and ``group_ids`` have different lengths.
    """
    if rewards.shape != group_ids.shape:
        raise ValueError(
            f"Shape mismatch: rewards {rewards.shape} vs group_ids {group_ids.shape}"
        )

    if rewards.numel() == 0:
        return rewards.clone(), {"nonzero_advantage_ratio": 0.0, "filtered_group_ratio": 0.0}

    advantages = torch.empty_like(rewards)
    unique_groups = group_ids.unique()
    n_total_groups = len(unique_groups)
    filtered_groups = 0
    nonzero_count = 0

    for gid in unique_groups:
        mask = group_ids == gid
        group_rewards = rewards[mask]
        mean_r = group_rewards.mean()
        # Population std (unbiased=False) — matches GRPO paper convention
        # and avoids NaN for single-element groups.
        std_r = group_rewards.std(unbiased=False)

        if std_r < eps or mask.sum().item() <= 1:
            # All rewards in this group are identical — set advantages to zero.
            # This is a "filtered group" in DAPO terminology (no learning signal).
            advantages[mask] = 0.0
            filtered_groups += 1
        else:
            adv = (group_rewards - mean_r) / (std_r + eps)
            advantages[mask] = adv
            nonzero_count += mask.sum().item()

    # Metrics
    n = max(rewards.numel(), 1)
    metrics = {
        "nonzero_advantage_ratio": nonzero_count / n,
        "filtered_group_ratio": filtered_groups / max(n_total_groups, 1),
    }
    return advantages, metrics
