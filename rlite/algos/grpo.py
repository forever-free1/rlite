"""GRPO algorithm: combines advantage computation and loss into one call.

This is a **pure function** — it receives tensors and returns a loss scalar
plus a metrics dict.  There is no model, optimizer, or trainer dependency.
"""

from __future__ import annotations

import torch

from rlite.algos.advantages import compute_group_advantages
from rlite.algos.losses import (
    clipped_surrogate_loss,
    compute_log_ratio,
    kl_divergence_approx,
    masked_sample_mean,
)


def grpo_loss(
    rewards: torch.Tensor,
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: torch.Tensor,
    eps_clip: float = 0.2,
    kl_coef: float = 0.0,
    advantage_eps: float = 1e-6,
    advantages: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the GRPO training loss for one batch.

    Args:
        rewards: ``[B]`` scalar rewards per response.
        old_logprobs: ``[B, L]`` log-probs from the rollout (behaviour) policy.
        new_logprobs: ``[B, L]`` log-probs from the current trainable policy.
        response_mask: ``[B, L]`` boolean mask (True = response token).
        group_ids: ``[B]`` long tensor grouping responses by prompt.
        eps_clip: PPO clip epsilon.
        kl_coef: Weight for optional KL penalty (0 = disabled).
        advantage_eps: Epsilon for advantage std denominator.

    Returns:
        loss: Scalar GRPO loss ready for ``loss.backward()``.
        metrics: Dict with advantage stats, loss components, etc.
    """
    # 1. Compute per-group advantages
    if advantages is None:
        advantages, adv_metrics = compute_group_advantages(
            rewards, group_ids, eps=advantage_eps
        )
    else:
        if advantages.shape != rewards.shape:
            raise ValueError("advantages must have the same shape as rewards")
        adv_metrics = {}

    # 2. Log-probability ratio
    ratio = compute_log_ratio(new_logprobs, old_logprobs)

    # 3. Clipped surrogate loss [B, L]
    policy_loss = clipped_surrogate_loss(ratio, advantages, eps_clip=eps_clip)

    # 4. Mask and reduce
    policy_loss_scalar = masked_sample_mean(policy_loss, response_mask)

    # 5. Optional KL penalty
    kl_val = torch.tensor(0.0, device=policy_loss_scalar.device)
    if kl_coef > 0:
        kl_val = kl_divergence_approx(new_logprobs, old_logprobs, mask=response_mask)
    loss = policy_loss_scalar + kl_coef * kl_val

    # 6. Gather metrics
    metrics = {
        "loss": loss.detach().item(),
        "policy_loss": policy_loss_scalar.detach().item(),
        "kl": kl_val.detach().item(),
        "reward_mean": rewards.mean().detach().item() if rewards.numel() else 0.0,
        "reward_std": rewards.std(unbiased=False).detach().item() if rewards.numel() else 0.0,
        "clip_fraction": (
            (torch.abs(ratio[response_mask] - 1.0) > eps_clip).float().mean().detach().item()
            if response_mask.any() else 0.0
        ),
        **adv_metrics,
    }
    return loss, metrics
