"""GSPO objective with length-normalized sequence importance ratios."""

from __future__ import annotations

import torch


def gspo_loss(rewards, old_logprobs, new_logprobs, response_mask, group_ids,
              *, advantages, eps_clip=0.2, **_) -> tuple[torch.Tensor, dict]:
    if advantages.shape != rewards.shape:
        raise ValueError("advantages must have the same shape as rewards")
    mask = response_mask.to(new_logprobs.dtype)
    sequence_log_ratio = (
        ((new_logprobs - old_logprobs) * mask).sum(dim=-1)
        / mask.sum(dim=-1).clamp(min=1)
    )
    sequence_ratio = torch.exp(sequence_log_ratio)
    unclipped = -sequence_ratio * advantages
    clipped = -torch.clamp(
        sequence_ratio, 1.0 - eps_clip, 1.0 + eps_clip
    ) * advantages
    loss = torch.maximum(unclipped, clipped).mean()
    return loss, {
        "loss": loss.detach().item(),
        "policy_loss": loss.detach().item(),
        "kl": 0.0,
        "clip_fraction": ((sequence_ratio < 1.0 - eps_clip) |
                          (sequence_ratio > 1.0 + eps_clip)).float().mean().item(),
        "sequence_ratio_mean": sequence_ratio.detach().mean().item(),
        "sequence_ratio_std": sequence_ratio.detach().std(unbiased=False).item(),
    }
