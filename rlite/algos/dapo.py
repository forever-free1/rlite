"""DAPO policy objective: asymmetric clipping and token-level aggregation."""

from __future__ import annotations

import torch


def dapo_loss(rewards, old_logprobs, new_logprobs, response_mask, group_ids,
              *, advantages, eps_clip=0.2, eps_high=0.28, **_) -> tuple[torch.Tensor, dict]:
    if advantages.shape != rewards.shape:
        raise ValueError("advantages must have the same shape as rewards")
    ratio = torch.exp(new_logprobs - old_logprobs)
    advantage = advantages.unsqueeze(-1)
    unclipped = -ratio * advantage
    clipped = -torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_high) * advantage
    per_token = torch.maximum(unclipped, clipped)
    mask = response_mask.to(per_token.dtype)
    loss = (per_token * mask).sum() / mask.sum().clamp(min=1)
    active = response_mask.bool()
    return loss, {
        "loss": loss.detach().item(),
        "policy_loss": loss.detach().item(),
        "kl": 0.0,
        "clip_fraction": ((ratio[active] < 1.0 - eps_clip) |
                          (ratio[active] > 1.0 + eps_high)).float().mean().item(),
        "clip_fraction_low": (ratio[active] < 1.0 - eps_clip).float().mean().item(),
    }
