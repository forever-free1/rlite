"""GRPO loss components: log-prob ratio, clipped surrogate, KL penalty.

All functions are pure — they operate on tensors and have no state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_log_ratio(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Log-probability ratio: ``exp(new_logp - old_logp)``.

    Args:
        new_logprobs: ``[B, L]`` log-probs under the current policy.
        old_logprobs: ``[B, L]`` log-probs under the old (rollout) policy.

    Returns:
        ratio: ``[B, L]`` tensor of probability ratios.
    """
    return torch.exp(new_logprobs - old_logprobs)


def clipped_surrogate_loss(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    eps_clip: float = 0.2,
    eps_high: float | None = None,
) -> torch.Tensor:
    """PPO-style clipped surrogate objective.

    .. math::
        loss = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)

    Args:
        ratio: ``[B, L]`` probability ratios.
        advantage: ``[B]`` advantages, broadcast to ``[B, L]``.
        eps_clip: Clipping epsilon.

    Returns:
        loss: ``[B, L]`` per-token loss (not yet reduced).
    """
    advantage = advantage.unsqueeze(-1)  # [B] → [B, 1]
    upper = eps_clip if eps_high is None else eps_high
    clipped = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + upper)
    loss = -torch.min(ratio * advantage, clipped * advantage)
    return loss


def masked_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average over every valid response token in the whole batch."""
    mask_f = mask.to(values.dtype)
    return (values * mask_f).sum() / mask_f.sum().clamp(min=1)


def masked_sample_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average within each response first, then give every response equal weight."""
    mask_f = mask.to(values.dtype)
    per_sample = (values * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp(min=1)
    valid_samples = mask_f.sum(dim=-1) > 0
    if not valid_samples.any():
        return values.sum() * 0.0
    return per_sample[valid_samples].mean()


def apply_response_mask(
    loss: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mask out prompt tokens so loss is only computed on response tokens.

    Args:
        loss: ``[B, L]`` per-token loss.
        mask: ``[B, L]`` boolean / float mask (1 = response token, 0 = prompt/padding).
        reduction: ``"mean"`` (average over masked tokens) or ``"sum"``.

    Returns:
        Scalar loss after masking and reduction.
    """
    mask = mask.float()
    masked_loss = loss * mask
    if reduction == "mean":
        denom = mask.sum().clamp(min=1)
        return masked_loss.sum() / denom
    elif reduction == "sum":
        return masked_loss.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def kl_divergence_approx(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Approximate reverse KL: ``exp(old - new) - (old - new) - 1``.

    This is a second-order Taylor expansion of the KL divergence and is
    numerically stable for small log-ratio values.

    Args:
        new_logprobs: ``[B, L]``.
        old_logprobs: ``[B, L]``.
        mask: Optional ``[B, L]`` mask.  If ``None``, all tokens are included.

    Returns:
        Scalar approximate KL divergence (mean over masked tokens).
    """
    log_ratio = old_logprobs - new_logprobs
    kl_per_token = torch.exp(log_ratio) - log_ratio - 1.0
    if mask is not None:
        mask = mask.float()
        kl_per_token = kl_per_token * mask
        denom = mask.sum().clamp(min=1)
        return kl_per_token.sum() / denom
    return kl_per_token.mean()
