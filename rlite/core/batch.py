"""Batch construction: convert RolloutResp into GRPO training tensors.

Responsibilities:
  - Pad variable-length sequences to a common length.
  - Build ``response_mask`` (True = response token).
  - Build ``group_ids`` (responses from the same prompt share an id).
  - Compute ``new_logprobs`` via a model forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from rlite.core.rollout_types import RolloutResp


@dataclass
class GRPOBatch:
    """Typed tensors consumed by all supported policy objectives."""

    rewards: torch.Tensor
    old_logprobs: torch.Tensor
    new_logprobs: torch.Tensor
    response_mask: torch.Tensor
    group_ids: torch.Tensor

    def __getitem__(self, key: str) -> torch.Tensor:
        """Keep the original mapping-style API for compatibility."""
        return getattr(self, key)


@dataclass
class GRPOExperience:
    """CPU-side rollout data; model activations are materialized per microbatch."""

    input_ids: list[torch.Tensor]
    rewards: torch.Tensor
    old_logprobs: list[torch.Tensor]
    group_ids: torch.Tensor

    @property
    def token_count(self) -> int:
        return sum(ids.numel() for ids in self.input_ids)


def prepare_grpo_experience(
    rollout_resp: RolloutResp,
    tokenizer: PreTrainedTokenizer,
) -> GRPOExperience:
    """Tokenize once without creating any full-vocabulary model activations."""
    trajectories = rollout_resp.trajectories
    if not trajectories:
        raise ValueError("Cannot build a training batch from zero trajectories")
    task_to_gid: dict[str, int] = {}
    sequences, old_logprobs, rewards, group_ids = [], [], [], []
    for trajectory in trajectories:
        step = trajectory.steps[0]
        prompt_ids = tokenizer.encode(step.prompt, add_special_tokens=True)
        response_ids = step.token_ids or tokenizer.encode(
            trajectory.final_response, add_special_tokens=False
        )
        stored = step.logprobs or []
        aligned = min(len(response_ids), len(stored))
        if aligned == 0:
            raise ValueError("A trajectory has no aligned response log-probabilities")
        sequences.append(torch.tensor(prompt_ids + list(response_ids[:aligned])))
        old_logprobs.append(torch.tensor(stored[:aligned], dtype=torch.float32))
        task_to_gid.setdefault(trajectory.task_id, len(task_to_gid))
        group_ids.append(task_to_gid[trajectory.task_id])
        rewards.append(trajectory.reward if trajectory.reward is not None else 0.0)
    return GRPOExperience(
        input_ids=sequences,
        rewards=torch.tensor(rewards, dtype=torch.float32),
        old_logprobs=old_logprobs,
        group_ids=torch.tensor(group_ids, dtype=torch.long),
    )


def iter_grpo_microbatches(
    experience: GRPOExperience,
    model: PreTrainedModel,
    max_sequences: int = 1,
    max_tokens: int | None = None,
) -> Iterator[tuple[GRPOBatch, int]]:
    """Yield bounded forward passes and selected-token log-probabilities.

    The returned weight is the number of samples in this microbatch.  Callers
    use it to preserve the full-batch sample-mean objective during backward.
    """
    if max_sequences <= 0 or (max_tokens is not None and max_tokens <= 0):
        raise ValueError("microbatch limits must be positive")
    device = next(model.parameters()).device
    start = 0
    total = len(experience.input_ids)
    while start < total:
        end, token_budget = start, 0
        while end < total and end - start < max_sequences:
            candidate = experience.input_ids[end].numel()
            if end > start and max_tokens is not None and token_budget + candidate > max_tokens:
                break
            token_budget += candidate
            end += 1

        sequences = experience.input_ids[start:end]
        response_lens = [experience.old_logprobs[i].numel() for i in range(start, end)]
        max_len = max(ids.numel() for ids in sequences)
        ids = torch.zeros(len(sequences), max_len, dtype=torch.long, device=device)
        attention = torch.zeros_like(ids, dtype=torch.bool)
        for row, sequence in enumerate(sequences):
            ids[row, :sequence.numel()] = sequence.to(device)
            attention[row, :sequence.numel()] = True

        # Keeping B small is the important bound: HF models still return logits,
        # but the graph no longer contains the entire rollout batch's [B,L,V].
        logits = model(ids, attention_mask=attention).logits[:, :-1, :]
        target_logits = logits.gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        selected = target_logits - torch.logsumexp(logits, dim=-1)
        max_response = max(response_lens)
        new_lp = torch.zeros(len(sequences), max_response, device=device)
        old_lp = torch.zeros_like(new_lp)
        response_mask = torch.zeros_like(new_lp, dtype=torch.bool)
        for row, response_len in enumerate(response_lens):
            prompt_len = sequences[row].numel() - response_len
            new_lp[row, :response_len] = selected[
                row, prompt_len - 1:prompt_len - 1 + response_len
            ]
            old_lp[row, :response_len] = experience.old_logprobs[start + row].to(device)
            response_mask[row, :response_len] = True
        yield GRPOBatch(
            rewards=experience.rewards[start:end].to(device),
            old_logprobs=old_lp,
            new_logprobs=new_lp,
            response_mask=response_mask,
            group_ids=experience.group_ids[start:end].to(device),
        ), end - start
        start = end


def build_grpo_batch(
    rollout_resp: RolloutResp,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> GRPOBatch:
    """Convert a rollout response into a training-ready batch.

    Args:
        rollout_resp: Generated trajectories with ``token_ids`` and ``logprobs``.
        model: Current policy model (for computing ``new_logprobs``).
        tokenizer: Tokenizer for padding / decoding.

    Returns:
        Dict with keys:
        - ``rewards``: ``[B]``
        - ``old_logprobs``: ``[B, L_resp]``  (from rollout)
        - ``new_logprobs``: ``[B, L_resp]``  (from current model)
        - ``response_mask``: ``[B, L_resp]``
        - ``group_ids``: ``[B]``
    """
    device = next(model.parameters()).device
    trajectories = rollout_resp.trajectories
    if not trajectories:
        raise ValueError("Cannot build a training batch from zero trajectories")

    # ---- 1. Build prompt→response mapping -------------------------------------------------
    # Each trajectory knows its prompt and response token_ids.
    full_sequences: list[torch.Tensor] = []
    response_masks: list[torch.Tensor] = []
    group_ids: list[int] = []
    rewards_list: list[float] = []

    # Assign group ids: trajectories for the same task share an id.
    task_to_gid: dict[str, int] = {}
    gid_counter = 0

    for traj in trajectories:
        # Reconstruct full sequence: tokenize prompt + response tokens
        prompt_ids = tokenizer.encode(traj.steps[0].prompt, add_special_tokens=True)
        resp_ids = traj.steps[0].token_ids or tokenizer.encode(
            traj.final_response, add_special_tokens=False
        )
        full_ids = torch.tensor(prompt_ids + list(resp_ids), dtype=torch.long)

        full_sequences.append(full_ids)
        response_masks.append(
            torch.tensor(
                [False] * len(prompt_ids) + [True] * len(resp_ids),
                dtype=torch.bool,
            )
        )

        if traj.task_id not in task_to_gid:
            task_to_gid[traj.task_id] = gid_counter
            gid_counter += 1
        group_ids.append(task_to_gid[traj.task_id])

        rewards_list.append(traj.reward if traj.reward is not None else 0.0)

    # ---- 2. Pad to common length ---------------------------------------------------------
    max_len = max(s.shape[0] for s in full_sequences)
    B = len(full_sequences)

    padded_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
    attn_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
    resp_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)

    for i, (ids, rmask) in enumerate(zip(full_sequences, response_masks)):
        L = ids.shape[0]
        padded_ids[i, :L] = ids.to(device)
        attn_mask[i, :L] = True
        resp_mask[i, :L] = rmask.to(device)

    # ---- 3. Compute new logprobs (with grad for loss.backward()) -------------------------
    logits = model(padded_ids, attention_mask=attn_mask).logits
    new_lp_full = F.log_softmax(logits, dim=-1)
    # Shift: logprob of token t is at position t-1
    new_lp = new_lp_full[:, :-1, :].gather(
        2, padded_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)  # [B, L-1]

    # ---- 4. Extract response-only slices ------------------------------------------------
    # old_logprobs are stored per-trajectory (response-only).
    # We need to pad them to the same response length.
    max_resp_len = max(
        (traj.steps[0].token_ids or [0]).__len__() for traj in trajectories
    )
    old_lp_padded = torch.zeros(B, max_resp_len, device=device)
    new_lp_padded = torch.zeros(B, max_resp_len, device=device)
    resp_mask_padded = torch.zeros(B, max_resp_len, dtype=torch.bool, device=device)

    for i, traj in enumerate(trajectories):
        stored_lp = traj.steps[0].logprobs or []
        r_len = len(stored_lp)
        if r_len == 0:
            continue
        old_lp_padded[i, :r_len] = torch.tensor(stored_lp, device=device)

        # Find response start in padded_ids
        prompt_ids = tokenizer.encode(traj.steps[0].prompt, add_special_tokens=True)
        resp_start = len(prompt_ids) - 1  # last prompt token position in shifted logprobs
        new_lp_padded[i, :r_len] = new_lp[i, resp_start: resp_start + r_len]
        resp_mask_padded[i, :r_len] = True

    # ---- 5. Assemble batch ----------------------------------------------------------------
    if not resp_mask_padded.any():
        raise ValueError("No trajectory contains aligned response log-probabilities")
    return GRPOBatch(
        rewards=torch.tensor(rewards_list, dtype=torch.float32, device=device),
        old_logprobs=old_lp_padded,
        new_logprobs=new_lp_padded,
        response_mask=resp_mask_padded,
        group_ids=torch.tensor(group_ids, dtype=torch.long, device=device),
    )
