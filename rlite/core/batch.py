"""Batch construction: convert RolloutResp into GRPO training tensors.

Responsibilities:
  - Pad variable-length sequences to a common length.
  - Build ``response_mask`` (True = response token).
  - Build ``group_ids`` (responses from the same prompt share an id).
  - Compute ``new_logprobs`` via a model forward pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from rlite.core.rollout_types import RolloutResp
from rlite.core.types import Trajectory


def build_grpo_batch(
    rollout_resp: RolloutResp,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
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
    batch = {
        "rewards": torch.tensor(rewards_list, dtype=torch.float32, device=device),
        "old_logprobs": old_lp_padded,
        "new_logprobs": new_lp_padded,
        "response_mask": resp_mask_padded,
        "group_ids": torch.tensor(group_ids, dtype=torch.long, device=device),
    }
    return batch
