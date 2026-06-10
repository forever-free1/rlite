"""HuggingFace generate rollout engine.

This is the debug / low-dependency path.  For high-throughput training
use ``VLLMRolloutEngine`` (Phase 7).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from rlite.core.rollout_types import RolloutReq, RolloutResp
from rlite.core.types import Trajectory
from rlite.logging import logger
from rlite.rollout.base import RolloutEngine


class HFRolloutEngine(RolloutEngine):
    """Rollout engine using HuggingFace ``model.generate()``.

    Args:
        model: A HuggingFace ``PreTrainedModel`` (with optional LoRA adapter).
        tokenizer: Corresponding tokenizer.
        adapter_path: Path to a LoRA adapter checkpoint to load on init.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        adapter_path: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self._device = next(model.parameters()).device

        if adapter_path is not None:
            self.reload_adapter(adapter_path)

        # Ensure a pad token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, req: RolloutReq) -> RolloutResp:
        """Generate ``n_samples`` responses per prompt.

        Returns a ``RolloutResp`` where each trajectory carries:
        - ``token_ids``: generated token ids (response only).
        - ``logprobs``: log-probabilities per response token (old policy).
        - ``final_response``: decoded text.
        """
        trajectories: list[Trajectory] = []

        for task, prompt in zip(req.tasks, req.prompts, strict=True):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)
            prompt_len = inputs.input_ids.shape[1]

            for _ in range(req.n_samples):
                # --- generate ---
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature if req.temperature > 0 else 1.0,
                    top_p=req.top_p,
                    do_sample=req.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                full_ids = out[0]  # [prompt + response]
                gen_ids = full_ids[prompt_len:]  # response only

                response_text = self.tokenizer.decode(
                    gen_ids, skip_special_tokens=True
                )

                # --- compute old-policy logprobs ---
                token_lp = self._compute_logprobs(full_ids.unsqueeze(0))
                # response portion only (shifted by 1 for prediction alignment)
                response_lp = token_lp[0, prompt_len - 1: prompt_len - 1 + len(gen_ids)]

                trajectories.append(
                    Trajectory.from_single_response(
                        task_id=task.task_id,
                        prompt=prompt,
                        response=response_text,
                        token_ids=gen_ids.tolist(),
                        logprobs=response_lp.tolist(),
                    )
                )

        logger.debug(
            "HF rollout: %d prompts × %d samples = %d trajectories",
            len(req.tasks),
            req.n_samples,
            len(trajectories),
        )
        return RolloutResp(
            batch_id=req.batch_id,
            trajectories=trajectories,
            policy_version=req.policy_version,
        )

    def reload_adapter(self, path: str) -> None:
        """Load a PeftModel adapter from disk."""
        from peft import PeftModel

        logger.info("Loading LoRA adapter from %s", path)
        self.model = PeftModel.from_pretrained(self.model, path)
        self.model = self.model.merge_and_unload()  # merge for faster inference

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_logprobs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute token-level log-probabilities via a single forward pass.

        Args:
            input_ids: ``[B, L]``.

        Returns:
            ``[B, L-1]`` logprobs: for each position i, the logprob of token at i+1.
        """
        logits = self.model(input_ids).logits  # [B, L, V]
        logprobs = F.log_softmax(logits, dim=-1)
        # gather the logprob of the *actual* next token
        shifted = logprobs[:, :-1, :].gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        )
        return shifted.squeeze(-1)  # [B, L-1]
