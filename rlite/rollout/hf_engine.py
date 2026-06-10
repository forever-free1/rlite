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

        # Ensure a pad token is set and padding side is left for decoder-only models
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, req: RolloutReq) -> RolloutResp:
        """Generate ``n_samples`` responses per prompt (batched).

        All prompts × samples are generated in a single ``model.generate()``
        call for maximum GPU throughput.
        """
        K = req.n_samples
        N = len(req.prompts)
        device = self._device

        # ---- 1. Build batched input (repeat each prompt K times) -------
        flat_prompts = [p for p in req.prompts for _ in range(K)]
        tokenized = self.tokenizer(
            flat_prompts, return_tensors="pt", padding=True
        ).to(device)
        input_ids = tokenized.input_ids  # [B, L_pad]
        attn_mask = tokenized.attention_mask
        padded_len = input_ids.shape[1]  # all inputs share this length after padding

        # ---- 2. Generate all responses in one batch --------------------
        gen_kwargs = {
            "max_new_tokens": req.max_tokens,
            "top_p": req.top_p,
            "do_sample": req.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if req.temperature > 0:
            gen_kwargs["temperature"] = req.temperature

        out = self.model.generate(**tokenized, **gen_kwargs)  # [B, L_full]

        # ---- 3. Compute old-policy logprobs (batched forward) ---------
        all_logprobs = self._compute_logprobs(out)  # [B, L_full-1]

        # ---- 4. Extract per-trajectory results -------------------------
        # With left-padding, response starts at padded_len for all examples.
        trajectories: list[Trajectory] = []
        idx = 0
        for task, prompt in zip(req.tasks, req.prompts, strict=True):
            for _ in range(K):
                full_ids = out[idx]
                gen_ids = full_ids[padded_len:]
                gen_ids = gen_ids[gen_ids != self.tokenizer.pad_token_id]  # trim trailing pads
                response_text = self.tokenizer.decode(
                    gen_ids, skip_special_tokens=True
                )
                # response logprobs: position padded_len-1 predicts token at padded_len
                resp_start = padded_len - 1
                response_lp = all_logprobs[idx, resp_start: resp_start + len(gen_ids)]

                trajectories.append(
                    Trajectory.from_single_response(
                        task_id=task.task_id,
                        prompt=prompt,
                        response=response_text,
                        token_ids=gen_ids.tolist(),
                        logprobs=response_lp.tolist(),
                    )
                )
                idx += 1

        logger.debug(
            "HF rollout: %d prompts × %d samples = %d trajectories",
            N, K, len(trajectories),
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
