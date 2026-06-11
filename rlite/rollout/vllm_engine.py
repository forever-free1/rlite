"""vLLM rollout engine — high-throughput generation with LoRA support.

Uses the vLLM offline ``LLM`` API with ``enable_lora=True``.  After each
training step the LoRA adapter is saved to a stable temp directory and
vLLM picks it up via ``LoRARequest`` — no engine restart required.
"""

from __future__ import annotations

from rlite.core.rollout_types import RolloutReq, RolloutResp
from rlite.core.types import Trajectory
from rlite.logging import logger
from rlite.rollout.base import RolloutEngine


class VLLMRolloutEngine(RolloutEngine):
    """Rollout engine using the vLLM offline API with per-request LoRA.

    Args:
        model_name: HuggingFace model id or local path.
        tokenizer: Corresponding tokenizer (used for encoding fallback).
        lora_rank: LoRA rank for ``max_lora_rank`` vLLM setting.
        tensor_parallel_size: GPUs to use for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        dtype: Model dtype (``"bfloat16"``, ``"float16"``, ``"auto"``).
        max_model_len: Optional max sequence length override.
    """

    def __init__(
        self,
        model_name: str,
        tokenizer,
        *,
        lora_rank: int = 16,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.35,
        dtype: str = "bfloat16",
        max_model_len: int | None = None,
    ):
        self._adapter_path: str | None = None
        self.tokenizer = tokenizer

        # vLLM is an optional dependency
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install it with: pip install vllm"
            )

        # Ensure left-padding for decoder-only models
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        # Build vLLM engine kwargs
        engine_kwargs: dict = {
            "model": model_name,
            "enable_lora": True,
            "max_lora_rank": lora_rank,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": True,
            # eager mode avoids CUDA graph memory overhead (~2-4 GB)
            "enforce_eager": True,
        }
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = max_model_len

        logger.info(
            "Initializing vLLM engine: model=%s lora_rank=%d tp=%d gpu_mem=%.2f",
            model_name, lora_rank, tensor_parallel_size, gpu_memory_utilization,
        )
        self.llm = LLM(**engine_kwargs)
        logger.info("vLLM engine ready")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Generate ``n_samples`` responses per prompt via vLLM.

        Logprobs are requested from vLLM directly (``SamplingParams.logprobs=1``),
        avoiding an extra forward pass.
        """
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        K = req.n_samples
        N = len(req.prompts)

        # ---- 1. Duplicate prompts: each prompt repeated K times -------
        flat_prompts: list[str] = []
        for p in req.prompts:
            flat_prompts.extend([p] * K)

        # ---- 2. Sampling params --------------------------------------
        sampling_params = SamplingParams(
            n=1,  # we duplicate prompts externally
            temperature=req.temperature if req.temperature > 0 else 0.0,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            logprobs=1,  # return logprob of the chosen token
        )

        # ---- 3. LoRA request (if adapter loaded) ----------------------
        lora_request = None
        if self._adapter_path is not None:
            lora_request = LoRARequest("policy", 1, self._adapter_path)

        # ---- 4. Generate ---------------------------------------------
        outputs = self.llm.generate(
            flat_prompts, sampling_params, lora_request=lora_request
        )

        # ---- 5. Extract trajectories ---------------------------------
        trajectories: list[Trajectory] = []

        for idx, output in enumerate(outputs):
            completion = output.outputs[0]  # n=1
            response_text = completion.text.strip()
            token_ids = completion.token_ids or []

            # Extract per-token logprobs
            logprobs: list[float] = []
            if completion.logprobs is not None and token_ids:
                for lp_dict, tid in zip(completion.logprobs, token_ids):
                    if tid in lp_dict:
                        logprobs.append(lp_dict[tid].logprob)
                    else:
                        # Fallback: logprob not available for this token
                        logprobs.append(-float("inf"))

            # Determine the owning task (prompts were duplicated K times)
            task = req.tasks[idx // K]
            prompt = req.prompts[idx // K]

            trajectories.append(
                Trajectory.from_single_response(
                    task_id=task.task_id,
                    prompt=prompt,
                    response=response_text,
                    token_ids=token_ids,
                    logprobs=logprobs,
                )
            )

        logger.debug(
            "vLLM rollout: %d prompts × %d samples = %d trajectories",
            N, K, len(trajectories),
        )
        return RolloutResp(
            batch_id=req.batch_id,
            trajectories=trajectories,
            policy_version=req.policy_version,
        )

    def reload_adapter(self, path: str) -> None:
        """Point vLLM to updated LoRA adapter weights.

        The adapter is loaded on the next ``generate()`` call via
        ``LoRARequest`` — no engine restart is needed.
        """
        self._adapter_path = path
        logger.info("vLLM adapter path updated: %s", path)
