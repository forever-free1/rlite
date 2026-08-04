"""vLLM rollout engine — high-throughput generation with LoRA support.

Uses the vLLM offline ``LLM`` API with ``enable_lora=True``.  After each
training step the LoRA adapter is saved to a stable temp directory and
vLLM picks it up via ``LoRARequest`` — no engine restart required.
"""

from __future__ import annotations

import time

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
        enable_prefix_caching: bool = True,
        group_admission: str = "native",
        enable_lora: bool = True,
    ):
        self._adapter_path: str | None = None
        self._adapter_id = 0
        self.tokenizer = tokenizer
        if group_admission not in {"native", "leader", "duplicate"}:
            raise ValueError(
                "group_admission must be 'native', 'leader', or 'duplicate'"
            )
        self.group_admission = group_admission

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
            "enable_lora": enable_lora,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": True,
            # eager mode avoids CUDA graph memory overhead (~2-4 GB)
            "enforce_eager": True,
            "enable_prefix_caching": enable_prefix_caching,
        }
        if enable_lora:
            engine_kwargs["max_lora_rank"] = lora_rank
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

        # Native admission lets vLLM own n=K scheduling. Leader admission
        # submits one request per prompt first and releases its followers as
        # soon as that leader produces a token, making cached prompt blocks
        # visible before the remaining group members enter the scheduler.
        sampling_params = SamplingParams(
            n=K if self.group_admission == "native" else 1,
            temperature=req.temperature if req.temperature > 0 else 0.0,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            logprobs=1,  # return logprob of the chosen token
        )

        # ---- 3. LoRA request (if adapter loaded) ----------------------
        lora_request = None
        if self._adapter_path is not None:
            lora_request = LoRARequest(
                f"policy_v{self._adapter_id}", self._adapter_id, self._adapter_path
            )

        started = time.perf_counter()
        if self.group_admission == "leader" and K > 1:
            grouped_outputs = self._generate_leader(
                req, sampling_params, lora_request
            )
            cached_tokens = self._last_cached_tokens
        elif self.group_admission == "duplicate":
            flat_prompts = [
                prompt for prompt in req.prompts for _ in range(K)
            ]
            outputs = self.llm.generate(
                flat_prompts, sampling_params, lora_request=lora_request,
                use_tqdm=False,
            )
            grouped_outputs = [
                [outputs[prompt_idx * K + sample_idx].outputs[0]
                 for sample_idx in range(K)]
                for prompt_idx in range(N)
            ]
            cached_tokens = [output.num_cached_tokens or 0 for output in outputs]
        else:
            outputs = self.llm.generate(
                req.prompts, sampling_params, lora_request=lora_request,
                use_tqdm=False,
            )
            grouped_outputs = [output.outputs for output in outputs]
            cached_tokens = [output.num_cached_tokens or 0 for output in outputs]

        # ---- 5. Extract trajectories ---------------------------------
        trajectories: list[Trajectory] = []

        for prompt_idx, completions in enumerate(grouped_outputs):
            if len(completions) != K:
                raise RuntimeError(
                    f"vLLM returned {len(completions)} samples for prompt "
                    f"{prompt_idx}; expected {K}"
                )
            for completion in completions:
                response_text = completion.text.strip()
                token_ids = completion.token_ids or []

            # Extract per-token logprobs
                logprobs: list[float] = []
                if completion.logprobs is not None and token_ids:
                    for lp_dict, tid in zip(completion.logprobs, token_ids):
                        if tid in lp_dict:
                            logprobs.append(lp_dict[tid].logprob)
                        else:
                            logprobs.append(-float("inf"))

            # Determine the owning task (prompts were duplicated K times)
                task = req.tasks[prompt_idx]
                prompt = req.prompts[prompt_idx]

                trajectories.append(
                    Trajectory.from_single_response(
                        task_id=task.task_id, prompt=prompt,
                        response=response_text, token_ids=token_ids,
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
            metadata={
                "group_admission": self.group_admission,
                "generation_seconds": time.perf_counter() - started,
                "cached_prompt_tokens": cached_tokens,
            },
        )

    def _generate_leader(self, req, sampling_params, lora_request):
        """Run leader-gated admission on vLLM's synchronous engine loop."""
        engine = self.llm.llm_engine
        K = req.n_samples
        completed: dict[tuple[int, int], object] = {}
        released: set[int] = set()
        request_map: dict[str, tuple[int, int]] = {}
        cached_tokens: dict[tuple[int, int], int] = {}

        def add(prompt_idx: int, sample_idx: int) -> None:
            request_id = (
                f"rlite-{req.batch_id}-p{prompt_idx}-s{sample_idx}-"
                f"v{req.policy_version}"
            )
            request_map[request_id] = (prompt_idx, sample_idx)
            engine.add_request(
                request_id, req.prompts[prompt_idx], sampling_params,
                lora_request=lora_request,
            )

        for prompt_idx in range(len(req.prompts)):
            add(prompt_idx, 0)

        while engine.has_unfinished_requests():
            for output in engine.step():
                prompt_idx, sample_idx = request_map[output.request_id]
                if sample_idx == 0 and prompt_idx not in released:
                    has_first_token = bool(
                        output.outputs and output.outputs[0].token_ids
                    )
                    if has_first_token:
                        released.add(prompt_idx)
                        for follower_idx in range(1, K):
                            add(prompt_idx, follower_idx)
                if output.finished:
                    completed[(prompt_idx, sample_idx)] = output.outputs[0]
                    cached_tokens[(prompt_idx, sample_idx)] = (
                        output.num_cached_tokens or 0
                    )

        expected = len(req.prompts) * K
        if len(completed) != expected:
            raise RuntimeError(
                f"leader admission completed {len(completed)}/{expected} requests"
            )
        self._last_cached_tokens = [
            cached_tokens[(prompt_idx, sample_idx)]
            for prompt_idx in range(len(req.prompts))
            for sample_idx in range(K)
        ]
        return [
            [completed[(prompt_idx, sample_idx)] for sample_idx in range(K)]
            for prompt_idx in range(len(req.prompts))
        ]

    def reload_adapter(self, path: str) -> None:
        """Point vLLM to updated LoRA adapter weights.

        The adapter is loaded on the next ``generate()`` call via
        ``LoRARequest`` — no engine restart is needed.
        """
        self._adapter_path = path
        self._adapter_id += 1
        reset_cache = getattr(self.llm.llm_engine, "reset_prefix_cache", None)
        if reset_cache is not None:
            reset_cache()
        logger.info("vLLM adapter path updated: %s", path)
