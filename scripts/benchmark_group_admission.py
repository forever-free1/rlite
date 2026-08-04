"""Benchmark one vLLM grouped-admission mode in a fresh process.

Run native and leader separately so both start with a cold prefix cache. The
reported generation time excludes model construction and weight loading.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from transformers import AutoTokenizer

from rlite.core.rollout_types import RolloutReq
from rlite.core.types import Task
from rlite.rollout.vllm_engine import VLLMRolloutEngine


def make_prompt(tokenizer, target_tokens: int, prompt_idx: int) -> str:
    seed = f"Problem {prompt_idx}: Work carefully and show your reasoning. "
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    filler_ids = tokenizer.encode(
        "We need solve the following mathematical problem step by step. ",
        add_special_tokens=False,
    )
    ids = list(seed_ids)
    while len(ids) < target_tokens:
        ids.extend(filler_ids)
    return tokenizer.decode(ids[:target_tokens])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--mode", choices=("duplicate", "native", "leader"), required=True
    )
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        make_prompt(tokenizer, args.prompt_tokens, i)
        for i in range(args.prompts)
    ]
    tasks = [Task(task_id=f"bench-{i}") for i in range(args.prompts)]
    engine = VLLMRolloutEngine(
        args.model,
        tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="float16",
        max_model_len=args.prompt_tokens + args.max_tokens + 64,
        enable_prefix_caching=True,
        group_admission=args.mode,
        enable_lora=False,
    )
    request = RolloutReq(
        batch_id=f"bench-{args.mode}-{time.time_ns()}",
        tasks=tasks,
        prompts=prompts,
        n_samples=args.samples,
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.max_tokens,
        policy_version=0,
    )
    response = engine.generate(request)
    output_lengths = [
        len(trajectory.steps[0].token_ids or [])
        for trajectory in response.trajectories
    ]
    elapsed = response.metadata["generation_seconds"]
    result = {
        "mode": args.mode,
        "model": args.model,
        "prompt_tokens": args.prompt_tokens,
        "prompts": args.prompts,
        "samples": args.samples,
        "max_tokens": args.max_tokens,
        "trajectories": len(response.trajectories),
        "generation_seconds": elapsed,
        "output_tokens": sum(output_lengths),
        "output_tokens_per_second": sum(output_lengths) / elapsed,
        "mean_output_tokens": statistics.mean(output_lengths),
        "cached_prompt_tokens": response.metadata["cached_prompt_tokens"],
    }
    print("RLITE_BENCHMARK=" + json.dumps(result, sort_keys=True))
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
