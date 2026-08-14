# Prefix-aware grouped admission benchmark

## Environment

- GPU: NVIDIA RTX 3090 24GB (one GPU per measured process)
- Model: Qwen2.5-1.5B-Instruct, FP16
- Runtime: PyTorch 2.5.1+cu124, vLLM 0.6.6.post1
- vLLM: eager execution, Automatic Prefix Caching enabled
- Every measurement starts a fresh process; model initialization is excluded
  from `generation_seconds`.

Model weights and package caches live under `/hy-tmp`. Code, small datasets,
and raw logs live on the system disk under `/root/rlite-work`,
`/root/rlite-data`, and `/root/rlite-logs`.

## Modes

- `duplicate`: legacy baseline, K independent requests with `n=1`.
- `native`: one request with `SamplingParams(n=K)`.
- `leader`: K independent requests, but followers enter after the leader emits
  its first token.

## Results

Four prompts and K=8 were used in both Qwen measurements.

| Prompt / output cap | duplicate | native | leader | leader vs native |
|---|---:|---:|---:|---:|
| 1024 / 128 | 4.305 s | 4.100 s | 4.061 s | 1.0% faster |
| 2048 / 32, run 1 | 2.127 s | 2.032 s | 1.733 s | 14.7% faster |
| 2048 / 32, run 2 | — | 1.934 s | 1.782 s | 7.9% faster |

In the 2048/32 case, native averaged 1.983 seconds and leader averaged 1.758
seconds across the two independent runs: an 11.3% reduction. Every follower
reported 1872 cached prompt tokens; the uncached remainder is consistent with
full-block-only cache reuse plus prompt re-tokenization effects.

A small OPT-125M smoke case (128-token prompt, K=4, 16-token output) regressed
from 0.323 seconds native to 0.360 seconds leader. This validates that the
extra admission round is harmful when prefill is cheap.

## Decision

`native` remains the default. `leader` is retained as an explicit experimental
mode for prefill-dominated rollouts: long prompts, K at least four, and outputs
much shorter than prompts. It should not be described as a universal vLLM
optimization. The main production improvement over the original framework is
switching from duplicated `n=1` requests to native `n=K`; leader admission is a
narrow additional optimization with a measured workload-dependent benefit.

Raw remote logs:

```text
/root/rlite-logs/benchmarks/qwen-duplicate-p1024-k8-clean.log
/root/rlite-logs/benchmarks/qwen-native-p1024-k8-clean.log
/root/rlite-logs/benchmarks/qwen-leader-p1024-k8-clean.log
/root/rlite-logs/benchmarks/qwen-duplicate-p2048-k8.log
/root/rlite-logs/benchmarks/qwen-native-p2048-k8.log
/root/rlite-logs/benchmarks/qwen-leader-p2048-k8.log
/root/rlite-logs/benchmarks/qwen-native-p2048-k8-repeat.log
/root/rlite-logs/benchmarks/qwen-leader-p2048-k8-repeat.log
```
