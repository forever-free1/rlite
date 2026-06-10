#!/bin/bash
### GSM8K LoRA-GRPO training with HF rollout
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================"
echo "rlite — GSM8K LoRA-GRPO (HF rollout)"
echo "============================================"

python -m rlite.train --config configs/gsm8k_grpo_lora_hf.yaml
