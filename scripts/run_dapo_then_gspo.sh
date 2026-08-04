#!/usr/bin/env bash
set -euo pipefail

cd /root/rlite-work

python -m rlite.ray_train \
  --config configs/remote_ray_dapo_qwen1.5b.yaml \
  > /root/rlite-logs/ray-dapo-qwen1.5b-seed42/console.log 2>&1

python -m rlite.ray_train \
  --config configs/remote_ray_gspo_qwen1.5b.yaml \
  > /root/rlite-logs/ray-gspo-qwen1.5b-seed42/console.log 2>&1
