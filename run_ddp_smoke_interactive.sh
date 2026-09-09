#!/bin/bash
set -euo pipefail
cd /home/x_emcar/master-thesis-repo
echo "=== nvidia-smi ==="
nvidia-smi -L
echo "=== launching torchrun ==="
uv run torchrun --standalone --nproc_per_node=2 train.py --config configs/ddp_smoke_test.json5
