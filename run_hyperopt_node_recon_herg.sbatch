#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# Rerun of finetune_hyperopt_search.py on hERG with the widened search space
# (job 17450028's first run underperformed the fixed-recipe frozen-embedding
# probe -- best lr/weight_decay both hugged the old search floor, which
# prompted lowering both floors by an order of magnitude) and now also
# searching dropout (encoder + head) and LR schedule (constant vs cosine),
# plus a wider epoch choice (up to 75). Averages the search objective across
# ALL 5 official seeds (not just 3) for the most robust possible signal on
# this small (~650 compound) dataset. Search still never touches test.

PYTHONUNBUFFERED=1 uv run python finetune_hyperopt_search.py \
  --model NODE_RECON_WARMUP_TEST_60EP \
  --checkpoint-name final_model.pth \
  --dataset herg \
  --device cuda \
  --max-evals 25 \
  --search-seeds 1,2,3,4,5
