#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# Rerun of finetune_hyperopt_search.py on AMES with the widened search space
# (job 17450158's run showed a modest but real gain over the fixed-recipe
# frozen-embedding probe; this tests whether widening lr/weight_decay floors,
# adding dropout + LR schedule as search dimensions, and averaging the search
# objective across ALL 5 official seeds instead of 3 pushes that gain
# further). Search still never touches test.

PYTHONUNBUFFERED=1 uv run python finetune_hyperopt_search.py \
  --model NODE_RECON_WARMUP_TEST_60EP \
  --checkpoint-name final_model.pth \
  --dataset ames \
  --device cuda \
  --max-evals 25 \
  --search-seeds 1,2,3,4,5
