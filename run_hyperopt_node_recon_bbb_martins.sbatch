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

# Third real hyperopt search run of finetune_hyperopt_search.py, this time on
# BBB_Martins (~2000 compounds, between hERG's ~650 and AMES's ~7300). Uses
# the same widened search space as the rerun hERG/AMES jobs (lr/weight_decay
# floors lowered, dropout + LR schedule added as search dimensions, epoch
# choices extended to 75) and averages the search objective across ALL 5
# official seeds. Search still never touches test.

PYTHONUNBUFFERED=1 uv run python finetune_hyperopt_search.py \
  --model NODE_RECON_WARMUP_TEST_60EP \
  --checkpoint-name final_model.pth \
  --dataset bbb_martins \
  --device cuda \
  --max-evals 25 \
  --search-seeds 1,2,3,4,5
