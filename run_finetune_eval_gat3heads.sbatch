#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# Finetuning eval for KHOP_GAT3HEADS_TEST_60EP: finetunes the full encoder
# (not a frozen probe) on LIPO/BACE/Tox21 separately, 5 seeds each, then
# evaluates via KNN/linear-probe/RF on the finetuned embeddings. Explicit
# --checkpoint-name best_model.pth to match the SSL-val-loss-selected
# checkpoint convention used throughout this session's other comparisons
# (this script's own default checkpoint priority is the opposite -- it tries
# best_online_eval_model.pth first).

uv run python finetune_eval_many_models.py \
  --models KHOP_GAT3HEADS_TEST_60EP \
  --checkpoint-name best_model.pth \
  --device cuda \
  --finetune-epochs 30 \
  --batch-size 256
