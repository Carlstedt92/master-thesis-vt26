#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# Phase 1 of the split finetuning eval, TDC ADMET datasets only (bbb_martins/
# herg/ames), for MIXED_GAT16HEADS_TEST_60EP -- mirrors
# run_finetune_phase1_gat3heads.sbatch's structure exactly, just pointed at
# the 3 newly-added TDC datasets instead of lipo/bace/tox21. finetune_phase1_train.py
# always uses TDC's official 5 leaderboard seeds (1-5) for these three,
# regardless of --random-split-seeds, so nothing to pass for that here.
# Estimated cost: ~90 min GPU time (hERG/BBB_Martins/AMES x 5 seeds x 30
# finetune epochs, mini-batched) -- 2h wall-clock leaves margin.
# PYTHONUNBUFFERED=1 so the log is never lost even if this gets killed.

PYTHONUNBUFFERED=1 uv run python finetune_phase1_train.py \
  --models MIXED_GAT16HEADS_TEST_60EP \
  --checkpoint-name final_model.pth \
  --device cuda \
  --finetune-epochs 30 \
  --batch-size 256 \
  --datasets bbb_martins,herg,ames
