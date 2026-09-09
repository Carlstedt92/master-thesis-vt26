#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# Phase 2 of the split finetuning eval, TDC ADMET datasets only, for
# MIXED_GAT16HEADS_TEST_60EP -- loads the checkpoints saved by
# run_finetune_phase1_mixed16_tdc.sbatch and evaluates them via the same
# MLP/RF methodology (evaluation/mlp_rf.py) used for the frozen-embedding
# results, so the two are directly comparable. Submitted with
# --dependency=afterok on phase 1's job id -- won't start until phase 1
# actually succeeds.
# PYTHONUNBUFFERED=1 so the log is never lost even if this gets killed.

PYTHONUNBUFFERED=1 uv run python finetune_phase2_eval.py \
  --models MIXED_GAT16HEADS_TEST_60EP \
  --device cuda \
  --datasets bbb_martins,herg,ames \
  --allow-partial-results
