#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# Phase 1 of the split finetuning eval: GPU-only encoder finetuning on
# LIPO/BACE/Tox21, 5 seeds each, saves finetuned checkpoints to
# models/KHOP_GAT3HEADS_TEST_60EP/finetune_checkpoints/. No CPU-heavy
# sklearn work in this job (that's phase 2) -- avoids the interleaved
# GPU-idle pattern suspected of triggering the earlier job's cancellation.
# PYTHONUNBUFFERED=1 so the log is never lost even if this gets killed again.

PYTHONUNBUFFERED=1 uv run python finetune_phase1_train.py \
  --models KHOP_GAT3HEADS_TEST_60EP \
  --checkpoint-name best_model.pth \
  --device cuda \
  --finetune-epochs 30 \
  --batch-size 256
