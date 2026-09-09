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

# Phase 2 of the split finetuning eval: loads the checkpoints saved by phase 1
# and evaluates them via MLP/RF (evaluation/mlp_rf.py), matching the exact
# methodology used for every frozen-embedding comparison this session.
# PYTHONUNBUFFERED=1 so the log is never lost even if this gets killed again.

PYTHONUNBUFFERED=1 uv run python finetune_phase2_eval.py \
  --models KHOP_GAT3HEADS_TEST_60EP \
  --device cuda \
  --allow-partial-results
