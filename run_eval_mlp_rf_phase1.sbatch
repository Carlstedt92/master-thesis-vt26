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

# Phase 1 of the split MLP/RF eval: GPU-only embedding extraction (see
# eval_phase1_extract_embeddings.py for why -- job 17376620 interleaved this
# with CPU-only fingerprint/RF work in one job and got auto-cancelled after
# ~64 min for low aggregate GPU utilization). Saves cached embeddings to
# embedding_cache/, consumed by phase 2's CPU-only job.
# PYTHONUNBUFFERED=1 so the log is never lost even if this gets killed again.

# Optional: a checkpoint filename (relative to each model's checkpoints/ dir)
# to use for every model in this run, e.g. final_model.pth. Without this,
# eval_phase1_extract_embeddings.py's resolve_checkpoint_path() defaults to
# best_model.pth (falls back to best_online_eval_model.pth) -- matches
# run_eval_mlp_rf.sbatch's behavior exactly.
MODELS="${1:?Usage: sbatch run_eval_mlp_rf_phase1.sbatch <comma-separated-model-names> [checkpoint-name] [datasets]}"
CHECKPOINT_NAME="${2:-}"
DATASETS="${3:-}"

EXTRA_ARGS=()
if [ -n "${CHECKPOINT_NAME}" ]; then
  EXTRA_ARGS+=(--checkpoint-name "${CHECKPOINT_NAME}")
fi
if [ -n "${DATASETS}" ]; then
  EXTRA_ARGS+=(--datasets "${DATASETS}")
fi

PYTHONUNBUFFERED=1 uv run python eval_phase1_extract_embeddings.py \
  --models "${MODELS}" \
  --device cuda \
  --cache-dir embedding_cache \
  "${EXTRA_ARGS[@]}"
