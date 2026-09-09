#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 02:00:00
#SBATCH -p berzelius-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# Phase 2 of the split MLP/RF eval: CPU-only probe fitting (MLP head + RF,
# RandomForest uses n_jobs=-1 so --cpus-per-task=8 is used) on the
# embeddings cached by phase 1, plus Morgan fingerprints. No GPU requested
# -- this is the CPU-bound part that made the combined job look GPU-idle
# and get auto-cancelled (job 17376620).
# PYTHONUNBUFFERED=1 so the log is never lost even if this gets killed again.

MODELS="${1:?Usage: sbatch run_eval_mlp_rf_phase2.sbatch <comma-separated-model-names> [datasets]}"
DATASETS="${2:-}"

EXTRA_ARGS=()
if [ -n "${DATASETS}" ]; then
  EXTRA_ARGS+=(--datasets "${DATASETS}")
fi

PYTHONUNBUFFERED=1 uv run python eval_phase2_fit_probes.py \
  --models "${MODELS}" \
  --device cpu \
  --allow-partial-results \
  --cache-dir embedding_cache \
  --output eval_many_models_mlp_rf_results.json \
  "${EXTRA_ARGS[@]}"
