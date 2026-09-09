#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# One-off validation: run both phases of the new split eval on a SINGLE
# model/dataset/seed (KHOP_200EP_EPOCH10, LIPO, seed=0), which already has a
# known-correct result from the original combined eval_many_models_mlp_rf.py
# run to compare against. GPU for phase 1, CPU for phase 2 (fine to share one
# job for this short validation run -- not the real usage pattern).

rm -rf /tmp/eval_cache_validation

PYTHONUNBUFFERED=1 uv run python eval_phase1_extract_embeddings.py \
  --models PHASESPLIT_VALIDATION \
  --datasets lipo \
  --random-split-seeds 0 \
  --device cuda \
  --cache-dir /tmp/eval_cache_validation

PYTHONUNBUFFERED=1 uv run python eval_phase2_fit_probes.py \
  --models PHASESPLIT_VALIDATION \
  --datasets lipo \
  --random-split-seeds 0 \
  --device cpu \
  --cache-dir /tmp/eval_cache_validation \
  --output /tmp/eval_cache_validation_results.json \
  --allow-partial-results
