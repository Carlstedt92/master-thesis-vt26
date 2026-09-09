#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 01:00:00
#SBATCH -p berzelius-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# Phase 2 (CPU) of the node-recon epoch-20-vs-final comparison -- same 4
# combos as phase 1, fits MLP/RF probes on the cached embeddings. Runs on
# the dedicated berzelius-cpu partition instead of the shared login node
# (RandomForest uses n_jobs=-1, so more cores helps here specifically --
# --cpus-per-task=16, double run_eval_mlp_rf_phase2.sbatch's usual 8, since
# this job covers 4 combos back to back instead of 1).
# PYTHONUNBUFFERED=1 so the log is never lost even if this gets killed.

LABELS="NODE_RECON_WARMUP_TEST_60EP_EPOCH20,NODE_RECON_WARMUP_TEST_60EP_FINAL,NODE_RECON_EXTFEAT_WARMUP_TEST_60EP_EPOCH20,NODE_RECON_EXTFEAT_WARMUP_TEST_60EP_FINAL"

PYTHONUNBUFFERED=1 uv run python eval_phase2_fit_probes.py \
  --models "${LABELS}" \
  --device cpu \
  --datasets lipo,bace,tox21,bbb_martins,herg,ames \
  --cache-dir embedding_cache \
  --allow-partial-results

echo "PHASE 2: ALL 4 COMBOS COMPLETE"
