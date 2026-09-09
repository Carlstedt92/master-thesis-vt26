#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 00:45:00
#SBATCH -p berzelius-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# Phase 2 (CPU) of the BACE admet_benchmark rerun -- fits MLP/RF probes for
# all 7 models on the newly-cached 5-seed BACE embeddings.

LABELS="MIXED_GAT4HEADS_TEST_60EP,MIXED_GAT16HEADS_TEST_60EP,WEIGHT_DECAY_FIX_TEST_60EP,NODE_RECON_WARMUP_TEST_60EP_EPOCH20,NODE_RECON_WARMUP_TEST_60EP_FINAL,NODE_RECON_EXTFEAT_WARMUP_TEST_60EP_EPOCH20,NODE_RECON_EXTFEAT_WARMUP_TEST_60EP_FINAL"

PYTHONUNBUFFERED=1 uv run python eval_phase2_fit_probes.py \
  --models "${LABELS}" \
  --device cpu \
  --datasets bace \
  --cache-dir embedding_cache \
  --allow-partial-results

echo "PHASE 2: BACE RERUN COMPLETE (7 MODELS)"
