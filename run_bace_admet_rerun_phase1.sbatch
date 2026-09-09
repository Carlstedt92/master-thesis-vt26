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

# Re-extract BACE embeddings (5 seeds, admet_benchmark protocol) for the 7
# models already in the node-recon comparison table, now that BACE has real
# seed variance instead of DeepChem's inert scaffold splitter. Group A uses
# default checkpoint resolution (best_model.pth); Group B are synthetic
# epoch20/final labels needing an explicit --checkpoint-path each.

echo "=== Group A: default checkpoint resolution ==="
PYTHONUNBUFFERED=1 uv run python eval_phase1_extract_embeddings.py \
  --models MIXED_GAT4HEADS_TEST_60EP,MIXED_GAT16HEADS_TEST_60EP,WEIGHT_DECAY_FIX_TEST_60EP \
  --device cuda \
  --datasets bace \
  --cache-dir embedding_cache

declare -A GROUP_B=(
  ["NODE_RECON_WARMUP_TEST_60EP_EPOCH20"]="models/NODE_RECON_WARMUP_TEST_60EP/checkpoints/checkpoint_epoch_20.pth"
  ["NODE_RECON_WARMUP_TEST_60EP_FINAL"]="models/NODE_RECON_WARMUP_TEST_60EP/checkpoints/final_model.pth"
  ["NODE_RECON_EXTFEAT_WARMUP_TEST_60EP_EPOCH20"]="models/NODE_RECON_EXTFEAT_WARMUP_TEST_60EP/checkpoints/checkpoint_epoch_20.pth"
  ["NODE_RECON_EXTFEAT_WARMUP_TEST_60EP_FINAL"]="models/NODE_RECON_EXTFEAT_WARMUP_TEST_60EP/checkpoints/final_model.pth"
)
for label in "${!GROUP_B[@]}"; do
  ckpt="${GROUP_B[$label]}"
  echo "=== Group B: ${label} <- ${ckpt} ==="
  PYTHONUNBUFFERED=1 uv run python eval_phase1_extract_embeddings.py \
    --models "${label}" \
    --checkpoint-path "${ckpt}" \
    --device cuda \
    --datasets bace \
    --cache-dir embedding_cache
done

echo "PHASE 1: BACE RERUN COMPLETE (7 MODELS)"
