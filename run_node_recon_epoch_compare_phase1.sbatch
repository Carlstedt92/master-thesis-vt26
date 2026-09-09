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

# Phase 1 (GPU) of the node-recon epoch-20-vs-final comparison, both models
# (NODE_RECON_WARMUP_TEST_60EP, NODE_RECON_EXTFEAT_WARMUP_TEST_60EP) x both
# checkpoints (checkpoint_epoch_20.pth, final_model.pth) = 4 combos, each a
# synthetic model-name label (--checkpoint-path bypasses model-name-based
# checkpoint resolution entirely -- see resolve_checkpoint_path -- so the
# label is purely an output-grouping name under embedding_cache/<label>/).
# Matches run_eval_mlp_rf_phase1.sbatch's resource shape (GPU, not CPU --
# embedding extraction is per-molecule encoder forward passes, which is what
# --gres=gpu:1 is for; --device cpu on the shared login node was the wrong
# resource for this, not just an unlucky slow run).
# PYTHONUNBUFFERED=1 so the log is never lost even if this gets killed.

declare -A COMBOS=(
  ["NODE_RECON_WARMUP_TEST_60EP_EPOCH20"]="models/NODE_RECON_WARMUP_TEST_60EP/checkpoints/checkpoint_epoch_20.pth"
  ["NODE_RECON_WARMUP_TEST_60EP_FINAL"]="models/NODE_RECON_WARMUP_TEST_60EP/checkpoints/final_model.pth"
  ["NODE_RECON_EXTFEAT_WARMUP_TEST_60EP_EPOCH20"]="models/NODE_RECON_EXTFEAT_WARMUP_TEST_60EP/checkpoints/checkpoint_epoch_20.pth"
  ["NODE_RECON_EXTFEAT_WARMUP_TEST_60EP_FINAL"]="models/NODE_RECON_EXTFEAT_WARMUP_TEST_60EP/checkpoints/final_model.pth"
)

for label in "${!COMBOS[@]}"; do
  ckpt="${COMBOS[$label]}"
  echo "############################################################"
  echo "### ${label}  <-  ${ckpt}"
  echo "############################################################"

  PYTHONUNBUFFERED=1 uv run python eval_phase1_extract_embeddings.py \
    --models "${label}" \
    --checkpoint-path "${ckpt}" \
    --device cuda \
    --datasets lipo,bace,tox21,bbb_martins,herg,ames \
    --cache-dir embedding_cache
done

echo "PHASE 1: ALL 4 COMBOS COMPLETE"
