#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:8
#SBATCH --constraint=fat
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# Extended-features test: full 9M-shard dataset (26-dim node features --
# Gasteiger partial charge + topological eccentricity added), 60 epochs,
# 8-GPU DDP (torchrun --standalone). Single un-segmented job. Same setup as
# weight_decay_fix_test.json5 (already confirmed: loss improves through
# epoch ~44, encoder_embedding_std stable through epoch ~50, BatchNorm gamma
# stays ~0.997) -- this run checks that the same schedule dynamics hold with
# the wider feature set, not a downstream-performance test.

SCRATCH_DATA_DIR=/scratch/local/precomputed_9M_data_extended
mkdir -p "${SCRATCH_DATA_DIR}"
echo "Staging extended-feature 9M-shard precomputed dataset to ${SCRATCH_DATA_DIR}..."
time ls /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data_extended/*.pt | xargs -n 1 -P 8 cp -t "${SCRATCH_DATA_DIR}/"
cp /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data_extended/metadata.json "${SCRATCH_DATA_DIR}/"
echo "Staging complete."

uv run torchrun --standalone --nproc_per_node=8 train.py --config configs/extended_features_test.json5
