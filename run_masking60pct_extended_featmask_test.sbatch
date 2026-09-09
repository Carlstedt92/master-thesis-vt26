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

# Feature-dropout test: same as masking60pct_extended_features_test (26-dim
# node features, 60% node-mask ratio, 60 epochs, 8-GPU DDP) except
# feature_mask_ratio 0 -> 0.15 -- tests whether generic (untargeted) feature-
# column dropout on local views affects the extended-features downstream
# regression.

SCRATCH_DATA_DIR=/scratch/local/precomputed_9M_data_extended_mask60_fm
mkdir -p "${SCRATCH_DATA_DIR}"
echo "Staging extended-feature 9M-shard precomputed dataset to ${SCRATCH_DATA_DIR}..."
time ls /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data_extended/*.pt | xargs -n 1 -P 8 cp -t "${SCRATCH_DATA_DIR}/"
cp /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data_extended/metadata.json "${SCRATCH_DATA_DIR}/"
echo "Staging complete."

uv run torchrun --standalone --nproc_per_node=8 train.py --config configs/masking60pct_extended_features_featmask_test.json5
